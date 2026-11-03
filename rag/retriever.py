"""검색 — 챗봇(app)과 평가 하네스(eval)가 같은 검색기를 쓴다.

- LangChain Chroma 로 디스크의 컬렉션을 열어 조회만 한다. 컬렉션이 없으면 만들지 않고 오류를 낸다
  (챗봇 계층은 인덱스에 쓰지 않는다). 임베딩은 질문 벡터화에만 쓴다
- 컬렉션 메타데이터의 임베딩 모델이 현재 설정과 다르면 즉시 오류를 낸다 (오류 없이 품질만 무너지는 것을 막음)
- 필터는 Chroma where 조건으로 바꾼다. 기간은 published_ymd(YYYYMMDD 정수) 비교라서
  기간 필터를 걸면 발행일 미상(0) 청크는 빠진다
- 점수는 코사인 유사도(1 - 거리). 높을수록 유사하다. 거절 판단(유사도 임계값)은 체인이 한다
- 같은 문서의 청크가 상위를 채우지 않도록 문서당 청크 수를 제한한다 (기본 1개). 이 때문에 후보를
  k 보다 넉넉히(fetch_k) 가져온 뒤 문서 단위로 고른다
- 같은 날 같은 제목인 기사는 하나만 남긴다 (기본). 짧은 요약 기사와 긴 전문 기사처럼 길이 차이로
  본문 유사도가 낮아 중복 처리를 빠져나간 경우다 (verify 샘플: '서울 골목상권 외식업 매출 14% 증가'
  두 건이 2·3위를 차지). 인덱스를 다시 만들지 않고 검색 단계에서 다룬다
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

import chromadb
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from common.config import get_settings
from common.embeddings import get_embedding_model
from common.schema import DocType

FETCH_MULTIPLIER = 4  # 문서당 청크 제한으로 걸러질 것을 감안한 후보 배수
_NON_WORD = re.compile(r"[\W_]+")


def _title_key(metadata: dict[str, Any]) -> tuple[str, str] | None:
    title = _NON_WORD.sub("", str(metadata.get("title", "")))
    return (title, str(metadata.get("published_at", ""))[:10]) if title else None


@dataclass(frozen=True)
class SearchFilters:
    doc_types: tuple[DocType, ...] = ()
    date_from: date | None = None
    date_to: date | None = None
    scopes: tuple[str, ...] = ()  # 뉴스 범위: "seoul", "national"

    def to_where(self) -> dict[str, Any] | None:
        conditions: list[dict[str, Any]] = []
        if self.doc_types:
            conditions.append({"doc_type": {"$in": [t.value for t in self.doc_types]}})
        if self.scopes:
            conditions.append({"scope": {"$in": list(self.scopes)}})
        if self.date_from:
            conditions.append({"published_ymd": {"$gte": int(self.date_from.strftime("%Y%m%d"))}})
        if self.date_to:
            conditions.append({"published_ymd": {"$lte": int(self.date_to.strftime("%Y%m%d"))}})
        if not conditions:
            return None
        # Chroma 는 조건이 하나일 때 $and 로 감싸면 거부한다.
        return conditions[0] if len(conditions) == 1 else {"$and": conditions}


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    text: str
    score: float  # 코사인 유사도 (1 - 거리). 높을수록 유사
    metadata: dict[str, Any] = field(default_factory=dict)


class Retriever:
    def __init__(
        self,
        collection: str,
        *,
        chroma_dir: Path | None = None,
        embeddings: Embeddings | None = None,
        embedding_model_name: str | None = None,
        max_chunks_per_doc: int = 1,
        collapse_same_title: bool = True,
    ) -> None:
        if max_chunks_per_doc < 1:
            raise ValueError("max_chunks_per_doc 는 1 이상이어야 합니다")
        path = chroma_dir or get_settings().chroma_dir
        client = chromadb.PersistentClient(path=str(path))
        names = {c if isinstance(c, str) else c.name for c in client.list_collections()}
        if collection not in names:
            raise RuntimeError(f"컬렉션이 없습니다: {collection}. python -m indexer build 로 먼저 만드세요.")

        metadata = client.get_collection(collection).metadata or {}
        expected = embedding_model_name or get_settings().embedding_model
        if metadata.get("embedding_model") != expected:
            raise RuntimeError(
                f"컬렉션 {collection} 의 임베딩 모델({metadata.get('embedding_model')})이 "
                f"현재 설정({expected})과 다릅니다. EMBEDDING_MODEL 을 맞추거나 인덱스를 다시 만드세요."
            )
        self.collection = collection
        self.collection_metadata = dict(metadata)
        self.max_chunks_per_doc = max_chunks_per_doc
        self.collapse_same_title = collapse_same_title
        self._store = Chroma(
            collection_name=collection,
            embedding_function=embeddings or get_embedding_model(),
            client=client,
            create_collection_if_not_exists=False,
        )

    def retrieve(self, query: str, *, k: int, filters: SearchFilters | None = None) -> list[RetrievedChunk]:
        if k < 1:
            raise ValueError("k 는 1 이상이어야 합니다")
        if not query.strip():
            return []
        where = filters.to_where() if filters else None
        hits = self._store.similarity_search_with_score(query, k=k * FETCH_MULTIPLIER * self.max_chunks_per_doc,
                                                        filter=where)
        results: list[RetrievedChunk] = []
        per_doc: dict[str, int] = defaultdict(int)
        title_owner: dict[tuple[str, str], str] = {}
        for document, distance in hits:  # 거리 오름차순 = 유사도 내림차순
            meta = dict(document.metadata)
            doc_id = meta["doc_id"]
            if per_doc[doc_id] >= self.max_chunks_per_doc:
                continue
            key = _title_key(meta) if self.collapse_same_title else None
            if key is not None and title_owner.setdefault(key, doc_id) != doc_id:
                continue  # 같은 날 같은 제목의 다른 기사
            per_doc[doc_id] += 1
            results.append(RetrievedChunk(chunk_id=meta["chunk_id"], doc_id=doc_id, text=document.page_content,
                                          score=1.0 - distance, metadata=meta))
            if len(results) == k:
                break
        return results


@lru_cache(maxsize=4)
def get_retriever(collection: str | None = None) -> Retriever:
    """설정의 컬렉션(CHROMA_COLLECTION)을 기본으로 쓰는 검색기. 챗봇·평가에서 재사용한다."""
    return Retriever(collection or get_settings().chroma_collection)


def retrieve(
    query: str,
    *,
    k: int,
    filters: SearchFilters | None = None,
    collection: str | None = None,
) -> list[RetrievedChunk]:
    """collection 을 생략하면 설정의 CHROMA_COLLECTION 을 사용한다."""
    return get_retriever(collection).retrieve(query, k=k, filters=filters)
