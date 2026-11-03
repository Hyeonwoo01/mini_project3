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

하이브리드 검색 (mode="hybrid")
- 의미 검색만으로는 질문의 연도·기관명·수치를 놓친다 (기준선 실패 분석: '2027년도 예산안'에 2024년 예산
  기사, '중기부 예산'에 정부 전체 예산 기사). 원문 DB 의 키워드 색인(BM25)을 함께 조회해 보완한다
- 두 결과는 문서 단위 순위로 RRF(Reciprocal Rank Fusion) 결합한다: 점수 = Σ 1 / (RRF_K + 순위).
  점수 척도가 다른 두 검색을 정규화 없이 합칠 수 있다
- 반환 청크의 score 는 의미 검색과 같은 코사인 유사도다 (체인의 거절 임계값이 같은 의미를 갖도록).
  키워드 검색에서만 찾은 문서는 질문 벡터로 그 문서의 가장 유사한 청크를 골라 붙인다
- 문서 단위 결합이므로 문서당 청크 1개(max_chunks_per_doc=1)일 때만 쓴다
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field, replace
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

import chromadb
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from common.config import RETRIEVAL_MODES, get_settings
from common.docstore import open_db, search_keyword_index
from common.embeddings import get_embedding_model
from common.schema import DocType
from common.text import query_terms

FETCH_MULTIPLIER = 4  # 문서당 청크 제한으로 걸러질 것을 감안한 후보 배수
HYBRID_CANDIDATES = 50  # 하이브리드: 의미 검색·키워드 검색에서 각각 가져오는 문서 후보 수
RRF_K = 60  # RRF 상수. 원 논문(Cormack et al., 2009)의 기본값
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
        mode: str = "vector",
        db_path: Path | None = None,
    ) -> None:
        if max_chunks_per_doc < 1:
            raise ValueError("max_chunks_per_doc 는 1 이상이어야 합니다")
        if mode not in RETRIEVAL_MODES:
            raise ValueError(f"mode 는 {RETRIEVAL_MODES} 중 하나여야 합니다: {mode!r}")
        if mode == "hybrid" and max_chunks_per_doc != 1:
            raise ValueError("하이브리드 검색은 문서 단위로 결합하므로 max_chunks_per_doc=1 만 지원합니다")
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
        self.mode = mode
        self._db_path = db_path  # None 이면 설정의 docs_db_path (키워드 검색용, 읽기 전용)
        self._embeddings = embeddings or get_embedding_model()
        self._store = Chroma(
            collection_name=collection,
            embedding_function=self._embeddings,
            client=client,
            create_collection_if_not_exists=False,
        )

    def retrieve(self, query: str, *, k: int, filters: SearchFilters | None = None) -> list[RetrievedChunk]:
        if k < 1:
            raise ValueError("k 는 1 이상이어야 합니다")
        if not query.strip():
            return []
        where = filters.to_where() if filters else None
        embedding = self._embeddings.embed_query(query)
        if self.mode == "vector":
            fetch = k * FETCH_MULTIPLIER * self.max_chunks_per_doc
            return self._vector_chunks(embedding, fetch=fetch, where=where, limit=k)
        vector_docs = self._vector_chunks(embedding, fetch=HYBRID_CANDIDATES * FETCH_MULTIPLIER, where=where,
                                          limit=HYBRID_CANDIDATES)
        return self._fuse(query, embedding, vector_docs, k=k, where=where, filters=filters)

    def _vector_chunks(self, embedding: list[float], *, fetch: int, where: dict[str, Any] | None,
                       limit: int) -> list[RetrievedChunk]:
        """유사도 순 청크에서 문서당 청크 수 제한과 같은 날 같은 제목 묶기를 적용해 limit 개를 고른다."""
        hits = self._store.similarity_search_by_vector_with_relevance_scores(embedding, k=fetch, filter=where)
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
            results.append(_to_chunk(document, distance))
            if len(results) == limit:
                break
        return results

    def _keyword_doc_ids(self, query: str, filters: SearchFilters | None) -> list[str]:
        f = filters or SearchFilters()
        with open_db(self._db_path, read_only=True) as conn:
            hits = search_keyword_index(
                conn, query_terms(query), limit=HYBRID_CANDIDATES,
                doc_types=[t.value for t in f.doc_types], scopes=list(f.scopes),
                date_from=f.date_from.isoformat() if f.date_from else None,
                date_to=f.date_to.isoformat() if f.date_to else None,
            )
        return [doc_id for doc_id, _ in hits]

    def _best_chunk(self, embedding: list[float], doc_id: str, where: dict[str, Any] | None) -> RetrievedChunk | None:
        """키워드 검색에서만 찾은 문서의 청크 중 질문과 가장 유사한 것."""
        doc_filter: dict[str, Any] = {"doc_id": doc_id}
        hits = self._store.similarity_search_by_vector_with_relevance_scores(
            embedding, k=1, filter={"$and": [doc_filter, where]} if where else doc_filter)
        return _to_chunk(*hits[0]) if hits else None

    def _fuse(self, query: str, embedding: list[float], vector_docs: list[RetrievedChunk], *, k: int,
              where: dict[str, Any] | None, filters: SearchFilters | None) -> list[RetrievedChunk]:
        vector_rank = {c.doc_id: rank for rank, c in enumerate(vector_docs, start=1)}
        keyword_rank = {doc_id: rank for rank, doc_id in enumerate(self._keyword_doc_ids(query, filters), start=1)}
        fused: dict[str, float] = defaultdict(float)
        for ranks in (vector_rank, keyword_rank):
            for doc_id, rank in ranks.items():
                fused[doc_id] += 1.0 / (RRF_K + rank)
        missing = len(vector_docs) + 1
        order = sorted(fused, key=lambda d: (-fused[d], vector_rank.get(d, missing), keyword_rank.get(d, missing)))

        chunk_of = {c.doc_id: c for c in vector_docs}
        results: list[RetrievedChunk] = []
        title_owner: dict[tuple[str, str], str] = {}
        for doc_id in order:
            chunk = chunk_of.get(doc_id) or self._best_chunk(embedding, doc_id, where)
            if chunk is None:
                continue  # 키워드 색인에는 있으나 컬렉션에 없는 문서 (prepare 후 build 를 안 한 경우)
            key = _title_key(chunk.metadata) if self.collapse_same_title else None
            if key is not None and title_owner.setdefault(key, doc_id) != doc_id:
                continue
            results.append(replace(chunk, metadata={
                **chunk.metadata, "vector_rank": vector_rank.get(doc_id), "keyword_rank": keyword_rank.get(doc_id),
                "rrf_score": round(fused[doc_id], 5)}))
            if len(results) == k:
                break
        return results


def _to_chunk(document, distance: float) -> RetrievedChunk:
    meta = dict(document.metadata)
    return RetrievedChunk(chunk_id=meta["chunk_id"], doc_id=meta["doc_id"], text=document.page_content,
                          score=1.0 - distance, metadata=meta)


@lru_cache(maxsize=4)
def get_retriever(collection: str | None = None, mode: str | None = None) -> Retriever:
    """설정의 컬렉션(CHROMA_COLLECTION)과 검색 방식(RAG_RETRIEVAL_MODE)을 기본으로 쓰는 검색기."""
    settings = get_settings()
    return Retriever(collection or settings.chroma_collection, mode=mode or settings.rag_retrieval_mode)


def retrieve(
    query: str,
    *,
    k: int,
    filters: SearchFilters | None = None,
    collection: str | None = None,
) -> list[RetrievedChunk]:
    """collection 을 생략하면 설정의 CHROMA_COLLECTION 을 사용한다."""
    return get_retriever(collection).retrieve(query, k=k, filters=filters)
