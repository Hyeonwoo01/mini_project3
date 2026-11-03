from datetime import date

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from common.docstore import open_db, replace_processed_documents, upsert_documents
from common.schema import CollectMethod, DocType, DocumentRecord, ProcessedRecord
from indexer.chunk import get_chunk_config
from indexer.embed import build_collection
from rag.retriever import Retriever, SearchFilters

SENTENCE = "소상공인 지원 정책이 확대된다. 골목상권 매출 회복이 기대된다. "
DOCS = {  # doc_id: (발행일, 범위, 반복 횟수 → 청크 수)
    "old": ("2022-03-01T09:00:00", "national", 3),
    "seoul": ("2025-06-01T09:00:00", "seoul", 3),
    "long": ("2026-01-08T09:00:00", "national", 60),   # 청크 여러 개
}
FAKE = DeterministicFakeEmbedding(size=16)


@pytest.fixture(scope="module")
def chroma(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("retriever")
    db, chroma_dir = tmp / "docs.db", tmp / "chroma"
    with open_db(db) as conn:
        upsert_documents(conn, [
            DocumentRecord(doc_id=d, doc_type=DocType.NEWS, source="연합뉴스", title=f"{d} 기사", url=f"https://x/{d}",
                           collected_at="2026-10-16", collect_method=CollectMethod.PROVIDED, published_at=published)
            for d, (published, _, _) in DOCS.items()
        ])
        replace_processed_documents(conn, [
            ProcessedRecord(doc_id=d, status="kept", processed_at="2026-11-03", clean_text=SENTENCE * n,
                            canonical_doc_id=d, scope=scope)
            for d, (_, scope, n) in DOCS.items()
        ])
    build_collection(get_chunk_config("c500"), "docs__test", embeddings=FAKE, embedding_model_name="fake",
                     chroma_dir=chroma_dir, db_path=db, progress=lambda _: None)
    return chroma_dir


def _retriever(chroma, **kwargs):
    return Retriever("docs__test", chroma_dir=chroma, embeddings=FAKE, embedding_model_name="fake", **kwargs)


def test_returns_one_chunk_per_document_with_scores(chroma):
    results = _retriever(chroma).retrieve("골목상권 매출", k=5)
    doc_ids = [r.doc_id for r in results]
    assert len(doc_ids) == len(set(doc_ids)) == 3          # 'long' 의 여러 청크가 상위를 채우지 않음
    assert all(-1.0 <= r.score <= 1.0 for r in results)
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)
    assert results[0].metadata["title"].endswith("기사")


def test_max_chunks_per_doc_allows_more_chunks(chroma):
    results = _retriever(chroma, max_chunks_per_doc=3).retrieve("골목상권 매출", k=10)
    assert sum(r.doc_id == "long" for r in results) > 1


def test_filters(chroma):
    retriever = _retriever(chroma)
    recent = retriever.retrieve("소상공인", k=5, filters=SearchFilters(date_from=date(2025, 1, 1)))
    assert {r.doc_id for r in recent} == {"seoul", "long"}
    seoul = retriever.retrieve("소상공인", k=5, filters=SearchFilters(scopes=("seoul",)))
    assert {r.doc_id for r in seoul} == {"seoul"}
    window = retriever.retrieve("소상공인", k=5, filters=SearchFilters(date_from=date(2022, 1, 1), date_to=date(2022, 12, 31),
                                                                      doc_types=(DocType.NEWS,)))
    assert {r.doc_id for r in window} == {"old"}
    assert retriever.retrieve("소상공인", k=5, filters=SearchFilters(doc_types=(DocType.PRESS_RELEASE,))) == []


def test_where_clause_shapes():
    assert SearchFilters().to_where() is None
    assert SearchFilters(scopes=("seoul",)).to_where() == {"scope": {"$in": ["seoul"]}}
    both = SearchFilters(date_from=date(2025, 1, 1), date_to=date(2025, 12, 31)).to_where()
    assert both == {"$and": [{"published_ymd": {"$gte": 20250101}}, {"published_ymd": {"$lte": 20251231}}]}


def test_errors(chroma):
    with pytest.raises(RuntimeError, match="컬렉션이 없습니다"):
        Retriever("docs__missing", chroma_dir=chroma, embeddings=FAKE, embedding_model_name="fake")
    with pytest.raises(RuntimeError, match="임베딩 모델"):
        Retriever("docs__test", chroma_dir=chroma, embeddings=FAKE, embedding_model_name="other-model")
    retriever = _retriever(chroma)
    assert retriever.retrieve("   ", k=3) == []
    with pytest.raises(ValueError):
        retriever.retrieve("소상공인", k=0)
