from datetime import date

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from common.docstore import open_db, replace_processed_documents, upsert_documents
from common.schema import CollectMethod, DocType, DocumentRecord, ProcessedRecord
from indexer.chunk import get_chunk_config
from indexer.embed import build_collection
from rag.retriever import RetrievedChunk, Retriever, SearchFilters

SENTENCE = "소상공인 지원 정책이 확대된다. 골목상권 매출 회복이 기대된다. "
DOCS = {  # doc_id: (발행일, 범위, 반복 횟수 → 청크 수)
    "old": ("2022-03-01T09:00:00", "national", 3),
    "seoul": ("2025-06-01T09:00:00", "seoul", 3),
    "long": ("2026-01-08T09:00:00", "national", 60),   # 청크 여러 개
    "twin1": ("2023-10-16T09:00:00", "national", 2),   # 같은 날 같은 제목 (요약·전문 기사)
    "twin2": ("2023-10-16T15:00:00", "national", 8),
}
TITLES = {"twin1": "서울 골목상권 매출 14% 증가", "twin2": "“서울 골목상권 매출 14% 증가”"}
FAKE = DeterministicFakeEmbedding(size=16)


@pytest.fixture(scope="module")
def chroma(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("retriever")
    db, chroma_dir = tmp / "docs.db", tmp / "chroma"
    with open_db(db) as conn:
        upsert_documents(conn, [
            DocumentRecord(doc_id=d, doc_type=DocType.NEWS, source="연합뉴스", title=TITLES.get(d, f"{d} 기사"), url=f"https://x/{d}",
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
    results = _retriever(chroma).retrieve("골목상권 매출", k=10)
    doc_ids = [r.doc_id for r in results]
    assert len(doc_ids) == len(set(doc_ids)) == 4          # 'long' 의 여러 청크가 상위를 채우지 않음
    assert len({"twin1", "twin2"} & set(doc_ids)) == 1      # 같은 날 같은 제목은 하나만
    assert all(-1.0 <= r.score <= 1.0 for r in results)
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)


def test_same_title_collapse_can_be_disabled(chroma):
    results = _retriever(chroma, collapse_same_title=False).retrieve("골목상권 매출", k=10)
    assert {"twin1", "twin2"} <= {r.doc_id for r in results}


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


HYBRID_DOCS = {  # doc_id: (제목, 본문) — 질의 '2027년도 예산안' 과 겹치는 토큰은 b2027 에만 있다
    "b2027": ("소상공인 금융지원 확대", "중기부는 2027년도 예산안에 금융지원 4조2500억원을 담았다. " * 3),
    "alley": ("서울 골목상권 매출 증가", "서울 골목상권 매출이 늘었다. 상인회는 축제를 열었다. " * 3),
    "market": ("전통시장 축제 개최", "전통시장 상인회가 가을 축제를 열고 할인 행사를 진행했다. " * 3),
}


@pytest.fixture(scope="module")
def hybrid_env(tmp_path_factory):
    from common.docstore import replace_keyword_index
    tmp = tmp_path_factory.mktemp("hybrid")
    db, chroma_dir = tmp / "docs.db", tmp / "chroma"
    with open_db(db) as conn:
        upsert_documents(conn, [
            DocumentRecord(doc_id=d, doc_type=DocType.NEWS, source="연합뉴스", title=t, url=f"https://x/{d}",
                           collected_at="2026-10-16", collect_method=CollectMethod.PROVIDED,
                           published_at="2026-09-01T09:00:00")
            for d, (t, _) in HYBRID_DOCS.items()
        ])
        replace_processed_documents(conn, [
            ProcessedRecord(doc_id=d, status="kept", processed_at="2026-11-03", clean_text=x, canonical_doc_id=d,
                            scope="national")
            for d, (_, x) in HYBRID_DOCS.items()
        ])
        replace_keyword_index(conn, [(d, t, x) for d, (t, x) in HYBRID_DOCS.items()])
    build_collection(get_chunk_config("c500"), "docs__hybrid", embeddings=FAKE, embedding_model_name="fake",
                     chroma_dir=chroma_dir, db_path=db, progress=lambda _: None)
    return db, chroma_dir


def _hybrid(env, **kwargs):
    db, chroma_dir = env
    return Retriever("docs__hybrid", chroma_dir=chroma_dir, embeddings=FAKE, embedding_model_name="fake",
                     mode="hybrid", db_path=db, **kwargs)


def test_hybrid_ranks_keyword_match_first_and_records_ranks(hybrid_env):
    results = _hybrid(hybrid_env).retrieve("2027년도 예산안 규모는?", k=3)
    assert results[0].doc_id == "b2027"                       # 키워드 1위 + 의미 순위 → RRF 1위
    assert results[0].metadata["keyword_rank"] == 1 and results[0].metadata["vector_rank"] in (1, 2, 3)
    assert {r.doc_id for r in results} == set(HYBRID_DOCS)
    assert all(r.metadata["keyword_rank"] is None for r in results[1:])
    assert all(-1.0 <= r.score <= 1.0 for r in results)       # score 는 여전히 코사인 유사도


def test_hybrid_fetches_chunk_for_keyword_only_document(hybrid_env):
    retriever = _hybrid(hybrid_env)
    retriever._vector_chunks = lambda *args, **kwargs: []    # 의미 검색 후보에 없는 상황을 만든다
    results = retriever.retrieve("2027년도 예산안 규모는?", k=3)
    assert [r.doc_id for r in results] == ["b2027"]            # 키워드로만 찾은 문서
    assert results[0].metadata["vector_rank"] is None and results[0].metadata["keyword_rank"] == 1
    assert results[0].chunk_id.startswith("b2027#") and results[0].text   # 질문 벡터로 그 문서의 청크를 골라 붙임
    assert -1.0 <= results[0].score <= 1.0


def test_retrieval_mode_validation(hybrid_env):
    with pytest.raises(ValueError):
        _hybrid(hybrid_env, max_chunks_per_doc=2)
    db, chroma_dir = hybrid_env
    with pytest.raises(ValueError):
        Retriever("docs__hybrid", chroma_dir=chroma_dir, embeddings=FAKE, embedding_model_name="fake", mode="bm25")


def _piece(index, body, title="제목"):
    return RetrievedChunk(chunk_id=f"d#{index:03d}", doc_id="d", text=f"{title}\n\n{body}", score=0.6,
                          metadata={"title": title, "chunk_index": index, "doc_id": "d"})


def test_merge_chunk_texts_removes_overlap_and_repeated_title():
    from rag.retriever import merge_chunk_texts
    adjacent = merge_chunk_texts([_piece(0, "첫 문장이다. 두 청크에 겹치는 문장이다."),
                                  _piece(1, "두 청크에 겹치는 문장이다. 다음 문장이다.")])
    assert adjacent == "제목\n\n첫 문장이다. 두 청크에 겹치는 문장이다. 다음 문장이다."
    coincidence = merge_chunk_texts([_piece(0, "앞 문장이다."), _piece(1, "다음 문장이다.")])
    assert coincidence == "제목\n\n앞 문장이다. 다음 문장이다."      # '다' 한 글자 일치는 지우지 않음
    apart = merge_chunk_texts([_piece(0, "앞부분."), _piece(3, "뒷부분.")])
    assert apart == "제목\n\n앞부분.\n…\n뒷부분."
    assert merge_chunk_texts([_piece(2, "하나뿐.")]) == "제목\n\n하나뿐."


def test_expand_context_adds_second_chunk_of_same_document(chroma):
    retriever = _retriever(chroma)
    long_chunk = next(r for r in retriever.retrieve("골목상권 매출", k=10) if r.doc_id == "long")
    expanded = retriever.expand_context("골목상권 매출", [long_chunk], per_doc=2)[0]
    ids = expanded.metadata["context_chunk_ids"]
    assert len(ids) == 2 and long_chunk.chunk_id in ids and all(i.startswith("long#") for i in ids)
    assert ids == sorted(ids)                                     # 기사 순서대로
    assert expanded.text.count(TITLES.get("long", "long 기사")) == 1   # 제목 머리말은 한 번만
    assert expanded.score == long_chunk.score and expanded.chunk_id == long_chunk.chunk_id
    assert retriever.expand_context("골목상권 매출", [long_chunk], per_doc=1) == [long_chunk]
    single = next(r for r in retriever.retrieve("골목상권 매출", k=10) if r.doc_id == "old")
    assert len(retriever.expand_context("q", [single], per_doc=3)[0].metadata["context_chunk_ids"]) >= 1
