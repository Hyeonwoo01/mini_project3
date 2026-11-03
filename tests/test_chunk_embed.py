import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from common.docstore import open_db, replace_processed_documents, upsert_documents
from common.schema import CHUNK_METADATA_FIELDS, CollectMethod, DocType, DocumentRecord, ProcessedRecord
from indexer.chunk import ChunkConfig, ChunkSource, chunk_document, get_chunk_config
from indexer.embed import build_collection, verify_collection

SENTENCES = "서울시는 골목상권 지원을 확대한다. 대상은 영세 소상공인이다. 신청은 다음 달부터 받는다. "


def _source(text=SENTENCES * 12, published_at="2026-01-08T09:00:00", scope="seoul"):
    return ChunkSource(doc_id="news_1", doc_type=DocType.NEWS, title="서울시 골목상권 지원 확대", text=text,
                       source="연합뉴스", url="https://x/1", published_at=published_at, collected_at="2026-10-16",
                       scope=scope)


def test_chunks_carry_title_metadata_and_sentence_boundaries():
    chunks = chunk_document(_source(), get_chunk_config("c500"))
    assert len(chunks) == 2
    assert [c.chunk_id for c in chunks] == ["news_1#000", "news_1#001"]
    assert all(c.text.startswith("서울시 골목상권 지원 확대\n\n") for c in chunks)
    assert all(len(c.body) <= 500 for c in chunks)
    assert chunks[0].body.endswith("다.")                       # 문장 끝에서 자른다
    meta = chunks[0].metadata
    assert set(meta) == set(CHUNK_METADATA_FIELDS)
    assert meta["published_ymd"] == 20260108 and meta["source_tier"] == "언론 보도" and meta["scope"] == "seoul"
    assert all(v is not None for v in meta.values())


def test_no_title_config_and_missing_values():
    chunks = chunk_document(_source(published_at=None, scope=None), get_chunk_config("c500-notitle"))
    assert not chunks[0].text.startswith("서울시 골목상권 지원 확대")
    assert chunks[0].metadata["published_at"] == "" and chunks[0].metadata["published_ymd"] == 0
    assert chunks[0].metadata["scope"] == ""
    assert chunk_document(_source(text="   "), get_chunk_config("c500")) == []


def test_chunk_config_validation():
    with pytest.raises(ValueError):
        ChunkConfig(name="x", chunk_size=100, chunk_overlap=100)
    with pytest.raises(SystemExit):
        get_chunk_config("does-not-exist")


def _seed(db, docs):
    with open_db(db) as conn:
        upsert_documents(conn, [
            DocumentRecord(doc_id=d, doc_type=DocType.NEWS, source="연합뉴스", title=f"{d} 소상공인 기사",
                           url=f"https://x/{d}", collected_at="2026-10-16", collect_method=CollectMethod.PROVIDED,
                           published_at="2026-01-08T09:00:00")
            for d in docs
        ])
        replace_processed_documents(conn, [
            ProcessedRecord(doc_id=d, status="kept", processed_at="2026-11-03", clean_text=SENTENCES * n,
                            canonical_doc_id=d, scope="national")
            for d, n in docs.items()
        ])


@pytest.fixture
def env(tmp_path):
    db, chroma = tmp_path / "docs.db", tmp_path / "chroma"
    _seed(db, {"a": 12, "b": 3})
    kwargs = dict(embeddings=DeterministicFakeEmbedding(size=16), embedding_model_name="fake",
                  chroma_dir=chroma, db_path=db, progress=lambda _: None)
    return db, chroma, kwargs


def test_dry_run_reports_without_writing(env):
    db, chroma, kwargs = env
    stats = build_collection(get_chunk_config("c500"), "docs__c500", dry_run=True, **kwargs)
    assert (stats.documents, stats.chunks, stats.added) == (2, 3, 0)
    assert not chroma.exists()
    assert "청크 3개" in stats.summary()


def test_build_resumes_removes_stale_and_verifies(env):
    db, chroma, kwargs = env
    config = get_chunk_config("c500")
    first = build_collection(config, "docs__c500", **kwargs)
    assert (first.added, first.already_present) == (3, 0)

    again = build_collection(config, "docs__c500", **kwargs)    # 이어서 적재: 이미 있는 청크는 건너뜀
    assert (again.added, again.already_present) == (0, 3)

    _seed(db, {"a": 12})                                        # 지식베이스에서 b 가 빠짐
    shrunk = build_collection(config, "docs__c500", **kwargs)
    assert shrunk.removed_stale == 1

    report = verify_collection("docs__c500", queries=["골목상권 지원"], embeddings=kwargs["embeddings"],
                               chroma_dir=chroma, db_path=db)
    assert (report.chunks, report.documents, report.missing_documents, report.empty_chunks) == (2, 1, 0, 0)
    assert report.metadata["embedding_model"] == "fake" and report.metadata["chunk_config"] == "c500"
    assert len(report.samples[0][1]) == 2


def test_embedding_model_mismatch_requires_rebuild(env):
    db, chroma, kwargs = env
    build_collection(get_chunk_config("c500"), "docs__c500", **kwargs)
    with pytest.raises(SystemExit):
        build_collection(get_chunk_config("c500"), "docs__c500", **{**kwargs, "embedding_model_name": "other"})
    rebuilt = build_collection(get_chunk_config("c500"), "docs__c500", rebuild=True,
                               **{**kwargs, "embedding_model_name": "other"})
    assert rebuilt.added == 3


def test_verify_missing_collection(env):
    db, chroma, kwargs = env
    with pytest.raises(SystemExit):
        verify_collection("docs__none", queries=(), chroma_dir=chroma, db_path=db)
