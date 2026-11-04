import sqlite3

import pytest

from common.docstore import get_document, open_db, upsert_documents
from common.schema import CollectMethod, DocType, DocumentRecord


def _record(**overrides) -> DocumentRecord:
    fields = dict(
        doc_id="kr_20260115_1a2b3c4d",
        doc_type=DocType.PRESS_RELEASE,
        source="중소벤처기업부",
        title="소상공인 정책자금 공급 계획",
        url="https://www.korea.kr/briefing/pressReleaseView.do?newsId=1",
        collected_at="2026-11-03T09:00:00+09:00",
        collect_method=CollectMethod.CRAWL,
        published_at="2026-01-15",
        extra={"attachments": ["보도자료.pdf"]},
    )
    fields.update(overrides)
    return DocumentRecord(**fields)


def test_upsert_is_idempotent_and_roundtrips(tmp_path):
    db = tmp_path / "docs.db"
    with open_db(db) as conn:
        upsert_documents(conn, [_record()])
        upsert_documents(conn, [_record(title="수정된 제목")])

    with open_db(db, read_only=True) as conn:
        assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1
        doc = get_document(conn, "kr_20260115_1a2b3c4d")

    assert doc["title"] == "수정된 제목"
    assert doc["doc_type"] == "press_release"
    assert doc["extra"] == {"attachments": ["보도자료.pdf"]}
    assert doc["clean_text"] is None  # 인덱서 실행 전
    assert doc["quality_flags"] == []


def test_get_document_includes_processed_fields(tmp_path):
    db = tmp_path / "docs.db"
    with open_db(db) as conn:
        upsert_documents(conn, [_record()])
        conn.execute(
            "INSERT INTO processed_documents "
            "(doc_id, clean_text, canonical_doc_id, status, quality_flags, processed_at) "
            "VALUES (?, ?, ?, 'kept', ?, ?)",
            ("kr_20260115_1a2b3c4d", "정제 본문", "kr_20260115_1a2b3c4d",
             '["missing_source"]', "2026-11-03T10:00:00+09:00"),
        )

    with open_db(db, read_only=True) as conn:
        doc = get_document(conn, "kr_20260115_1a2b3c4d")

    assert doc["clean_text"] == "정제 본문"
    assert doc["status"] == "kept"
    assert doc["quality_flags"] == ["missing_source"]


def test_same_url_with_different_ids_is_kept_for_dedup(tmp_path):
    db = tmp_path / "docs.db"
    with open_db(db) as conn:
        upsert_documents(conn, [_record(doc_id="news_a"), _record(doc_id="news_b")])
        assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 2


def test_read_only_connection_rejects_writes(tmp_path):
    db = tmp_path / "docs.db"
    with open_db(db) as conn:
        upsert_documents(conn, [_record()])

    with pytest.raises(sqlite3.OperationalError):
        with open_db(db, read_only=True) as conn:
            upsert_documents(conn, [_record(doc_id="other")])


def test_read_only_requires_existing_db(tmp_path):
    with pytest.raises(FileNotFoundError):
        with open_db(tmp_path / "missing.db", read_only=True):
            pass


def test_failed_write_is_rolled_back(tmp_path):
    db = tmp_path / "docs.db"
    with pytest.raises(RuntimeError):
        with open_db(db) as conn:
            upsert_documents(conn, [_record()])
            raise RuntimeError("중간 실패")

    with open_db(db, read_only=True) as conn:
        assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0


@pytest.mark.parametrize(
    "overrides",
    [
        {"title": "  "},
        {"doc_type": "unknown"},
        {"published_at": "2026/01/15"},
        {"collected_at": "어제"},
    ],
)
def test_record_validation(overrides):
    with pytest.raises(ValueError):
        _record(**overrides)


def _keyword_db(tmp_path):
    from common.docstore import replace_keyword_index, replace_processed_documents
    from common.schema import ProcessedRecord
    docs = {  # doc_id: (제목, 발행일, 범위, 본문)
        "b2024": ("중기부 2024년 예산 편성", "2023-08-29", "national", "중기부는 2024년 예산을 14조원으로 편성했다."),
        "b2027": ("소상공인 금융지원 확대", "2026-09-01", "national", "중기부는 2027년도 예산안에 금융지원 4조2500억원을 담았다."),
        "alley": ("서울 골목상권 매출", "2025-01-01", "seoul", "서울 골목상권 매출이 늘었다."),
    }
    db = tmp_path / "docs.db"
    with open_db(db) as conn:
        upsert_documents(conn, [DocumentRecord(doc_id=d, doc_type=DocType.NEWS, source="연합뉴스", title=t,
                                               url=f"https://x/{d}", collected_at="2026-10-16",
                                               collect_method=CollectMethod.PROVIDED, published_at=f"{p}T09:00:00")
                                for d, (t, p, _, _) in docs.items()])
        replace_processed_documents(conn, [ProcessedRecord(doc_id=d, status="kept", processed_at="2026-11-03",
                                                           clean_text=x, canonical_doc_id=d, scope=s)
                                           for d, (_, _, s, x) in docs.items()])
        assert replace_keyword_index(conn, [(d, t, x) for d, (t, _, _, x) in docs.items()]) == 3
    return db


def test_keyword_index_ranks_year_and_institution_matches_first(tmp_path):
    from common.docstore import search_keyword_index
    from common.text import query_terms
    db = _keyword_db(tmp_path)
    with open_db(db, read_only=True) as conn:
        hits = search_keyword_index(conn, query_terms("중기부 2027년도 예산안의 금융지원 규모는?"), limit=5)
        assert [d for d, _ in hits][:2] == ["b2027", "b2024"] and hits[0][1] > hits[1][1]
        assert "alley" not in {d for d, _ in hits}                       # 공통 토큰이 없으면 나오지 않음
        assert [d for d, _ in search_keyword_index(conn, ["예산"], limit=5, date_from="2025-01-01")] == ["b2027"]
        assert [d for d, _ in search_keyword_index(conn, ["매출"], limit=5, scopes=["seoul"])] == ["alley"]
        assert search_keyword_index(conn, [], limit=5) == []


def test_keyword_search_explains_missing_index(tmp_path):
    from common.docstore import search_keyword_index
    db = tmp_path / "docs.db"
    with open_db(db):
        pass
    with open_db(db, read_only=True) as conn, pytest.raises(RuntimeError, match="indexer prepare"):
        search_keyword_index(conn, ["예산"], limit=5)


def test_composition_counts_by_status_and_month(tmp_path):
    from common.docstore import document_status_counts, kept_documents_by_month, replace_processed_documents
    from common.schema import ProcessedRecord
    db = tmp_path / "docs.db"
    with open_db(db) as conn:
        upsert_documents(conn, [
            _record(doc_id="kr_1", published_at="2026-01-15"),
            _record(doc_id="kr_2", published_at="2026-01-20T10:00:00"),
            _record(doc_id="kr_3", published_at=None),
            _record(doc_id="news_1", doc_type=DocType.NEWS, published_at="2025-12-01"),
            _record(doc_id="news_2", doc_type=DocType.NEWS, published_at="2025-12-01"),
            _record(doc_id="news_3", doc_type=DocType.NEWS, published_at="2025-12-02"),  # 인덱서 실행 전
        ])
        replace_processed_documents(conn, [
            ProcessedRecord(doc_id="kr_1", status="kept", processed_at="2026-11-03", canonical_doc_id="kr_1"),
            ProcessedRecord(doc_id="kr_2", status="kept", processed_at="2026-11-03", canonical_doc_id="kr_2"),
            ProcessedRecord(doc_id="kr_3", status="kept", processed_at="2026-11-03", canonical_doc_id="kr_3"),
            ProcessedRecord(doc_id="news_1", status="kept", processed_at="2026-11-03", canonical_doc_id="news_1"),
            ProcessedRecord(doc_id="news_2", status="duplicate", processed_at="2026-11-03", canonical_doc_id="news_1"),
        ])

    with open_db(db, read_only=True) as conn:
        assert document_status_counts(conn) == [
            ("news", None, 1), ("news", "duplicate", 1), ("news", "kept", 1), ("press_release", "kept", 3)]
        assert kept_documents_by_month(conn) == [
            ("news", "2025-12", 1), ("press_release", None, 1), ("press_release", "2026-01", 2)]
