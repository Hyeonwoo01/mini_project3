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
