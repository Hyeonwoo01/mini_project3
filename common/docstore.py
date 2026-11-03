"""원문 DB(store/docs.db) 접근 — ③ 저장 계층의 원문 측 단일 접점.

벡터 DB 에는 청크와 메타데이터만 저장하고, 원문 전체는 이 DB 에 보관한다 (가이드 4-5).

테이블 소유권 — 각 테이블은 한 계층만 기록한다.
    documents            collector 가 기록. 수집 원문과 출처 메타데이터.
    processed_documents  indexer 가 기록. 정제 본문, 중복 판정 결과.
rag / app / eval 은 open_db(read_only=True) 로만 연다. SQLite 수준에서 쓰기가 차단된다.

URL 에 UNIQUE 제약을 두지 않는다. 제공 데이터의 완전 중복(같은 URL)도 일단 모두 적재해야
중복 처리 전후 건수를 기록할 수 있기 때문이다 (가이드 4-4). 중복 판정은 indexer 가 한다.
"""

import json
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from common.config import get_settings
from common.schema import DocumentRecord

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id          TEXT PRIMARY KEY,
    doc_type        TEXT NOT NULL,
    source          TEXT NOT NULL,
    title           TEXT NOT NULL,
    url             TEXT NOT NULL,
    published_at    TEXT,
    collected_at    TEXT NOT NULL,
    collect_method  TEXT NOT NULL,
    keyword         TEXT,
    raw_path        TEXT,
    raw_text        TEXT,
    extra           TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_documents_url ON documents(url);
CREATE INDEX IF NOT EXISTS idx_documents_type_date ON documents(doc_type, published_at);

CREATE TABLE IF NOT EXISTS processed_documents (
    doc_id            TEXT PRIMARY KEY REFERENCES documents(doc_id) ON DELETE CASCADE,
    clean_text        TEXT,
    content_hash      TEXT,
    canonical_doc_id  TEXT REFERENCES documents(doc_id),  -- 대표 문서 id (대표 문서는 자기 자신)
    status            TEXT NOT NULL CHECK (status IN ('kept', 'duplicate', 'excluded')),
    excluded_reason   TEXT,                      -- indexer.quality.ExclusionReason 값
    quality_flags     TEXT NOT NULL DEFAULT '[]', -- indexer.quality.QualityFlag 값 목록 (JSON)
    processed_at      TEXT NOT NULL
);
"""

_DOCUMENT_COLUMNS = (
    "doc_id", "doc_type", "source", "title", "url", "published_at", "collected_at",
    "collect_method", "keyword", "raw_path", "raw_text", "extra",
)


def _connect(db_path: Path, read_only: bool) -> sqlite3.Connection:
    if read_only:
        if not db_path.exists():
            raise FileNotFoundError(
                f"원문 DB 가 없습니다: {db_path}. 수집기(python -m collector)를 먼저 실행하세요."
            )
        conn = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    else:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path)
        conn.executescript(SCHEMA)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def open_db(db_path: Path | None = None, *, read_only: bool = False) -> Iterator[sqlite3.Connection]:
    """연결을 열고 종료 시 닫는다. 쓰기 모드는 정상 종료 시 commit, 예외 시 rollback 한다."""
    conn = _connect(db_path or get_settings().docs_db_path, read_only)
    try:
        yield conn
        if not read_only:
            conn.commit()
    except BaseException:
        if not read_only:
            conn.rollback()
        raise
    finally:
        conn.close()


def upsert_documents(conn: sqlite3.Connection, records: Iterable[DocumentRecord]) -> int:
    """doc_id 기준으로 삽입하거나 갱신한다. 같은 doc_id 로 재수집하면 덮어쓴다."""
    rows = [
        (
            r.doc_id, r.doc_type.value, r.source, r.title, r.url, r.published_at,
            r.collected_at, r.collect_method.value, r.keyword, r.raw_path, r.raw_text,
            json.dumps(r.extra, ensure_ascii=False),
        )
        for r in records
    ]
    columns = ", ".join(_DOCUMENT_COLUMNS)
    placeholders = ", ".join("?" for _ in _DOCUMENT_COLUMNS)
    updates = ", ".join(f"{c} = excluded.{c}" for c in _DOCUMENT_COLUMNS if c != "doc_id")
    conn.executemany(
        f"INSERT INTO documents ({columns}) VALUES ({placeholders}) "
        f"ON CONFLICT(doc_id) DO UPDATE SET {updates}",
        rows,
    )
    return len(rows)


def get_document(conn: sqlite3.Connection, doc_id: str) -> dict[str, Any] | None:
    """원문 확인 영역용 조회. 정제 본문이 있으면 함께 반환한다."""
    row = conn.execute(
        """
        SELECT d.*, p.clean_text, p.status, p.canonical_doc_id, p.quality_flags
        FROM documents AS d
        LEFT JOIN processed_documents AS p USING (doc_id)
        WHERE d.doc_id = ?
        """,
        (doc_id,),
    ).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["extra"] = json.loads(result["extra"])
    # 인덱서 실행 전이면 processed_documents 행이 없어 NULL 이다.
    result["quality_flags"] = json.loads(result["quality_flags"] or "[]")
    return result
