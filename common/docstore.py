"""원문 DB(store/docs.db) 접근 — ③ 저장 계층의 원문 측 단일 접점.

벡터 DB 에는 청크와 메타데이터만 저장하고, 원문 전체는 이 DB 에 보관한다 (가이드 4-5).

테이블 소유권 — 각 테이블은 한 계층만 기록한다.
    documents            collector 가 기록. 수집 원문과 출처 메타데이터.
    processed_documents  indexer 가 기록. 정제 본문, 중복 판정 결과.
    keyword_index        indexer 가 기록. 인덱싱 대상(kept) 문서의 키워드 검색 색인 (FTS5).
rag / app / eval 은 open_db(read_only=True) 로만 연다. SQLite 수준에서 쓰기가 차단된다.

URL 에 UNIQUE 제약을 두지 않는다. 제공 데이터의 완전 중복(같은 URL)도 일단 모두 적재해야
중복 처리 전후 건수를 기록할 수 있기 때문이다 (가이드 4-4). 중복 판정은 indexer 가 한다.
"""

import json
import sqlite3
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from common.config import get_settings
from common.schema import DocumentRecord, ProcessedRecord
from common.text import keyword_terms

DOCUMENTS_SCHEMA = """
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
"""

# indexer 가 prepare 실행마다 통째로 다시 만드는 파생 테이블이다 (replace_processed_documents).
PROCESSED_SCHEMA = """
CREATE TABLE IF NOT EXISTS processed_documents (
    doc_id            TEXT PRIMARY KEY REFERENCES documents(doc_id) ON DELETE CASCADE,
    clean_text        TEXT,
    content_hash      TEXT,
    canonical_doc_id  TEXT REFERENCES documents(doc_id),  -- 대표 문서 id (대표 문서는 자기 자신)
    status            TEXT NOT NULL CHECK (status IN ('kept', 'duplicate', 'excluded')),
    excluded_reason   TEXT,                      -- indexer.quality.ExclusionReason 값
    quality_flags     TEXT NOT NULL DEFAULT '[]', -- indexer.quality.QualityFlag 값 목록 (JSON)
    scope             TEXT,                      -- indexer.quality.Scope 값 (뉴스만)
    processed_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_processed_status ON processed_documents(status);
"""

# indexer 가 prepare 실행마다 다시 만드는 키워드 검색 색인. 인덱싱 대상(kept) 문서만 담는다.
# SQLite 기본 토크나이저는 한글을 어절 단위로만 자르므로, common.text.keyword_terms 로 미리 자른 토큰을
# 공백으로 이어 넣는다. 제목과 본문을 다른 열에 두어 BM25 에서 제목 일치에 가중치를 준다.
KEYWORD_SCHEMA = """
CREATE VIRTUAL TABLE keyword_index USING fts5(doc_id UNINDEXED, title_terms, body_terms, tokenize = 'unicode61');
"""
KEYWORD_TITLE_WEIGHT = 2.0

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
        conn.executescript(DOCUMENTS_SCHEMA + PROCESSED_SCHEMA)
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
    """원문 확인 영역용 조회. 인덱서 처리 결과(정제 본문·상태·제외 사유)가 있으면 함께 반환한다."""
    row = conn.execute(
        """
        SELECT d.*, p.clean_text, p.status, p.canonical_doc_id, p.excluded_reason, p.quality_flags, p.scope
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


def iter_documents(conn: sqlite3.Connection) -> Iterator[sqlite3.Row]:
    """수집된 문서 전체를 doc_id 순으로 읽는다."""
    yield from conn.execute("SELECT * FROM documents ORDER BY doc_id")


def replace_processed_documents(conn: sqlite3.Connection, records: Iterable[ProcessedRecord]) -> int:
    """processed_documents 를 비우고 다시 채운다. prepare 는 항상 전체를 다시 계산한다.

    테이블을 지우고 새로 만들어서, 이전 버전 스키마로 만들어진 DB 도 새 컬럼을 갖게 한다.
    """
    conn.executescript("DROP TABLE IF EXISTS processed_documents;" + PROCESSED_SCHEMA)
    rows = [
        (
            r.doc_id, r.clean_text, r.content_hash, r.canonical_doc_id, r.status, r.excluded_reason,
            json.dumps(list(r.quality_flags), ensure_ascii=False), r.scope, r.processed_at,
        )
        for r in records
    ]
    conn.executemany(
        "INSERT INTO processed_documents (doc_id, clean_text, content_hash, canonical_doc_id, status, "
        "excluded_reason, quality_flags, scope, processed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    return len(rows)


def replace_keyword_index(conn: sqlite3.Connection, docs: Iterable[tuple[str, str, str]]) -> int:
    """(doc_id, 제목, 정제 본문) 으로 키워드 색인을 다시 만든다. 색인된 문서 수를 반환한다."""
    conn.executescript("DROP TABLE IF EXISTS keyword_index;" + KEYWORD_SCHEMA)
    rows = ((doc_id, " ".join(keyword_terms(title)), " ".join(keyword_terms(text or "")))
            for doc_id, title, text in docs)
    return conn.executemany(
        "INSERT INTO keyword_index (doc_id, title_terms, body_terms) VALUES (?, ?, ?)", rows
    ).rowcount


def search_keyword_index(
    conn: sqlite3.Connection,
    terms: Sequence[str],
    *,
    limit: int,
    doc_types: Sequence[str] = (),
    date_from: str | None = None,
    date_to: str | None = None,
    scopes: Sequence[str] = (),
) -> list[tuple[str, float]]:
    """토큰 중 하나라도 포함한 문서를 BM25 순으로 반환한다. (doc_id, 점수) — 점수는 클수록 관련이 높다.

    date_from/date_to 는 'YYYY-MM-DD'. 기간 조건이 있으면 발행일 미상 문서는 빠진다 (의미 검색과 같은 규칙).
    """
    if not terms:
        return []
    match = " OR ".join('"' + term.replace('"', '""') + '"' for term in dict.fromkeys(terms))
    sql = ["SELECT keyword_index.doc_id, -bm25(keyword_index, 0.0, ?, 1.0) AS score",
           "FROM keyword_index JOIN documents AS d ON d.doc_id = keyword_index.doc_id",
           "JOIN processed_documents AS p ON p.doc_id = keyword_index.doc_id",
           "WHERE keyword_index MATCH ?"]
    params: list[object] = [KEYWORD_TITLE_WEIGHT, match]
    if doc_types:
        sql.append(f"AND d.doc_type IN ({', '.join('?' for _ in doc_types)})")
        params += list(doc_types)
    if scopes:
        sql.append(f"AND p.scope IN ({', '.join('?' for _ in scopes)})")
        params += list(scopes)
    if date_from:
        sql.append("AND substr(d.published_at, 1, 10) >= ?")
        params.append(date_from)
    if date_to:
        sql.append("AND substr(d.published_at, 1, 10) <= ?")
        params.append(date_to)
    sql.append("ORDER BY score DESC LIMIT ?")
    params.append(limit)
    try:
        rows = conn.execute("\n".join(sql), params).fetchall()
    except sqlite3.OperationalError as exc:
        if "no such table: keyword_index" in str(exc):
            raise RuntimeError("키워드 색인이 없습니다. python -m indexer prepare 를 다시 실행하세요.") from exc
        raise
    return [(row[0], float(row[1])) for row in rows]
