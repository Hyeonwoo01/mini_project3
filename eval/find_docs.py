"""평가셋 작성용 문서 검색 — 정답 문서(gold_doc_ids)를 찾는다.

    python -m eval.find_docs 상생페이백 11월
    python -m eval.find_docs 지역화폐 예산 삭감 --year 2022
    python -m eval.find_docs 골목상권 매출 --scope seoul --limit 30
    python -m eval.find_docs --doc news_20251216_1a2b3c4d     # 문서 전문 확인

- 인덱싱 대상(processed_documents.status = 'kept')만 검색한다. 중복으로 빠진 문서는 대표 문서로
  대체되었으므로, 평가셋에는 여기서 찾은 doc_id 를 적어야 검색 지표가 올바르게 계산된다.
- 검색어는 모두 포함(AND)해야 하며, 제목 또는 정제 본문에 있으면 된다.
- 원문 DB 를 읽기 전용으로 연다.
"""

import argparse
import sqlite3
from dataclasses import dataclass

from common.docstore import open_db

SNIPPET_RADIUS = 60


@dataclass(frozen=True)
class FoundDoc:
    doc_id: str
    published_at: str | None
    source: str
    doc_type: str
    scope: str | None
    title: str
    snippet: str


def _snippet(text: str, terms: list[str]) -> str:
    positions = [pos for term in terms if (pos := text.find(term)) >= 0]
    if not positions:
        return text[: SNIPPET_RADIUS * 2]
    start = max(min(positions) - SNIPPET_RADIUS, 0)
    return ("…" if start else "") + text[start:start + SNIPPET_RADIUS * 2].replace("\n", " ") + "…"


def search(
    conn: sqlite3.Connection,
    terms: list[str],
    *,
    scope: str | None = None,
    year: int | None = None,
    limit: int = 20,
) -> list[FoundDoc]:
    if not terms:
        raise ValueError("검색어를 하나 이상 입력하세요")
    sql = [
        "SELECT d.doc_id, d.published_at, d.source, d.doc_type, d.title, p.scope, p.clean_text",
        "FROM documents AS d JOIN processed_documents AS p USING (doc_id)",
        "WHERE p.status = 'kept'",
    ]
    params: list[object] = []
    for term in terms:
        sql.append("AND (d.title LIKE ? OR p.clean_text LIKE ?)")
        params += [f"%{term}%", f"%{term}%"]
    if scope:
        sql.append("AND p.scope = ?")
        params.append(scope)
    if year:
        sql.append("AND substr(d.published_at, 1, 4) = ?")
        params.append(str(year))
    sql.append("ORDER BY d.published_at DESC LIMIT ?")
    params.append(limit)
    rows = conn.execute("\n".join(sql), params).fetchall()
    return [
        FoundDoc(r["doc_id"], r["published_at"], r["source"], r["doc_type"], r["scope"], r["title"],
                 _snippet(r["clean_text"] or "", terms))
        for r in rows
    ]


def show_document(conn: sqlite3.Connection, doc_id: str) -> str:
    row = conn.execute(
        "SELECT d.*, p.status, p.canonical_doc_id, p.clean_text, p.excluded_reason "
        "FROM documents AS d LEFT JOIN processed_documents AS p USING (doc_id) WHERE d.doc_id = ?",
        (doc_id,),
    ).fetchone()
    if row is None:
        return f"문서가 없습니다: {doc_id}"
    status = row["status"] or "미처리"
    if status == "duplicate":
        status += f" → 대표 문서 {row['canonical_doc_id']} 를 정답으로 쓰세요"
    elif status == "excluded":
        status += f" ({row['excluded_reason']}) → 정답 문서로 쓸 수 없습니다"
    return "\n".join([
        f"{row['doc_id']}  {(row['published_at'] or '날짜 미상')[:10]}  {row['source']}  상태: {status}",
        f"제목: {row['title']}",
        f"URL: {row['url']}",
        "",
        row["clean_text"] or row["raw_text"] or "(본문 없음)",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m eval.find_docs", description="평가셋 정답 문서 검색")
    parser.add_argument("terms", nargs="*", help="검색어 (모두 포함)")
    parser.add_argument("--doc", help="이 doc_id 의 전문을 출력")
    parser.add_argument("--scope", choices=["seoul", "national"], help="뉴스 범위")
    parser.add_argument("--year", type=int, help="발행 연도")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    if not args.doc and not args.terms:
        parser.error("검색어 또는 --doc 이 필요합니다")

    try:
        with open_db(read_only=True) as conn:
            if args.doc:
                print(show_document(conn, args.doc))
                return
            found = search(conn, args.terms, scope=args.scope, year=args.year, limit=args.limit)
    except sqlite3.OperationalError as exc:
        raise SystemExit(f"검색 실패: {exc}. python -m indexer prepare 를 먼저 실행했는지 확인하세요.") from exc

    if not found:
        print("검색 결과가 없습니다. 답변 불가 문항 후보이거나, 직접 수집으로 보강할 영역입니다.")
        return
    for doc in found:
        date = (doc.published_at or "날짜 미상")[:10]
        print(f"{doc.doc_id}  {date}  {doc.source}  [{doc.doc_type}/{doc.scope or '-'}]")
        print(f"    {doc.title}")
        print(f"    {doc.snippet}")
    print(f"\n{len(found)}건 (최대 {args.limit}건, 최신순)")


if __name__ == "__main__":
    main()
