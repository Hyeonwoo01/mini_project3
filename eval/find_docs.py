"""평가셋 작성용 문서 검색 — 정답 문서(gold_doc_ids)를 찾는다.

    python -m eval.find_docs 상생페이백 11월
    python -m eval.find_docs 지역화폐 예산 삭감 --year 2022
    python -m eval.find_docs 골목상권 매출 --scope seoul --limit 30
    python -m eval.find_docs --doc news_20251216_1a2b3c4d     # 문서 전문 확인
    python -m eval.find_docs --doc kr_A kr_B seoul_C           # 여러 문서 전문 → data/doc_texts.md
    python -m eval.find_docs --batch data/question_drafts.txt  # 문항 후보별 검색 결과를 파일로
    python -m eval.find_docs 부담경감 크레딧 --doc-type press_release announcement  # 1차 자료만

--batch 파일 형식 (한 줄에 한 문항, # 으로 시작하면 주석):
    f001 | 상생페이백 11월분 환급액과 누적 지급액은? | 상생페이백 11월
결과는 data/eval_candidates.md 에 UTF-8 로 저장한다 (기사 본문 일부가 들어가므로 git 에 올리지 않는다).

- 인덱싱 대상(processed_documents.status = 'kept')만 검색한다. 중복으로 빠진 문서는 대표 문서로
  대체되었으므로, 평가셋에는 여기서 찾은 doc_id 를 적어야 검색 지표가 올바르게 계산된다.
- 검색어는 모두 포함(AND)해야 하며, 제목 또는 정제 본문에 있으면 된다.
- 원문 DB 를 읽기 전용으로 연다.
"""

import argparse
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from common.config import PROJECT_ROOT
from common.docstore import open_db
from common.schema import DocType

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
    doc_types: list[str] | None = None,
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
    if doc_types:
        sql.append(f"AND d.doc_type IN ({', '.join('?' for _ in doc_types)})")
        params += list(doc_types)
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


BATCH_LIMIT = 8
DEFAULT_BATCH_OUT = PROJECT_ROOT / "data" / "eval_candidates.md"
DEFAULT_DOCS_OUT = PROJECT_ROOT / "data" / "doc_texts.md"   # 여러 문서 전문 (기사·보도자료 본문이라 git 에 올리지 않는다)


def write_documents(conn: sqlite3.Connection, doc_ids: list[str], out_path: Path) -> None:
    sections = [f"# 문서 전문 {len(doc_ids)}건", ""]
    for doc_id in doc_ids:
        sections += [f"## {doc_id}", "", show_document(conn, doc_id), ""]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(sections), encoding="utf-8")


def parse_batch_line(line: str) -> tuple[str, str, list[str]] | None:
    """'id | 질문 | 검색어' 한 줄을 읽는다. 빈 줄·주석은 None."""
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = [part.strip() for part in line.split("|")]
    if len(parts) != 3 or not all(parts):
        raise ValueError(f"형식 오류 (id | 질문 | 검색어): {line}")
    return parts[0], parts[1], parts[2].split()


def run_batch(conn: sqlite3.Connection, batch_path: Path, out_path: Path, limit: int = BATCH_LIMIT,
              doc_types: list[str] | None = None) -> int:
    """문항 후보마다 검색해 결과를 마크다운으로 저장한다. 검색 결과가 없는 문항 수를 반환한다."""
    # Windows 메모장은 UTF-8 저장 시 BOM 을 붙이므로 utf-8-sig 로 읽는다
    text = batch_path.read_text(encoding="utf-8-sig")
    entries = [e for line in text.splitlines() if (e := parse_batch_line(line))]
    lines = ["# 평가셋 후보 검색 결과", "",
             "후보 기사를 읽고 답이 실제로 있는지 확인한 뒤 gold_doc_ids 를 정한다 (--doc 으로 전문 확인).", ""]
    empty = 0
    for qid, question, terms in entries:
        found = search(conn, terms, limit=limit, doc_types=doc_types)
        lines += [f"## {qid} — {question}", f"검색어: {' '.join(terms)} · {len(found)}건", ""]
        if not found:
            empty += 1
            lines += ["(검색 결과 없음 — 답변 불가 후보이거나 검색어를 바꿔 볼 것)", ""]
        for doc in found:
            lines.append(f"- `{doc.doc_id}` {(doc.published_at or '날짜 미상')[:10]} {doc.source} "
                         f"[{doc.doc_type}/{doc.scope or '-'}] **{doc.title}**")
            lines.append(f"  - {doc.snippet}")
        lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return empty


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m eval.find_docs", description="평가셋 정답 문서 검색")
    parser.add_argument("terms", nargs="*", help="검색어 (모두 포함)")
    parser.add_argument("--doc", nargs="+", metavar="DOC_ID",
                        help="이 doc_id 의 전문을 출력 (여러 개면 data/doc_texts.md 에 저장)")
    parser.add_argument("--batch", type=Path, help="'id | 질문 | 검색어' 파일의 문항 후보를 한꺼번에 검색")
    parser.add_argument("--out", type=Path, default=DEFAULT_BATCH_OUT, help="--batch 결과 파일")
    parser.add_argument("--scope", choices=["seoul", "national"], help="뉴스 범위")
    parser.add_argument("--year", type=int, help="발행 연도")
    parser.add_argument("--doc-type", nargs="+", choices=[t.value for t in DocType], metavar="TYPE",
                        help="이 문서 종류만 (예: press_release announcement)")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    if not args.doc and not args.terms and not args.batch:
        parser.error("검색어, --doc 또는 --batch 가 필요합니다")
    if args.batch and not args.batch.is_file():
        parser.error(f"--batch 파일이 없습니다: {args.batch} ('id | 질문 | 검색어' 형식으로 먼저 만드세요)")

    try:
        with open_db(read_only=True) as conn:
            if args.doc:
                if len(args.doc) == 1:
                    print(show_document(conn, args.doc[0]))
                else:
                    write_documents(conn, args.doc, DEFAULT_DOCS_OUT)
                    print(f"저장: {DEFAULT_DOCS_OUT} ({len(args.doc)}건)")
                return
            if args.batch:
                try:
                    empty = run_batch(conn, args.batch, args.out, doc_types=args.doc_type)
                except ValueError as exc:
                    raise SystemExit(f"{args.batch}: {exc}") from exc
                print(f"저장: {args.out} (검색 결과 없는 문항 {empty}개)")
                return
            found = search(conn, args.terms, scope=args.scope, year=args.year, doc_types=args.doc_type,
                           limit=args.limit)
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
