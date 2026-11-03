"""답변 직접 채점 결과 집계와 이전 채점 옮기기.

    python -m eval.grade summary eval/results/<실험>/answers.csv          # 채점 집계 → grades_summary.json
    python -m eval.grade carry eval/results/<이전>/answers.csv eval/results/<새 실험>/answers.csv

answers.csv 의 채점 칸 (가이드 4-3: 답변 품질 채점은 직접 수행한다)
    grade        정확 | 부분 | 오답   — 답변 불가 문항은 거절했으면 정확
    citation_ok  Y | N | -          — 인용한 문서가 그 주장을 실제로 뒷받침하는가 (거절·답변 불가는 -)
    note         부분·오답의 원인 (검색 실패 / 근거 청크 부족 / 잘못된 거절 / 시점 혼동 / 지어냄 ...)

carry 는 개선 실험마다 30문항을 다시 채점하지 않도록, 답변 문장이 이전 실험과 똑같은 문항의 채점을 옮긴다.
인용 출처(sources)까지 같으면 citation_ok 도 옮기고, 출처가 바뀌었으면 grade 만 옮긴다.
엑셀에서 저장할 때는 'CSV UTF-8' 형식으로 저장한다 (일반 CSV 는 한글이 깨질 수 있다).
"""

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from eval.run_eval import ANSWER_COLUMNS

GRADES = ("정확", "부분", "오답")
CITATIONS = ("Y", "N", "-")
_CHUNK_COUNT = re.compile(r", 청크 \d+개")


def _source_docs(sources: str) -> str:
    """출처 비교용. 근거 확장 실험의 '청크 n개' 표시는 인용 문서가 같으면 판정에 영향이 없으므로 뺀다."""
    return _CHUNK_COUNT.sub("", sources).strip()


def read_answers(path: Path) -> list[dict[str, str]]:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        raise SystemExit(f"{path} 를 UTF-8 로 읽을 수 없습니다. 엑셀에서 'CSV UTF-8(쉼표로 분리)' 형식으로 다시 저장하세요.")
    return list(csv.DictReader(text.splitlines(keepends=True)))


def write_answers(path: Path, rows: list[dict[str, str]]) -> None:
    fields = list(rows[0].keys()) if rows else ANSWER_COLUMNS
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate(rows: list[dict[str, str]]) -> list[str]:
    problems = []
    for row in rows:
        grade, citation = row.get("grade", "").strip(), row.get("citation_ok", "").strip()
        if grade not in GRADES:
            problems.append(f"{row['id']}: grade 가 비었거나 잘못됨 ({grade!r}, {'/'.join(GRADES)} 중 하나)")
        if citation not in CITATIONS:
            problems.append(f"{row['id']}: citation_ok 가 비었거나 잘못됨 ({citation!r}, {'/'.join(CITATIONS)} 중 하나)")
    return problems


def _rate(n: int, total: int) -> float | None:
    return round(n / total, 4) if total else None


def summarize_grades(rows: list[dict[str, str]]) -> dict:
    answerable = [r for r in rows if r["type"] != "unanswerable"]
    unanswerable = [r for r in rows if r["type"] == "unanswerable"]
    by_type: dict[str, Counter] = defaultdict(Counter)
    for r in answerable:
        by_type[r["type"]][r["grade"].strip()] += 1
    counts = Counter(r["grade"].strip() for r in answerable)
    cited = [r["citation_ok"].strip() for r in rows if r["citation_ok"].strip() in ("Y", "N")]
    return {
        "answerable": {
            "count": len(answerable),
            "grades": {g: counts[g] for g in GRADES},
            "accuracy": _rate(counts["정확"], len(answerable)),                       # 정확만
            "accuracy_with_partial": _rate(counts["정확"] + counts["부분"], len(answerable)),
            "by_type": {t: {"count": sum(c.values()), **{g: c[g] for g in GRADES},
                            "accuracy": _rate(c["정확"], sum(c.values()))}
                        for t, c in sorted(by_type.items())},
        },
        "unanswerable": {"count": len(unanswerable),
                         "correct": sum(r["grade"].strip() == "정확" for r in unanswerable)},
        "citation": {"judged": len(cited), "ok": cited.count("Y"), "accuracy": _rate(cited.count("Y"), len(cited))},
        "issues": [{"id": r["id"], "grade": r["grade"].strip(), "note": r.get("note", "").strip()}
                   for r in answerable if r["grade"].strip() != "정확"],
    }


def format_grades(summary: dict) -> str:
    a, u, c = summary["answerable"], summary["unanswerable"], summary["citation"]
    pct = lambda v: "-" if v is None else f"{v:.1%}"  # noqa: E731
    lines = [f"[직접 채점] 답변 가능 {a['count']}문항: 정확 {a['grades']['정확']} · 부분 {a['grades']['부분']} · "
             f"오답 {a['grades']['오답']} → 정답률 {pct(a['accuracy'])} (부분 포함 {pct(a['accuracy_with_partial'])})"]
    for t, s in a["by_type"].items():
        lines.append(f"  {t:12} 정확 {s['정확']} · 부분 {s['부분']} · 오답 {s['오답']}  ({s['count']}문항, 정답률 {pct(s['accuracy'])})")
    lines.append(f"  답변 불가 {u['count']}문항 중 올바른 거절 {u['correct']}")
    lines.append(f"  출처 정확도 {pct(c['accuracy'])} (판정 {c['judged']}건 중 Y {c['ok']}건)")
    if summary["issues"]:
        lines.append("  부분·오답:")
        lines += [f"    {i['id']} {i['grade']}: {i['note']}" for i in summary["issues"]]
    return "\n".join(lines)


def carry_grades(previous: list[dict[str, str]], current: list[dict[str, str]]) -> tuple[int, int, list[str]]:
    """답변이 같은 문항의 채점을 옮긴다. (옮긴 문항 수, 그중 citation_ok 까지 옮긴 수, 새로 채점할 id) 반환."""
    prev = {r["id"]: r for r in previous}
    carried = with_citation = 0
    todo = []
    for row in current:
        old = prev.get(row["id"])
        if old and old["answer"].strip() == row["answer"].strip() and old.get("grade", "").strip():
            row["grade"], row["note"] = old["grade"], old.get("note", "")
            carried += 1
            # 같은 답변이면서 인용 문서가 같거나, 인용 판정 대상이 아니었던(-) 경우 citation_ok 도 옮긴다
            if old.get("citation_ok", "").strip() == "-" or _source_docs(old.get("sources", "")) == _source_docs(row.get("sources", "")):
                row["citation_ok"] = old.get("citation_ok", "")
                with_citation += 1
        if not row.get("grade", "").strip() or not row.get("citation_ok", "").strip():
            todo.append(row["id"])
    return carried, with_citation, todo


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m eval.grade", description="답변 직접 채점 집계")
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("summary", help="채점 집계 (grades_summary.json 저장)")
    s.add_argument("answers", type=Path)
    c = sub.add_parser("carry", help="답변이 같은 문항의 채점을 이전 실험에서 옮긴다")
    c.add_argument("previous", type=Path)
    c.add_argument("current", type=Path)
    args = parser.parse_args()

    if args.command == "summary":
        rows = read_answers(args.answers)
        problems = validate(rows)
        if problems:
            print("채점이 끝나지 않은 칸이 있습니다:")
            print("\n".join(f"  {p}" for p in problems))
            raise SystemExit(1)
        summary = summarize_grades(rows)
        out = args.answers.with_name("grades_summary.json")
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(format_grades(summary))
        print(f"  저장: {out}")
    else:
        previous, current = read_answers(args.previous), read_answers(args.current)
        carried, with_citation, todo = carry_grades(previous, current)
        write_answers(args.current, current)
        print(f"채점 옮김: {carried}문항 (그중 출처까지 같아 citation_ok 도 옮긴 문항 {with_citation})")
        print(f"새로 채점할 문항 {len(todo)}개: {' '.join(todo) if todo else '없음'}")


if __name__ == "__main__":
    main()
