"""정답 문서 보정 (pooling) — 검색 결과에서 정답 목록 밖 문서를 모아 판정하고 평가셋에 반영한다.

    python -m eval.pool eval/results/<기준선1> eval/results/<기준선2>   # 판정표 생성 → data/pool_candidates.md
    python -m eval.pool --apply data/pool_candidates.md                # 체크한 문서를 정답에 추가

정보검색 평가의 pooling 방식이다. 평가셋을 만들 때 정답 문서를 엄격하게 골라서, 내용상 정답인 문서를
검색기가 찾아도 오답으로 집계되는 경우가 있다 (기준선 실패 분석의 '정답 목록 누락').
- 판정 대상은 실험 결과 상위 k 건 중 정답 목록에 없는 문서다. 검색기 순위를 보지 않고 문서 내용만으로
  판정한다: 문항의 answer_key 에 해당하는 사실이 본문에 있으면 정답이다
- 판정표에는 answer_key 의 수치가 나오는 부분을 발췌해 보여 준다 (수치가 없으면 본문 앞부분)
- --apply 는 체크한 문서를 gold_doc_ids 에 추가하고, 체크하지 않은 문서는 eval/pool_judged.json 에
  '정답 아님'으로 기록한다. 다음 실험 결과로 다시 모을 때 이미 판정한 문서는 빠진다
- 판정은 비교할 모든 실험 결과에 똑같이 적용해야 한다. 판정이 끝나면 비교 대상 실험을 모두 다시 측정한다
- 판정표는 기사 본문 일부를 담으므로 git 에 올리지 않는다 (data/ 아래, .gitignore)
"""

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from common.config import PROJECT_ROOT, get_settings
from common.docstore import open_db
from eval.run_eval import load_eval_set

DEFAULT_TOP = 5
DEFAULT_OUT = PROJECT_ROOT / "data" / "pool_candidates.md"
JUDGED_PATH = PROJECT_ROOT / "eval" / "pool_judged.json"
SNIPPET_RADIUS = 90
LEAD_CHARS = 200

_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_HEADER = re.compile(r"^## (\S+)")
_CHECKED = re.compile(r"^- \[([ xX])\] `([^`]+)`")


@dataclass
class PoolEntry:
    doc_id: str
    hits: list[str] = field(default_factory=list)  # "태그 n위"


def answer_numbers(answer_key: str) -> list[str]:
    """판정에 쓸 수치. 연도처럼 어디에나 나오는 값과 한두 자리 숫자는 뺀다."""
    numbers = []
    for n in _NUMBER.findall(answer_key):
        is_year = len(n) == 4 and n.isdigit() and 1990 <= int(n) <= 2035
        if ("." in n or len(n) >= 3) and not is_year:
            numbers.append(n)
    return list(dict.fromkeys(numbers))


def snippet(text: str, numbers: list[str]) -> tuple[str, list[str]]:
    """answer_key 수치가 처음 나오는 부분의 발췌와, 본문에 나오는 수치 목록."""
    found = [n for n in numbers if n in text]
    if not found:
        return text[:LEAD_CHARS].replace("\n", " ") + ("…" if len(text) > LEAD_CHARS else ""), []
    pos = min(text.index(n) for n in found)
    start = max(pos - SNIPPET_RADIUS, 0)
    body = text[start:pos + SNIPPET_RADIUS].replace("\n", " ")
    return ("…" if start else "") + body + "…", found


def load_judged(path: Path = JUDGED_PATH) -> dict[str, list[str]]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def collect_pool(result_dirs: list[Path], top: int, gold: dict[str, set[str]],
                 judged: dict[str, list[str]]) -> dict[str, list[PoolEntry]]:
    """문항별로 상위 top 건 중 정답 목록에도, 이미 판정한 목록에도 없는 문서를 모은다."""
    pool: dict[str, dict[str, PoolEntry]] = {}
    for result_dir in result_dirs:
        tag = result_dir.name.split("_", 1)[-1]
        for line in (result_dir / "retrieval.jsonl").read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            qid = record["id"]
            if not gold.get(qid):
                continue  # 답변 불가 문항은 정답 문서가 없다
            skip = gold[qid] | set(judged.get(qid, []))
            for hit in record["retrieved"][:top]:
                if hit["doc_id"] in skip:
                    continue
                entry = pool.setdefault(qid, {}).setdefault(hit["doc_id"], PoolEntry(hit["doc_id"]))
                entry.hits.append(f"{tag} {hit['rank']}위")
    return {qid: list(entries.values()) for qid, entries in pool.items()}


def write_pool(path: Path, pool: dict[str, list[PoolEntry]], items, result_dirs: list[Path], top: int,
               db_path: Path | None = None) -> int:
    lines = ["# 정답 문서 보정 판정표", "",
             f"대상 실험: {', '.join(d.name for d in result_dirs)} · 상위 {top}건 중 정답 목록 밖 문서",
             "",
             "판정 기준: 문항의 answer_key 에 해당하는 사실이 본문에 있으면 정답. 검색 순위는 판정에 쓰지 않는다.",
             "정답이면 `[ ]` 를 `[x]` 로 바꾼다. 발췌만으로 판단이 어려우면 python -m eval.find_docs --doc <doc_id>",
             f"판정이 끝나면: python -m eval.pool --apply {path.as_posix()}", ""]
    count = 0
    with open_db(db_path, read_only=True) as conn:
        for item in items:
            entries = pool.get(item.id)
            if not entries:
                continue
            numbers = answer_numbers(item.answer_key)
            lines += [f"## {item.id} — {item.q}", f"answer_key: {item.answer_key}", ""]
            for entry in entries:
                row = conn.execute(
                    "SELECT d.title, d.source, d.published_at, p.clean_text FROM documents AS d "
                    "JOIN processed_documents AS p USING (doc_id) WHERE d.doc_id = ?", (entry.doc_id,)).fetchone()
                if row is None:
                    continue
                text, found = snippet(row["clean_text"] or "", numbers)
                lines.append(f"- [ ] `{entry.doc_id}` {(row['published_at'] or '날짜 미상')[:10]} {row['source']} "
                             f"**{row['title']}** ({', '.join(entry.hits)})")
                lines.append(f"  - 본문의 answer_key 수치: {', '.join(found) if found else '없음'}")
                lines.append(f"  - {text}")
                count += 1
            lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return count


def parse_judgments(text: str) -> dict[str, tuple[list[str], list[str]]]:
    """판정표에서 문항별 (정답으로 체크한 문서, 체크하지 않은 문서)를 읽는다."""
    result: dict[str, tuple[list[str], list[str]]] = {}
    qid = None
    for line in text.splitlines():
        if header := _HEADER.match(line):
            qid = header.group(1)
            result.setdefault(qid, ([], []))
        elif (box := _CHECKED.match(line)) and qid:
            result[qid][0 if box.group(1).lower() == "x" else 1].append(box.group(2))
    return result


def apply_judgments(eval_set: Path, judged_path: Path, judgments: dict[str, tuple[list[str], list[str]]]) -> dict[str, int]:
    """체크한 문서를 gold_doc_ids 에 추가하고, 나머지를 '정답 아님'으로 기록한다. 문항별 추가 건수를 반환한다."""
    added: dict[str, int] = {}
    lines = eval_set.read_text(encoding="utf-8-sig").splitlines()
    out = []
    for line in lines:
        if not line.strip():
            out.append(line)
            continue
        row = json.loads(line)
        positive = judgments.get(row["id"], ([], []))[0]
        new = [d for d in positive if d not in row["gold_doc_ids"]]
        if new and row["type"] != "unanswerable":
            row["gold_doc_ids"] = row["gold_doc_ids"] + new
            added[row["id"]] = len(new)
            line = json.dumps(row, ensure_ascii=False)
        out.append(line)
    eval_set.write_text("\n".join(out) + "\n", encoding="utf-8")

    judged = load_judged(judged_path)
    for qid, (_, negative) in judgments.items():
        if negative:
            judged[qid] = sorted(set(judged.get(qid, [])) | set(negative))
    judged_path.write_text(json.dumps(dict(sorted(judged.items())), ensure_ascii=False, indent=1) + "\n",
                           encoding="utf-8")
    return added


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m eval.pool", description="정답 문서 보정 (pooling)")
    parser.add_argument("results", nargs="*", type=Path, help="비교할 실험 결과 디렉터리 (eval/results/...)")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP, help="실험별로 판정할 상위 건수")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="판정표 파일")
    parser.add_argument("--apply", type=Path, help="판정이 끝난 판정표를 평가셋에 반영")
    args = parser.parse_args()
    settings = get_settings()

    if args.apply:
        if not args.apply.is_file():
            parser.error(f"판정표가 없습니다: {args.apply}")
        judgments = parse_judgments(args.apply.read_text(encoding="utf-8"))
        added = apply_judgments(settings.eval_set_path, JUDGED_PATH, judgments)
        checked = sum(len(p) for p, _ in judgments.values())
        rejected = sum(len(n) for _, n in judgments.values())
        print(f"판정 반영: 정답 추가 {sum(added.values())}건 ({len(added)}문항) · 정답 아님 기록 {rejected}건 "
              f"(체크 {checked}건)")
        for qid, n in added.items():
            print(f"  {qid}: +{n}")
        print("다음: python -m eval.run_eval --check 후, 비교할 실험을 모두 다시 측정하세요.")
        return

    if not args.results:
        parser.error("실험 결과 디렉터리 또는 --apply 가 필요합니다")
    missing = [d for d in args.results if not (d / "retrieval.jsonl").is_file()]
    if missing:
        parser.error(f"retrieval.jsonl 이 없습니다: {', '.join(map(str, missing))}")
    items = load_eval_set(settings.eval_set_path)
    if not items:
        raise SystemExit("평가셋이 비어 있습니다.")
    gold = {item.id: set(item.gold_doc_ids) for item in items}
    pool = collect_pool(args.results, args.top, gold, load_judged())
    count = write_pool(args.out, pool, items, args.results, args.top)
    print(f"저장: {args.out} (판정할 문서 {count}건, {len(pool)}문항)")


if __name__ == "__main__":
    main()
