"""검색·답변 품질 측정.

    python -m eval.run_eval --tag c500 --collection docs__c500
    python -m eval.run_eval --tag c500 --collection docs__c500 --with-answers
    python -m eval.run_eval --check      # 평가셋 형식·유형 구성·정답 문서 존재 여부만 검사

평가셋 형식 (eval/eval_set.jsonl, 한 줄에 한 문항):
    {"id": "f001", "q": "...", "type": "fact", "gold_doc_ids": ["kr_20260115_1a2b3c4d"], "answer_key": "..."}
    {"id": "u001", "q": "...", "type": "unanswerable", "gold_doc_ids": [], "answer_key": "답변 거절이 정답"}

id 는 실험 간 같은 문항을 대조하는 키이므로 문항을 수정·재정렬해도 바꾸지 않는다.

결과는 eval/results/<YYYYMMDD-HHMM>_<tag>/ 에 저장한다 (가이드 4-9).
    retrieval.jsonl  문항별 검색 결과·정답 순위·점수·응답 시간
    summary.json     설정(컬렉션·임베딩 모델·청킹 설정·k·커밋)과 Hit@1·Hit@3·Hit@k·MRR, 유형별 집계,
                     정답/답변 불가 문항의 최고 유사도 분포 (거절 임계값 설정 근거)
    answers.csv      --with-answers 시 생성 (RAG 체인 구현 후). 직접 채점용

검색은 챗봇과 같은 rag.retriever.Retriever 로 한다. 평가한 검색기와 서비스하는 검색기가 같아야
평가가 유효하다 (가이드 4-2). 검색 결과는 문서 단위(문서당 청크 1개)로 순위를 매긴다.
"""

import argparse
import json
import math
import re
import sqlite3
import subprocess
import time
from collections import Counter, defaultdict
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Protocol

from common.config import PROJECT_ROOT, get_settings
from common.docstore import open_db
from rag.retriever import RetrievedChunk, Retriever

QUESTION_TYPES = ("fact", "comparison", "temporal", "synthesis", "unanswerable")


@dataclass(frozen=True)
class EvalItem:
    id: str
    q: str
    type: str
    gold_doc_ids: tuple[str, ...]
    answer_key: str


def load_eval_set(path: Path) -> list[EvalItem]:
    """평가셋을 읽고 형식을 검증한다. 오류가 있으면 줄 번호와 함께 실패한다."""
    items: list[EvalItem] = []
    seen_ids: set[str] = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            gold = raw["gold_doc_ids"]
            if not isinstance(gold, list) or not all(isinstance(g, str) for g in gold):
                raise TypeError("gold_doc_ids 는 문자열 리스트여야 합니다")
            item = EvalItem(
                id=str(raw["id"]),
                q=str(raw["q"]),
                type=str(raw["type"]),
                gold_doc_ids=tuple(raw["gold_doc_ids"]),
                answer_key=str(raw["answer_key"]),
            )
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"{path}:{line_no}: 형식 오류 ({exc})") from exc
        if item.type not in QUESTION_TYPES:
            raise ValueError(f"{path}:{line_no}: 알 수 없는 type {item.type!r}")
        if (item.type == "unanswerable") != (not item.gold_doc_ids):
            raise ValueError(f"{path}:{line_no}: unanswerable 문항만 gold_doc_ids 가 비어 있어야 합니다")
        if item.id in seen_ids:
            raise ValueError(f"{path}:{line_no}: 중복 id {item.id!r}")
        seen_ids.add(item.id)
        items.append(item)
    return items


def _require_gold(gold_doc_ids: Collection[str]) -> None:
    # 답변 불가 문항을 검색 지표에 섞으면 오답으로 집계되어 수치가 왜곡된다.
    if not gold_doc_ids:
        raise ValueError("gold_doc_ids 가 빈 문항(답변 불가)은 검색 지표에서 제외해야 합니다")


def hit_at_k(ranked_doc_ids: Sequence[str], gold_doc_ids: Collection[str], k: int) -> bool:
    """상위 k개 결과에 정답 문서가 하나라도 있으면 True.

    ranked_doc_ids 는 검색 순위대로 나열한 각 결과의 doc_id 다. 청크 단위 결과를 그대로 넘기면
    같은 문서가 여러 번 나올 수 있으며, 이 경우 k 는 청크 개수 기준이 된다.
    """
    if k <= 0:
        raise ValueError(f"k 는 1 이상이어야 합니다: {k}")
    _require_gold(gold_doc_ids)
    return any(doc_id in gold_doc_ids for doc_id in ranked_doc_ids[:k])


def reciprocal_rank(ranked_doc_ids: Sequence[str], gold_doc_ids: Collection[str]) -> float:
    """첫 정답 문서 순위의 역수. 정답이 없으면 0.0. MRR 은 이 값의 문항 평균이다."""
    _require_gold(gold_doc_ids)
    for rank, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id in gold_doc_ids:
            return 1.0 / rank
    return 0.0


def check_eval_set(path: Path) -> list[str]:
    """형식을 검증하고 유형 구성과 정답 문서 존재 여부를 출력한다. 문제 목록을 반환한다."""
    items = load_eval_set(path)
    counts = Counter(item.type for item in items)
    print(f"{path}: {len(items)}문항")
    for question_type in QUESTION_TYPES:
        share = counts[question_type] / len(items) if items else 0
        print(f"  {question_type:13} {counts[question_type]:3}문항 ({share:.0%})")

    problems = []
    if not items:
        problems.append("평가셋이 비어 있습니다 (eval/README.md 의 절차로 작성)")
    elif not counts["unanswerable"]:
        problems.append("답변 불가(unanswerable) 문항이 없습니다 (가이드 4-3: 반드시 포함)")
    try:
        with open_db(read_only=True) as conn:
            kept = {row[0] for row in conn.execute("SELECT doc_id FROM processed_documents WHERE status = 'kept'")}
    except (FileNotFoundError, sqlite3.OperationalError):
        print("  (원문 DB 또는 인덱서 결과가 없어 정답 문서 확인을 건너뜁니다)")
        kept = None
    if kept is not None:
        for item in items:
            for doc_id in item.gold_doc_ids:
                if doc_id not in kept:
                    problems.append(f"{item.id}: 정답 문서 {doc_id} 가 인덱싱 대상(kept)이 아닙니다")
    for problem in problems:
        print(f"  ! {problem}")
    if not problems:
        print("  문제 없음")
    return problems


class SupportsRetrieve(Protocol):
    def retrieve(self, query: str, *, k: int) -> list[RetrievedChunk]: ...


def evaluate_retrieval(items: Sequence[EvalItem], retriever: SupportsRetrieve, k: int) -> list[dict[str, Any]]:
    """문항별 검색 결과와 지표. 답변 불가 문항은 검색 지표 없이 최고 유사도만 기록한다."""
    records = []
    for item in items:
        started = time.perf_counter()
        chunks = retriever.retrieve(item.q, k=k)
        latency_ms = (time.perf_counter() - started) * 1000
        ranked = [c.doc_id for c in chunks]
        record: dict[str, Any] = {
            "id": item.id,
            "type": item.type,
            "q": item.q,
            "gold_doc_ids": list(item.gold_doc_ids),
            "retrieved": [{"rank": rank, "doc_id": c.doc_id, "score": round(c.score, 4),
                           "title": c.metadata.get("title", ""), "published_at": c.metadata.get("published_at", ""),
                           "doc_type": c.metadata.get("doc_type", "")}
                          for rank, c in enumerate(chunks, start=1)],
            "top_score": round(chunks[0].score, 4) if chunks else None,
            "latency_ms": round(latency_ms, 1),
        }
        if item.gold_doc_ids:
            gold = set(item.gold_doc_ids)
            first = next((rank for rank, doc_id in enumerate(ranked, start=1) if doc_id in gold), None)
            record.update({
                "gold_rank": first,
                "hit@1": hit_at_k(ranked, gold, 1),
                "hit@3": hit_at_k(ranked, gold, 3),
                f"hit@{k}": hit_at_k(ranked, gold, k),
                "rr": reciprocal_rank(ranked, gold),
            })
        records.append(record)
    return records


def _metrics(records: Sequence[dict[str, Any]], k: int) -> dict[str, Any]:
    if not records:
        return {"count": 0}
    return {
        "count": len(records),
        "hit@1": round(mean(r["hit@1"] for r in records), 4),
        "hit@3": round(mean(r["hit@3"] for r in records), 4),
        f"hit@{k}": round(mean(r[f"hit@{k}"] for r in records), 4),
        "mrr": round(mean(r["rr"] for r in records), 4),
    }


def _score_stats(scores: Sequence[float]) -> dict[str, Any]:
    if not scores:
        return {"count": 0}
    return {"count": len(scores), "min": min(scores), "median": round(median(scores), 4), "max": max(scores)}


def summarize(records: Sequence[dict[str, Any]], k: int) -> dict[str, Any]:
    answerable = [r for r in records if r["gold_doc_ids"]]
    unanswerable = [r for r in records if not r["gold_doc_ids"]]
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in answerable:
        by_type[r["type"]].append(r)
    latencies = sorted(r["latency_ms"] for r in records)
    return {
        "overall": _metrics(answerable, k),
        "by_type": {t: _metrics(rs, k) for t, rs in sorted(by_type.items())},
        # 거절 임계값의 근거: 정답을 찾은 문항과 답변 불가 문항의 최고 유사도가 얼마나 갈리는가
        "top_score": {
            "answerable_hit": _score_stats([r["top_score"] for r in answerable if r["hit@1"] and r["top_score"] is not None]),
            "answerable_miss": _score_stats([r["top_score"] for r in answerable if not r[f"hit@{k}"] and r["top_score"] is not None]),
            "unanswerable": _score_stats([r["top_score"] for r in unanswerable if r["top_score"] is not None]),
        },
        # p95 는 최근접 순위 방식 (문항이 적을 때 평균보다 작게 나오지 않도록)
        "latency_ms": {"mean": round(mean(latencies), 1), "p95": latencies[max(math.ceil(0.95 * len(latencies)) - 1, 0)]}
        if latencies else {},
        "unanswerable_count": len(unanswerable),
    }


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
                             text=True, timeout=5, check=True)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def save_results(records: Sequence[dict[str, Any]], summary: dict[str, Any], config: dict[str, Any],
                 tag: str, results_dir: Path) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", tag):
        raise ValueError(f"tag 는 영문·숫자·._- 만 쓸 수 있습니다: {tag!r}")
    out = results_dir / f"{datetime.now():%Y%m%d-%H%M}_{tag}"
    out.mkdir(parents=True, exist_ok=False)
    with (out / "retrieval.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    (out / "summary.json").write_text(
        json.dumps({"config": config, **summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def format_summary(summary: dict[str, Any], k: int) -> str:
    lines = []
    overall = summary["overall"]
    if overall["count"]:
        lines.append(f"[검색 평가] 답변 가능 {overall['count']}문항 · 답변 불가 {summary['unanswerable_count']}문항")
        lines.append(f"  전체        Hit@1 {overall['hit@1']:.3f}  Hit@3 {overall['hit@3']:.3f}  "
                     f"Hit@{k} {overall[f'hit@{k}']:.3f}  MRR {overall['mrr']:.3f}")
        for question_type, m in summary["by_type"].items():
            lines.append(f"  {question_type:11} Hit@1 {m['hit@1']:.3f}  Hit@3 {m['hit@3']:.3f}  "
                         f"Hit@{k} {m[f'hit@{k}']:.3f}  MRR {m['mrr']:.3f}  ({m['count']}문항)")
    else:
        lines.append("[검색 평가] 답변 가능 문항이 없습니다")
    labels = {"answerable_hit": "정답이 1위인 문항", "answerable_miss": f"정답을 {k}위 안에 못 찾은 문항",
              "unanswerable": "답변 불가 문항"}
    for name, stats in summary["top_score"].items():
        if stats["count"]:
            lines.append(f"  1위 유사도 · {labels[name]}: 중앙값 {stats['median']:.3f} (최소 {stats['min']:.3f}, "
                         f"최대 {stats['max']:.3f}, {stats['count']}문항)")
    if summary["latency_ms"]:
        lines.append(f"  검색 응답 시간 평균 {summary['latency_ms']['mean']:.0f}ms · p95 {summary['latency_ms']['p95']:.0f}ms")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m eval.run_eval", description="평가 하네스")
    parser.add_argument("--check", action="store_true", help="평가셋 검사만 수행")
    parser.add_argument("--tag", help="실험 태그 (결과 디렉터리 이름에 사용)")
    parser.add_argument("--collection", help="평가할 Chroma 컬렉션 (기본: CHROMA_COLLECTION)")
    parser.add_argument("--k", type=int, default=5, help="검색 결과 수 (3 이상)")
    parser.add_argument("--with-answers", action="store_true", help="답변 생성 및 채점용 CSV 출력")
    args = parser.parse_args()
    settings = get_settings()
    if args.check:
        raise SystemExit(1 if check_eval_set(settings.eval_set_path) else 0)
    if not args.tag:
        parser.error("--tag 가 필요합니다 (평가 실행 시)")
    if args.k < 3:
        parser.error("--k 는 3 이상이어야 합니다 (Hit@3 계산)")
    if args.with_answers:
        raise SystemExit("답변 생성 평가는 RAG 체인(rag/chain.py) 구현 후 지원합니다.")

    items = load_eval_set(settings.eval_set_path)
    if not items:
        raise SystemExit("평가셋이 비어 있습니다. eval/README.md 의 절차로 작성한 뒤 실행하세요.")
    collection = args.collection or settings.chroma_collection
    try:
        retriever = Retriever(collection)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    records = evaluate_retrieval(items, retriever, args.k)
    summary = summarize(records, args.k)
    config = {
        "tag": args.tag, "collection": collection, "k": args.k,
        "max_chunks_per_doc": retriever.max_chunks_per_doc, "collection_metadata": retriever.collection_metadata,
        "eval_set_items": len(items), "git_commit": _git_commit(),
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
    }
    out = save_results(records, summary, config, args.tag, settings.eval_results_dir)
    print(format_summary(summary, args.k))
    print(f"  결과 저장: {out.relative_to(PROJECT_ROOT)}")

if __name__ == "__main__":
    main()
