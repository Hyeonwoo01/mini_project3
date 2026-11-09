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
    answers.csv      --with-answers 시 생성. 챗봇과 같은 rag.chain 으로 만든 답변과 근거 목록.
                     grade(정확·부분 정확·오답)와 citation_ok(출처 정확성)는 직접 채점한다 (가이드 4-3).
                     거절 정확도·잘못된 거절은 summary.json 에 자동 집계한다.

검색은 챗봇과 같은 rag.retriever.Retriever 로 한다. 평가한 검색기와 서비스하는 검색기가 같아야
평가가 유효하다 (가이드 4-2). 검색 결과는 문서 단위(문서당 청크 1개)로 순위를 매긴다.
"""

import argparse
import csv
import json
import math
import re
import sqlite3
import subprocess
import time
from collections import Counter, defaultdict
from collections.abc import Collection, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Protocol

from common.config import PROJECT_ROOT, RETRIEVAL_MODES, get_settings
from common.docstore import open_db
from common.schema import DocType
from rag.chain import DEFAULT_DECOMPOSE, RagChain, RagResult, get_chain
from rag.prompts import DEFAULT_PROMPT_VERSION, PROMPT_VERSIONS
from rag.retriever import HYBRID_CANDIDATES, RRF_K, RetrievedChunk, Retriever, SearchFilters

QUESTION_TYPES = ("fact", "comparison", "temporal", "synthesis", "unanswerable")


@dataclass(frozen=True)
class EvalItem:
    id: str
    q: str
    type: str
    gold_doc_ids: tuple[str, ...]
    answer_key: str
    # 근거 그룹: 대상·입장별 정답 문서 (그룹 이름, doc_id 들). 비교 문항(대상별)·종합 문항(입장별)처럼 근거가 둘 이상
    # 필요한 문항에 적고, 모든 그룹의 근거를 찾았는지(cover@k·all@k) 잰다. 한 문서가 여러 그룹에 들어갈 수 있다
    gold_groups: tuple[tuple[str, tuple[str, ...]], ...] = ()


def _parse_groups(raw: Any, gold: Sequence[str]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if raw is None:
        return ()
    if not isinstance(raw, dict) or len(raw) < 2:
        raise TypeError("gold_groups 는 대상 이름 → doc_id 리스트 사전이고 대상이 둘 이상이어야 합니다")
    groups = []
    for name, ids in raw.items():
        if not isinstance(ids, list) or not ids or not all(isinstance(i, str) for i in ids):
            raise TypeError(f"gold_groups[{name!r}] 는 비어 있지 않은 문자열 리스트여야 합니다")
        outside = sorted(set(ids) - set(gold))
        if outside:
            raise TypeError(f"gold_groups[{name!r}] 에 gold_doc_ids 밖의 문서가 있습니다: {', '.join(outside)}")
        groups.append((str(name), tuple(ids)))
    return tuple(groups)


def load_eval_set(path: Path) -> list[EvalItem]:
    """평가셋을 읽고 형식을 검증한다. 오류가 있으면 줄 번호와 함께 실패한다."""
    items: list[EvalItem] = []
    seen_ids: set[str] = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
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
                gold_groups=_parse_groups(raw.get("gold_groups"), gold),
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


def group_coverage(ranked_doc_ids: Sequence[str], groups: Sequence[tuple[str, Sequence[str]]], k: int) -> float:
    """상위 k개 안에 정답 문서가 하나라도 있는 대상의 비율 (비교 문항: 양쪽 근거를 모두 찾았는가)."""
    if k <= 0:
        raise ValueError(f"k 는 1 이상이어야 합니다: {k}")
    if not groups:
        raise ValueError("gold_groups 가 없는 문항입니다")
    top = set(ranked_doc_ids[:k])
    return sum(1 for _, ids in groups if top & set(ids)) / len(groups)


def _retrieval_metrics(item: EvalItem, ranked: Sequence[str], k: int) -> dict[str, Any]:
    """문항 하나의 검색 지표. 답변 불가 문항은 빈 사전."""
    if not item.gold_doc_ids:
        return {}
    gold = set(item.gold_doc_ids)
    metrics: dict[str, Any] = {
        "gold_rank": next((rank for rank, doc_id in enumerate(ranked, start=1) if doc_id in gold), None),
        "hit@1": hit_at_k(ranked, gold, 1),
        "hit@3": hit_at_k(ranked, gold, 3),
        f"hit@{k}": hit_at_k(ranked, gold, k),
        "rr": reciprocal_rank(ranked, gold),
    }
    if item.gold_groups:
        metrics.update({
            "cover@3": group_coverage(ranked, item.gold_groups, 3),
            f"cover@{k}": group_coverage(ranked, item.gold_groups, k),
            "groups_found": [name for name, ids in item.gold_groups if set(ranked[:k]) & set(ids)],
        })
    return metrics


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
    def retrieve(self, query: str, *, k: int, filters: SearchFilters | None = None) -> list[RetrievedChunk]: ...


# 하이브리드 검색의 결합 정보와, 질문 분해 검색에서 그 문서를 찾은 하위 질의
FUSION_FIELDS = ("vector_rank", "keyword_rank", "rrf_score", "sub_query")


def evaluate_retrieval(items: Sequence[EvalItem], retriever: SupportsRetrieve, k: int,
                       filters: SearchFilters | None = None,
                       progress=lambda message: None) -> list[dict[str, Any]]:
    """문항별 검색 결과와 지표. 답변 불가 문항은 검색 지표 없이 최고 유사도만 기록한다."""
    records = []
    for number, item in enumerate(items, start=1):
        if number == 1 or number % 10 == 0 or number == len(items):
            progress(f"  검색 [{number}/{len(items)}]")
        started = time.perf_counter()
        chunks = retriever.retrieve(item.q, k=k, filters=filters)
        latency_ms = (time.perf_counter() - started) * 1000
        ranked = [c.doc_id for c in chunks]
        record: dict[str, Any] = {
            "id": item.id,
            "type": item.type,
            "q": item.q,
            "gold_doc_ids": list(item.gold_doc_ids),
            "retrieved": [{"rank": rank, "doc_id": c.doc_id, "score": round(c.score, 4),
                           "title": c.metadata.get("title", ""), "published_at": c.metadata.get("published_at", ""),
                           "doc_type": c.metadata.get("doc_type", ""),
                           # 하이브리드 검색일 때만: 의미 검색·키워드 검색 각각의 문서 순위 (없으면 None)
                           **{key: c.metadata[key] for key in FUSION_FIELDS if key in c.metadata}}
                          for rank, c in enumerate(chunks, start=1)],
            # 거절 임계값과 비교하는 값이므로 순위와 관계없이 가장 높은 유사도 (하이브리드는 1위가 최고 유사도가 아닐 수 있음)
            "top_score": round(max(c.score for c in chunks), 4) if chunks else None,
            "latency_ms": round(latency_ms, 1),
        }
        sub = getattr(retriever, "last_sub_queries", None)
        if sub:
            record["sub_queries"] = list(sub)   # 질문 분해의 하위 질의 (보강 문서가 없어도 남긴다)
        record.update(_retrieval_metrics(item, ranked, k))
        records.append(record)
    return records


def rescore_records(records: Sequence[dict[str, Any]], items: Sequence[EvalItem], k: int) -> list[dict[str, Any]]:
    """저장된 검색 결과를 현재 평가셋의 정답 목록으로 다시 채점한다 (검색은 다시 하지 않는다).

    정답 목록만 고쳤을 때(판정 오류 수정 등) 검색 결과는 그대로이므로, 원문 DB·벡터 DB·API 없이 지표를 다시 낸다.
    """
    by_id = {r["id"]: r for r in records}
    missing = [item.id for item in items if item.id not in by_id]
    if missing:
        raise ValueError(f"저장된 결과에 없는 문항이 있습니다 (검색을 다시 실행해야 함): {', '.join(missing)}")
    rescored = []
    for item in items:
        old = by_id[item.id]
        record = {key: old[key] for key in ("id", "type", "q", "retrieved", "top_score", "latency_ms")}
        record["gold_doc_ids"] = list(item.gold_doc_ids)
        record.update(_retrieval_metrics(item, [hit["doc_id"] for hit in old["retrieved"]], k))
        rescored.append(record)
    return rescored


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
        # 근거 그룹이 있는 문항: 그룹별 근거를 상위 k 안에서 찾은 비율과, 모든 그룹을 찾은 문항 비율.
        # coverage 는 비교 문항만 (종합 문항에 그룹을 더하기 전 실험과 같은 기준), 유형별 값은 coverage_by_type
        "coverage": _coverage([r for r in records if r.get("type") == "comparison"], k),
        "coverage_by_type": {t: _coverage([r for r in records if r.get("type") == t], k)
                             for t in sorted({r.get("type") for r in records if "cover@3" in r})},
    }


def _coverage(records: Sequence[dict[str, Any]], k: int) -> dict[str, Any]:
    grouped = [r for r in records if "cover@3" in r]
    if not grouped:
        return {"count": 0}
    return {
        "count": len(grouped),
        "cover@3": round(mean(r["cover@3"] for r in grouped), 4),
        f"cover@{k}": round(mean(r[f"cover@{k}"] for r in grouped), 4),
        "all@3": round(mean(r["cover@3"] == 1 for r in grouped), 4),
        f"all@{k}": round(mean(r[f"cover@{k}"] == 1 for r in grouped), 4),
    }


class SupportsAnswer(Protocol):
    def answer(self, question: str, *, filters: SearchFilters | None = None) -> RagResult: ...


ANSWER_COLUMNS = ["id", "type", "q", "answer_key", "answer", "refused", "retrieval_refused", "cited", "sources",
                  "gold_doc_ids", "cited_gold_auto", "grade", "citation_ok", "note"]


def _chunk_count(chunk: RetrievedChunk) -> str:
    ids = chunk.metadata.get("context_chunk_ids") or []
    return f", 청크 {len(ids)}개" if len(ids) > 1 else ""


def evaluate_answers(items: Sequence[EvalItem], chain: SupportsAnswer,
                     progress=lambda message: None, filters: SearchFilters | None = None) -> list[dict[str, Any]]:
    """문항별 답변. grade·citation_ok 는 비워 두고 직접 채점한다."""
    rows = []
    for number, item in enumerate(items, start=1):
        result = chain.answer(item.q, filters=filters)
        status = "거절" if result.refused else f"인용 {len(result.cited)}건"
        progress(f"  [{number}/{len(items)}] {item.id} {result.latency_ms / 1000:.1f}초 · {status}")
        cited_docs = {result.chunks[n - 1].doc_id for n in result.cited}
        rows.append({
            "id": item.id,
            "type": item.type,
            "q": item.q,
            "answer_key": item.answer_key,
            "answer": result.answer,
            "refused": result.refused,
            "retrieval_refused": result.retrieval_refused,
            "cited": " ".join(f"[{n}]" for n in result.cited),
            "sources": "\n".join(
                f"[{i}] {c.metadata.get('title', '')} ({str(c.metadata.get('published_at', ''))[:10]}, {c.doc_id}, "
                f"유사도 {c.score:.3f}{_chunk_count(c)})" for i, c in enumerate(result.chunks, start=1)),
            "gold_doc_ids": " ".join(item.gold_doc_ids),
            # 인용한 근거 중 정답 문서가 있는지 (자동 참고값. 최종 출처 정확성은 직접 판단한다)
            "cited_gold_auto": bool(cited_docs & set(item.gold_doc_ids)) if item.gold_doc_ids else None,
            "grade": "", "citation_ok": "", "note": "",
            "latency_ms": round(result.latency_ms, 1),
        })
    return rows


def summarize_answers(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    answerable = [r for r in rows if r["type"] != "unanswerable"]
    unanswerable = [r for r in rows if r["type"] == "unanswerable"]
    answered = [r for r in answerable if not r["refused"]]
    rate = lambda hits, total: round(hits / total, 4) if total else None  # noqa: E731
    latencies = sorted(r["latency_ms"] for r in rows)
    return {
        "refusal_accuracy": rate(sum(r["refused"] for r in unanswerable), len(unanswerable)),
        "false_refusal_rate": rate(sum(r["refused"] for r in answerable), len(answerable)),
        "citation_rate": rate(sum(bool(r["cited"]) for r in answered), len(answered)),
        "cited_gold_rate_auto": rate(sum(bool(r["cited_gold_auto"]) for r in answered), len(answered)),
        "counts": {"answerable": len(answerable), "unanswerable": len(unanswerable), "answered": len(answered)},
        "latency_ms": {"mean": round(mean(latencies), 1), "p95": latencies[max(math.ceil(0.95 * len(latencies)) - 1, 0)]}
        if latencies else {},
    }


def write_answers_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    # 엑셀에서 한글이 깨지지 않도록 BOM 을 붙인다.
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ANSWER_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


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
    answers = summary.get("answers")
    if overall["count"]:
        lines.append(f"[검색 평가] 답변 가능 {overall['count']}문항 · 답변 불가 {summary['unanswerable_count']}문항")
        lines.append(f"  전체        Hit@1 {overall['hit@1']:.3f}  Hit@3 {overall['hit@3']:.3f}  "
                     f"Hit@{k} {overall[f'hit@{k}']:.3f}  MRR {overall['mrr']:.3f}")
        for question_type, m in summary["by_type"].items():
            lines.append(f"  {question_type:11} Hit@1 {m['hit@1']:.3f}  Hit@3 {m['hit@3']:.3f}  "
                         f"Hit@{k} {m[f'hit@{k}']:.3f}  MRR {m['mrr']:.3f}  ({m['count']}문항)")
    else:
        lines.append("[검색 평가] 답변 가능 문항이 없습니다")
    group_labels = {"comparison": "비교 문항 대상별", "synthesis": "종합 문항 입장별"}
    for question_type, coverage in summary.get("coverage_by_type", {}).items():
        lines.append(f"  {group_labels.get(question_type, question_type)} 근거 포함률 cover@3 {coverage['cover@3']:.3f} · "
                     f"cover@{k} {coverage[f'cover@{k}']:.3f} · 모든 그룹 포함 all@3 {coverage['all@3']:.3f} · "
                     f"all@{k} {coverage[f'all@{k}']:.3f} ({coverage['count']}문항)")
    labels = {"answerable_hit": "정답이 1위인 문항", "answerable_miss": f"정답을 {k}위 안에 못 찾은 문항",
              "unanswerable": "답변 불가 문항"}
    for name, stats in summary["top_score"].items():
        if stats["count"]:
            lines.append(f"  1위 유사도 · {labels[name]}: 중앙값 {stats['median']:.3f} (최소 {stats['min']:.3f}, "
                         f"최대 {stats['max']:.3f}, {stats['count']}문항)")
    if summary["latency_ms"]:
        lines.append(f"  검색 응답 시간 평균 {summary['latency_ms']['mean']:.0f}ms · p95 {summary['latency_ms']['p95']:.0f}ms")
    if answers:
        pct = lambda v: "-" if v is None else f"{v:.1%}"  # noqa: E731
        lines.append(f"[답변 평가] 거절 정확도 {pct(answers['refusal_accuracy'])} (답변 불가 문항) · "
                     f"잘못된 거절 {pct(answers['false_refusal_rate'])} (답변 가능 문항)")
        lines.append(f"  출처 표기율 {pct(answers['citation_rate'])} · 정답 문서 인용률(자동) {pct(answers['cited_gold_rate_auto'])}"
                     f" · 응답 시간 평균 {answers['latency_ms'].get('mean', 0) / 1000:.1f}초")
        lines.append("  answers.csv 의 grade(정확/부분/오답)·citation_ok 를 직접 채점하세요.")
    return "\n".join(lines)


def rescore(source: Path, tag: str, settings) -> Path:
    """--rescore: 저장된 검색 결과를 현재 평가셋으로 다시 채점해 새 결과 디렉터리로 저장한다."""
    if not (source / "retrieval.jsonl").is_file():
        raise SystemExit(f"retrieval.jsonl 이 없습니다: {source}")
    old = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    k = old["config"]["k"]
    records = [json.loads(line) for line in (source / "retrieval.jsonl").read_text(encoding="utf-8").splitlines()]
    items = load_eval_set(settings.eval_set_path)
    if old["config"].get("ids"):        # --ids 로 일부 문항만 실행한 결과는 그 문항만 다시 채점한다
        items = [item for item in items if item.id in set(old["config"]["ids"])]
    rescored = rescore_records(records, items, k)
    summary = summarize(rescored, k)
    config = {**old["config"], "tag": tag, "eval_set_items": len(items), "eval_set": settings.eval_set_path.name,
              "git_commit": _git_commit(),
              "rescored_from": source.name, "chain": None, "evaluated_at": datetime.now().isoformat(timespec="seconds")}
    out = save_results(rescored, summary, config, tag, settings.eval_results_dir)
    print(format_summary(summary, k))
    print(f"  다시 채점: {source.name} → {out.relative_to(PROJECT_ROOT)} (검색 결과는 그대로, 정답 목록만 현재 평가셋)")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m eval.run_eval", description="평가 하네스")
    parser.add_argument("--check", action="store_true", help="평가셋 검사만 수행")
    parser.add_argument("--tag", help="실험 태그 (결과 디렉터리 이름에 사용)")
    parser.add_argument("--collection", help="평가할 Chroma 컬렉션 (기본: CHROMA_COLLECTION)")
    parser.add_argument("--k", type=int, default=5, help="검색 결과 수 (3 이상)")
    parser.add_argument("--mode", choices=RETRIEVAL_MODES, help="검색 방식 (기본: RAG_RETRIEVAL_MODE)")
    parser.add_argument("--with-answers", action="store_true", help="답변 생성 및 채점용 CSV 출력")
    parser.add_argument("--context-chunks", type=int, choices=(1, 2, 3),
                        help="답변 근거로 넣는 문서당 청크 수 (기본: RAG_CONTEXT_CHUNKS)")
    parser.add_argument("--prompt-version", choices=PROMPT_VERSIONS,
                        help=f"답변 생성 시스템 프롬프트 버전 (기본: {DEFAULT_PROMPT_VERSION}, rag/prompts.py)")
    parser.add_argument("--seed", type=int, help="LLM seed (OpenAI 재현성 옵션, 반복 실행 변동 확인용)")
    parser.add_argument("--annotate-dates", action=argparse.BooleanOptionalAction, default=None,
                        help="답변 근거 청크의 상대 날짜에 발행일 기준 날짜를 덧붙인다 (기본: rag.chain.DEFAULT_ANNOTATE_DATES). "
                             "검색에는 영향이 없다")
    parser.add_argument("--decompose", action=argparse.BooleanOptionalAction, default=None,
                        help="비교 질문을 대상별 질의로 나눠 검색 (rag/decompose.py, 기본: rag.chain.DEFAULT_DECOMPOSE). "
                             "검색 평가에도 적용된다. 분해 채택 전 실험은 --no-decompose 로 재현한다")
    parser.add_argument("--rescore", type=Path, help="저장된 검색 결과(eval/results/...)를 현재 정답 목록으로 다시 채점")
    parser.add_argument("--eval-set", type=Path,
                        help="평가셋 파일 (기본: eval/eval_set.jsonl). 평가셋 밖 문항은 eval/holdout.jsonl")
    parser.add_argument("--ids", nargs="+", metavar="ID",
                        help="이 문항만 평가한다 (예: f021 t013). 일부 문항의 반복 실행 비교에 쓴다")
    parser.add_argument("--doc-types", nargs="+", choices=[t.value for t in DocType], metavar="TYPE",
                        help="이 문서 종류만 검색한다 (예: news). 같은 컬렉션에서 1차 자료를 뺀 비교 실험에 쓴다")
    args = parser.parse_args()
    settings = get_settings()
    if args.eval_set:
        if not args.eval_set.is_file():
            parser.error(f"평가셋 파일이 없습니다: {args.eval_set}")
        settings = replace(settings, eval_set_path=args.eval_set.resolve())
    if args.check:
        raise SystemExit(1 if check_eval_set(settings.eval_set_path) else 0)
    if not args.tag:
        parser.error("--tag 가 필요합니다 (평가 실행 시)")
    if args.rescore:
        rescore(args.rescore, args.tag, settings)
        return
    if args.k < 3:
        parser.error("--k 는 3 이상이어야 합니다 (Hit@3 계산)")
    decompose = DEFAULT_DECOMPOSE if args.decompose is None else args.decompose
    if (args.prompt_version or args.seed is not None) and not (args.with_answers or decompose):
        parser.error("--prompt-version, --seed 는 --with-answers 와 함께 쓴다 (답변 생성 설정)")
    if args.annotate_dates is not None and not args.with_answers:
        parser.error("--annotate-dates 는 --with-answers 와 함께 쓴다 (답변 근거 설정)")

    items = load_eval_set(settings.eval_set_path)
    if not items:
        raise SystemExit("평가셋이 비어 있습니다. eval/README.md 의 절차로 작성한 뒤 실행하세요.")
    if args.ids:
        unknown = sorted(set(args.ids) - {item.id for item in items})
        if unknown:
            parser.error(f"평가셋에 없는 문항: {', '.join(unknown)}")
        items = [item for item in items if item.id in set(args.ids)]
        print(f"평가 문항: {len(items)}개 ({', '.join(item.id for item in items)})")
    collection = args.collection or settings.chroma_collection
    mode = args.mode or settings.rag_retrieval_mode
    try:
        retriever = Retriever(collection, mode=mode)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    filters = SearchFilters(doc_types=tuple(DocType(t) for t in args.doc_types)) if args.doc_types else None
    if filters:
        print(f"검색 대상 문서 종류: {', '.join(args.doc_types)}")
    chain: RagChain | None = None
    if args.with_answers or decompose:
        try:
            chain = get_chain(collection, mode, args.context_chunks, args.prompt_version, args.seed, decompose,
                              args.annotate_dates)
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
    # 질문 분해는 체인이 감싼 검색기에 있으므로 검색 평가도 그 검색기로 한다 (답변과 같은 하위 질의, 캐시 공유)
    print(f"검색 평가 중: {len(items)}문항" + (" (질문 분해: 비교 질문은 LLM 호출 포함)" if decompose else ""))
    records = evaluate_retrieval(items, chain.retriever if decompose else retriever, args.k, filters,
                                 progress=lambda message: print(message, flush=True))
    summary = summarize(records, args.k)
    answer_rows: list[dict[str, Any]] = []
    if args.with_answers:
        print(f"답변 생성 중: {len(items)}문항 (문항당 수 초)")
        answer_rows = evaluate_answers(items, chain, progress=lambda message: print(message, flush=True),
                                       filters=filters)
        summary["answers"] = summarize_answers(answer_rows)
    config = {
        "tag": args.tag, "collection": collection, "k": args.k, "mode": mode,
        "hybrid": {"candidates": HYBRID_CANDIDATES, "rrf_k": RRF_K} if mode == "hybrid" else None,
        "doc_types": list(args.doc_types) if args.doc_types else None,
        "max_chunks_per_doc": retriever.max_chunks_per_doc, "collapse_same_title": retriever.collapse_same_title,
        "collection_metadata": retriever.collection_metadata,
        "eval_set_items": len(items), "eval_set": settings.eval_set_path.name,
        "ids": list(args.ids) if args.ids else None, "git_commit": _git_commit(),
        "chain": {"top_k": chain.config.top_k, "min_score": chain.config.min_score,
                  "context_chunks": chain.config.context_chunks, "chat_model": settings.chat_model,
                  "prompt_version": chain.config.prompt_version, "seed": args.seed,
                  "decompose": chain.config.decompose, "annotate_dates": chain.config.annotate_dates} if chain else None,
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
    }
    out = save_results(records, summary, config, args.tag, settings.eval_results_dir)
    if answer_rows:
        write_answers_csv(answer_rows, out / "answers.csv")
    print(format_summary(summary, args.k))
    print(f"  결과 저장: {out.relative_to(PROJECT_ROOT)}")

if __name__ == "__main__":
    main()
