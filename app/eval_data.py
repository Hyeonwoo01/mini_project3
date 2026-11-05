"""평가 결과 파일(eval/results/<실험>/) → 대시보드용 표 (Streamlit 비의존, 단위 테스트 대상).

app 계층은 eval 패키지를 import 하지 않는다 (계층 분리). 하네스가 남긴 파일만 읽는다.
- summary.json     실험 설정·검색 지표 요약
- retrieval.jsonl  문항별 검색 결과 (hit@k, rr, top_score, latency_ms)
- answers.csv      문항별 답변과 직접 채점 (grade: 정확/부분/오답, citation_ok: Y/N/-)
"""

import json
from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd

from app.experiments import (LAW_QUESTION_IDS, PRIMARY_QUESTION_IDS, REPORT_QUESTION_IDS, AnswerSetting,
                             RetrievalComparison, RunRef)

GRADES = ("정확", "부분", "오답")
GRADE_SCORE = {"정확": 1.0, "부분": 0.5, "오답": 0.0}
ANSWERABLE_TYPES = ("fact", "temporal", "comparison", "synthesis")
TYPE_LABELS = {"fact": "사실", "temporal": "시점", "comparison": "비교", "synthesis": "종합", "unanswerable": "답변 불가"}
METRICS = ("hit@1", "hit@3", "hit@5", "mrr")
METRIC_LABELS = {"hit@1": "Hit@1", "hit@3": "Hit@3", "hit@5": "Hit@5", "mrr": "MRR"}
QUESTION_SETS = {"news": "뉴스 문항", "primary": "1차 자료 문항", "report": "보고서 문항", "law": "법령 문항", "all": "전체"}
THRESHOLD_GROUPS = ("정답 1위", "정답 2~5위", "정답 못 찾음", "답변 불가")
DOC_TYPE_ORDER = ("news", "press_release", "announcement", "policy_report", "law", "blog")


class ResultsMissing(FileNotFoundError):
    """대시보드가 참조하는 결과 파일이 없다 (experiments.py 와 eval/results 가 어긋남)."""


def _run_file(results_dir: Path, run_id: str, name: str) -> Path:
    path = results_dir / run_id / name
    if not path.exists():
        raise ResultsMissing(f"평가 결과 파일이 없습니다: {path.relative_to(results_dir.parent.parent)}")
    return path


def load_summary(results_dir: Path, run_id: str) -> dict:
    return json.loads(_run_file(results_dir, run_id, "summary.json").read_text(encoding="utf-8"))


def load_retrieval(results_dir: Path, run_id: str) -> pd.DataFrame:
    path = _run_file(results_dir, run_id, "retrieval.jsonl")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    columns = ["id", "type", "q", "hit@1", "hit@3", "hit@5", "rr", "top_score", "gold_rank", "latency_ms"]
    return pd.DataFrame([{c: r.get(c) for c in columns} for r in records], columns=columns)


def load_answers(results_dir: Path, run_id: str) -> pd.DataFrame:
    frame = pd.read_csv(_run_file(results_dir, run_id, "answers.csv"), encoding="utf-8-sig", dtype=str,
                        keep_default_na=False)
    for column in ("refused", "retrieval_refused"):
        frame[column] = frame[column].str.strip().str.lower() == "true"
    frame["grade"] = frame["grade"].str.strip()
    frame["citation_ok"] = frame["citation_ok"].str.strip()
    return frame[["id", "type", "q", "grade", "refused", "retrieval_refused", "citation_ok"]]


# ---- 검색 ----

def retrieval_table(results_dir: Path, runs: Iterable[RunRef]) -> pd.DataFrame:
    """실험별 검색 지표 (답변 가능 문항 기준)와 평균 검색 시간."""
    rows = []
    for run in runs:
        summary = load_summary(results_dir, run.run_id)
        overall = summary["overall"]
        rows.append({"실험": run.label, "문항": overall["count"], **{METRIC_LABELS[m]: overall[m] for m in METRICS},
                     "평균 검색 시간(ms)": summary.get("latency_ms", {}).get("mean"), "결과 폴더": run.run_id})
    return pd.DataFrame(rows)


def comparison_metrics(results_dir: Path, comparison: RetrievalComparison) -> pd.DataFrame:
    """(설정, 역할, 지표, 값) — 역할은 baseline / candidate."""
    rows = []
    for role, run in (("baseline", comparison.baseline), ("candidate", comparison.candidate)):
        overall = load_summary(results_dir, run.run_id)["overall"]
        rows += [{"setting": run.label, "role": role, "metric": METRIC_LABELS[m], "value": overall[m]} for m in METRICS]
    return pd.DataFrame(rows)


def comparison_by_type(results_dir: Path, comparison: RetrievalComparison, metric: str = "hit@3") -> pd.DataFrame:
    """(설정, 역할, 유형, 값, 문항 수). 비교 실험에 없는 유형은 뺀다."""
    rows = []
    for role, run in (("baseline", comparison.baseline), ("candidate", comparison.candidate)):
        by_type = load_summary(results_dir, run.run_id)["by_type"]
        rows += [{"setting": run.label, "role": role, "type": TYPE_LABELS[t], "value": by_type[t][metric],
                  "count": by_type[t]["count"]} for t in ANSWERABLE_TYPES if t in by_type]
    return pd.DataFrame(rows)


def latency_points(results_dir: Path, comparison: RetrievalComparison) -> pd.DataFrame:
    """문항별 검색 시간 (설정, 역할, 문항, ms)."""
    frames = []
    for role, run in (("baseline", comparison.baseline), ("candidate", comparison.candidate)):
        frame = load_retrieval(results_dir, run.run_id)[["id", "latency_ms"]].copy()
        frame["setting"], frame["role"] = run.label, role
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def threshold_points(results_dir: Path, run_id: str) -> pd.DataFrame:
    """문항별 검색 1위 유사도와 검색 결과 구분 (run_eval.summarize 의 top_score 와 같은 기준)."""
    frame = load_retrieval(results_dir, run_id)
    frame = frame[frame["top_score"].notna()].copy()

    def group(row: pd.Series) -> str:
        if row["type"] == "unanswerable":
            return "답변 불가"
        if row["hit@1"]:
            return "정답 1위"
        return "정답 2~5위" if row["hit@5"] else "정답 못 찾음"

    frame["group"] = frame.apply(group, axis=1)
    return frame[["id", "type", "q", "top_score", "gold_rank", "group"]].reset_index(drop=True)


# ---- 답변 ----

def setting_frames(results_dir: Path, setting: AnswerSetting, question_set: str) -> list[pd.DataFrame]:
    """반복 실행별 채점 표. question_set: news(뉴스 문항) | primary(1차 자료 문항) | report(보고서 문항) | law(법령 문항) | all.
    실행 1회분의 결과 폴더들을 이어 붙인 뒤 문항 묶음으로 거른다."""
    if question_set not in QUESTION_SETS:
        raise ValueError(f"알 수 없는 문항 묶음: {question_set!r}")
    frames = []
    for run_ids in setting.repetitions:
        frame = pd.concat([load_answers(results_dir, run_id) for run_id in run_ids], ignore_index=True)
        duplicated = sorted(set(frame.loc[frame["id"].duplicated(), "id"]))
        if duplicated:
            raise ValueError(f"{setting.key}: 한 번의 실행에 같은 문항이 두 번 있습니다 ({', '.join(duplicated)})")
        kind = frame["id"].map(lambda i: "report" if i in REPORT_QUESTION_IDS else "law" if i in LAW_QUESTION_IDS
                               else "primary" if i in PRIMARY_QUESTION_IDS else "news")
        frame = frame.assign(question_set=kind)
        if question_set != "all":
            frame = frame[frame["question_set"] == question_set].reset_index(drop=True)
        frames.append(frame)
    return frames


def common_questions(*settings: Sequence[pd.DataFrame]) -> list[list[pd.DataFrame]]:
    """설정마다 반복 실행 표를 받아, 모든 설정·실행에 있는 문항만 남긴다 (평가셋 버전이 다른 설정을 같은 문항으로 비교)."""
    ids = set.intersection(*(set(frame["id"]) for frames in settings for frame in frames))
    return [[frame[frame["id"].isin(ids)].reset_index(drop=True) for frame in frames] for frames in settings]


def answer_scores(frame: pd.DataFrame) -> dict[str, float | int | None]:
    """한 번의 실행: 정답률·부분 포함·점수(정확 1, 부분 0.5)·출처 정확도·거절 정확도·잘못된 거절률."""
    answerable = frame[frame["type"] != "unanswerable"]
    unanswerable = frame[frame["type"] == "unanswerable"]
    judged = answerable[answerable["citation_ok"].isin(["Y", "N"])]
    n = len(answerable)

    def rate(hits: int, total: int) -> float | None:
        return hits / total if total else None

    return {
        "answerable": n,
        "correct": int((answerable["grade"] == "정확").sum()),
        "partial": int((answerable["grade"] == "부분").sum()),
        "wrong": int((answerable["grade"] == "오답").sum()),
        "accuracy": rate(int((answerable["grade"] == "정확").sum()), n),
        "accuracy_with_partial": rate(int(answerable["grade"].isin(["정확", "부분"]).sum()), n),
        "score": rate(float(answerable["grade"].map(GRADE_SCORE).fillna(0).sum()), n),
        "citation_accuracy": rate(int((judged["citation_ok"] == "Y").sum()), len(judged)),
        "refusal_accuracy": rate(int(unanswerable["refused"].sum()), len(unanswerable)),
        "false_refusal_rate": rate(int(answerable["refused"].sum()), n),
    }


def setting_summary(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """반복 실행별 점수 표 (행: 1회, 2회, …)."""
    rows = [{"실행": f"{i}회", **answer_scores(frame)} for i, frame in enumerate(frames, start=1)]
    return pd.DataFrame(rows)


def mean_scores(summary: pd.DataFrame) -> dict[str, float | None]:
    """반복 실행 평균. 값이 없는(None) 지표는 None."""
    result: dict[str, float | None] = {}
    for column in ("accuracy", "accuracy_with_partial", "score", "citation_accuracy", "refusal_accuracy",
                   "false_refusal_rate", "correct", "partial", "wrong"):
        values = summary[column].dropna()
        result[column] = float(values.mean()) if len(values) else None
    result["accuracy_min"] = float(summary["accuracy"].min()) if len(summary) else None
    result["accuracy_max"] = float(summary["accuracy"].max()) if len(summary) else None
    return result


def grade_shares_by_type(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """유형별 등급 비율 (반복 실행을 합쳐 문항·실행 단위로 센다). 열: type, grade, count, total, share."""
    pooled = pd.concat(frames, ignore_index=True)
    pooled = pooled[pooled["type"].isin(ANSWERABLE_TYPES)]
    rows = []
    for type_ in ANSWERABLE_TYPES:
        subset = pooled[pooled["type"] == type_]
        if subset.empty:
            continue
        for grade in GRADES:
            count = int((subset["grade"] == grade).sum())
            rows.append({"type": TYPE_LABELS[type_], "grade": grade, "count": count, "total": len(subset),
                         "share": count / len(subset)})
    return pd.DataFrame(rows)


def grade_matrix(runs_by_setting: dict[str, Sequence[pd.DataFrame]]) -> pd.DataFrame:
    """문항 × (설정, 실행) 등급 표. 답변 가능 문항만.
    - varies: 같은 설정 안에서 실행마다 등급이 다른 문항 (생성 변동)
    - differs: 설정 사이 평균 점수가 다른 문항
    - score_gap: 첫 설정 평균 점수 − 둘째 설정 평균 점수 (설정이 2개일 때)"""
    columns: dict[str, pd.Series] = {}
    groups: dict[str, list[str]] = {}
    meta = None
    for label, frames in runs_by_setting.items():
        for i, frame in enumerate(frames, start=1):
            answerable = frame[frame["type"] != "unanswerable"].set_index("id")
            column = f"{label} {i}회"
            columns[column] = answerable["grade"]
            groups.setdefault(label, []).append(column)
            if meta is None:
                meta = answerable[["type", "q"]]
    if meta is None:
        return pd.DataFrame()
    matrix = pd.concat([meta, pd.DataFrame(columns)], axis=1)
    means = {label: matrix[cols].apply(lambda col: col.map(GRADE_SCORE)).mean(axis=1) for label, cols in groups.items()}
    matrix["varies"] = pd.concat([matrix[cols].nunique(axis=1) > 1 for cols in groups.values()], axis=1).any(axis=1)
    mean_frame = pd.DataFrame(means)
    matrix["differs"] = mean_frame.nunique(axis=1) > 1 if len(means) > 1 else False
    labels = list(means)
    matrix["score_gap"] = (mean_frame[labels[0]] - mean_frame[labels[1]]) if len(labels) == 2 else float("nan")
    matrix["type"] = matrix["type"].map(TYPE_LABELS)
    return matrix.reset_index().rename(columns={"index": "id"})


def context_comparison_table(results_dir: Path, runs: Sequence[RunRef]) -> pd.DataFrame:
    """단일 실행 답변 실험의 점수 비교 (예: 문서당 청크 1개 vs 2개)."""
    rows = []
    for run in runs:
        scores = answer_scores(load_answers(results_dir, run.run_id))
        latency = load_summary(results_dir, run.run_id).get("answers", {}).get("latency_ms", {}).get("mean")
        rows.append({"설정": run.label, **scores, "평균 응답 시간(ms)": latency})
    return pd.DataFrame(rows)


def answer_latency(results_dir: Path, setting: AnswerSetting) -> dict[str, float | None]:
    """답변 생성까지의 응답 시간: 결과 폴더별 평균·p95 를 문항 수로 가중 평균한다 (p95 는 근사)."""
    total = weighted_mean = weighted_p95 = 0.0
    for run_ids in setting.repetitions:
        for run_id in run_ids:
            summary = load_summary(results_dir, run_id)
            latency = summary.get("answers", {}).get("latency_ms", {})
            if not latency:
                continue
            n = summary["config"].get("eval_set_items") or 1
            total += n
            weighted_mean += latency["mean"] * n
            weighted_p95 += latency["p95"] * n
    if not total:
        return {"mean": None, "p95": None}
    return {"mean": weighted_mean / total, "p95": weighted_p95 / total}


# ---- 지식베이스 구성 (common.docstore 조회 결과 → 표) ----

def status_table(rows: Iterable[tuple[str, str | None, int]], labels: dict[str, str]) -> pd.DataFrame:
    """문서 종류별 수집·중복·제외·인덱싱 대상 건수."""
    frame = pd.DataFrame(list(rows), columns=["doc_type", "status", "count"])
    if frame.empty:
        return pd.DataFrame(columns=["문서 종류", "수집", "인덱싱 대상", "중복", "품질 제외", "처리 전"])
    frame["status"] = frame["status"].fillna("unprocessed")
    pivot = frame.pivot_table(index="doc_type", columns="status", values="count", aggfunc="sum", fill_value=0)
    for status in ("kept", "duplicate", "excluded", "unprocessed"):
        if status not in pivot:
            pivot[status] = 0
    order = [t for t in DOC_TYPE_ORDER if t in pivot.index] + [t for t in pivot.index if t not in DOC_TYPE_ORDER]
    pivot = pivot.loc[order]
    return pd.DataFrame({
        "문서 종류": [labels.get(t, t) for t in pivot.index],
        "수집": pivot[["kept", "duplicate", "excluded", "unprocessed"]].sum(axis=1).astype(int).to_numpy(),
        "인덱싱 대상": pivot["kept"].astype(int).to_numpy(),
        "중복": pivot["duplicate"].astype(int).to_numpy(),
        "품질 제외": pivot["excluded"].astype(int).to_numpy(),
        "처리 전": pivot["unprocessed"].astype(int).to_numpy(),
    })


def monthly_table(rows: Iterable[tuple[str, str | None, int]], labels: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(발행 월별 건수, 발행일 미상 건수). 월별 표 열: doc_type(라벨), month(datetime), count."""
    frame = pd.DataFrame(list(rows), columns=["doc_type", "month", "count"])
    undated = frame[frame["month"].isna()].groupby("doc_type", as_index=False)["count"].sum()
    dated = frame[frame["month"].notna()].copy()
    dated["month"] = pd.to_datetime(dated["month"], format="%Y-%m", errors="coerce")
    dated = dated[dated["month"].notna()]
    order = {t: i for i, t in enumerate(DOC_TYPE_ORDER)}
    dated = dated.sort_values(["doc_type", "month"], key=lambda s: s.map(order) if s.name == "doc_type" else s)
    dated["doc_type"] = dated["doc_type"].map(lambda t: labels.get(t, t))
    undated["doc_type"] = undated["doc_type"].map(lambda t: labels.get(t, t))
    return dated.reset_index(drop=True), undated
