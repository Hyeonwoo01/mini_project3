"""대시보드 데이터 (저장소에 커밋된 eval/results 를 그대로 읽는다). 기대값은 README '실험 기록'의 수치."""

import json
from pathlib import Path

import pandas as pd
import pytest

from app import eval_data as ed
from app.experiments import (
    ADOPTED_ANSWER_SETTING,
    ANSWER_COMPARISONS,
    ANSWER_SETTINGS,
    PRIMARY_QUESTION_IDS,
    CONTEXT_CHUNK_COMPARISON,
    RETRIEVAL_COMPARISONS,
    THRESHOLD_RUNS,
)

RESULTS = Path(__file__).resolve().parent.parent / "eval" / "results"


def _setting(key):
    return next(s for s in ANSWER_SETTINGS if s.key == key)


def _all_run_ids():
    for c in RETRIEVAL_COMPARISONS:
        yield c.baseline.run_id
        yield c.candidate.run_id
    for s in ANSWER_SETTINGS:
        for run_ids in s.repetitions:
            yield from run_ids
    yield from (r.run_id for r in CONTEXT_CHUNK_COMPARISON)
    yield from (r.run_id for r in THRESHOLD_RUNS)


@pytest.mark.parametrize("run_id", sorted(set(_all_run_ids())))
def test_catalog_points_to_existing_results(run_id):
    assert (RESULTS / run_id / "summary.json").exists()
    assert (RESULTS / run_id / "retrieval.jsonl").exists()


def test_answer_runs_are_graded():
    for setting in ANSWER_SETTINGS:
        for run_ids in setting.repetitions:
            for run_id in run_ids:
                answers = ed.load_answers(RESULTS, run_id)
                assert set(answers["grade"]) <= set(ed.GRADES), run_id


def test_answer_comparisons_refer_to_known_settings():
    keys = {s.key for s in ANSWER_SETTINGS}
    assert ADOPTED_ANSWER_SETTING in keys
    for comparison in ANSWER_COMPARISONS:
        assert {comparison.baseline, comparison.candidate} <= keys


def test_primary_question_ids_match_primary_runs():
    config = json.loads((RESULTS / "20261104-1127_primary8-kb2-ans-r1" / "summary.json").read_text(encoding="utf-8"))
    assert set(config["config"]["ids"]) == PRIMARY_QUESTION_IDS


def test_combined_runs_split_into_question_sets():
    """평가셋 전체를 한 번에 돌린 실행(38문항)도 뉴스 문항 30 / 1차 자료 문항 8 로 나뉜다."""
    setting = _setting("kb2-p2")
    assert all(len(run_ids) == 1 for run_ids in setting.repetitions)
    news = ed.setting_frames(RESULTS, setting, "news")[0]
    primary = ed.setting_frames(RESULTS, setting, "primary")[0]
    assert (len(news), len(primary)) == (30, 8)
    assert set(primary["id"]) == PRIMARY_QUESTION_IDS


def test_missing_results_raise_clear_error(tmp_path):
    with pytest.raises(ed.ResultsMissing):
        ed.load_summary(tmp_path / "eval" / "results", "nope")


def test_setting_scores_match_readme():
    adopted = ed.mean_scores(ed.setting_summary(ed.setting_frames(RESULTS, _setting("kb2"), "news")))
    assert adopted["accuracy"] == pytest.approx(0.472, abs=1e-3)
    assert adopted["accuracy_with_partial"] == pytest.approx(0.708, abs=1e-3)
    assert adopted["citation_accuracy"] == pytest.approx(0.873, abs=1e-3)
    assert adopted["refusal_accuracy"] == 1.0

    combined = {s.key: ed.mean_scores(ed.setting_summary(ed.setting_frames(RESULTS, s, "all"))) for s in ANSWER_SETTINGS}
    assert combined["kb2"]["accuracy"] == pytest.approx(0.573, abs=1e-3)   # (11.3 + 7) / 32
    assert combined["news"]["accuracy"] == pytest.approx(0.438, abs=1e-3)  # (13 + 1) / 32

    primary = ed.setting_summary(ed.setting_frames(RESULTS, _setting(ADOPTED_ANSWER_SETTING), "primary"))
    assert list(primary["correct"]) == [7, 7, 7] and primary["refusal_accuracy"].isna().all()


def test_grade_matrix_separates_run_variance_from_setting_gap():
    settings = [_setting("kb2"), _setting("news")]
    matrix = ed.grade_matrix({s.label: ed.setting_frames(RESULTS, s, "news") for s in settings})
    assert set(matrix.loc[matrix["varies"], "id"]) == {"f013", "t007", "t012", "c002", "s001", "s006"}
    gaps = matrix.set_index("id")["score_gap"]
    assert gaps["t007"] == pytest.approx(-2 / 3)
    assert gaps["s006"] == 0 and not matrix.set_index("id").loc["s006", "differs"]
    assert gaps.sum() == pytest.approx(-0.83, abs=0.01)


def test_prompt_v2_experiment_matches_readme():
    """프롬프트 v2: 정답률은 v1 과 같고(57.3%), f006 이 3회 모두 잘못된 거절로 바뀌었다."""
    v2 = ed.mean_scores(ed.setting_summary(ed.setting_frames(RESULTS, _setting("kb2-p2"), "all")))
    assert v2["accuracy"] == pytest.approx(0.573, abs=1e-3)
    assert v2["accuracy_with_partial"] == pytest.approx(0.771, abs=1e-3)
    matrix = ed.grade_matrix({"v2": ed.setting_frames(RESULTS, _setting("kb2-p2"), "all"),
                              "v1": ed.setting_frames(RESULTS, _setting("kb2"), "all")}).set_index("id")
    assert list(matrix.loc["f006", ["v2 1회", "v2 2회", "v2 3회"]]) == ["오답"] * 3
    assert matrix.loc["t012", "score_gap"] == pytest.approx(0.5)
    assert matrix["score_gap"].sum() == pytest.approx(1 / 3, abs=0.01)


def test_prompt_v3_experiment_matches_readme():
    """프롬프트 v3: 회귀 없이 정답률 60.4%(v1 57.3%), t007 은 3회 모두 부분."""
    v3 = ed.mean_scores(ed.setting_summary(ed.setting_frames(RESULTS, _setting("kb2-p3"), "all")))
    assert v3["accuracy"] == pytest.approx(0.604, abs=1e-3)
    assert v3["accuracy_with_partial"] == pytest.approx(0.812, abs=1e-3)
    matrix = ed.grade_matrix({"v3": ed.setting_frames(RESULTS, _setting("kb2-p3"), "all"),
                              "v1": ed.setting_frames(RESULTS, _setting("kb2"), "all")}).set_index("id")
    assert list(matrix.loc["t007", ["v3 1회", "v3 2회", "v3 3회"]]) == ["부분"] * 3
    assert (matrix["score_gap"] >= 0).all()   # v1 보다 나빠진 문항 없음
    assert matrix["score_gap"].sum() == pytest.approx(1.5)



def test_decomposition_v2_experiment_matches_readme():
    """분해 v2: 정답률 57.3 → 61.5%, 비교 문항 c001·c002·c004 개선, t011·c003 은 등급이 같고(오답) 거절 대신 틀린 답."""
    dec = ed.setting_frames(RESULTS, _setting("kb2-dec2"), "all")
    scores = ed.mean_scores(ed.setting_summary(dec))
    assert scores["accuracy"] == pytest.approx(0.615, abs=1e-3)
    assert scores["accuracy_with_partial"] == pytest.approx(0.865, abs=1e-3)
    assert scores["false_refusal_rate"] == pytest.approx(1 / 32)
    matrix = ed.grade_matrix({"분해": dec, "v1": ed.setting_frames(RESULTS, _setting("kb2"), "all")}).set_index("id")
    assert list(matrix.loc["c002", ["분해 1회", "분해 2회", "분해 3회"]]) == ["정확"] * 3
    assert matrix.loc[["c001", "c004"], "score_gap"].tolist() == [pytest.approx(0.5)] * 2
    assert matrix.loc[matrix["score_gap"] < 0].index.tolist() == ["f016"]   # 검색 결과가 같은 생성 변동
    assert matrix["score_gap"].sum() == pytest.approx(2.5)
    for frame in dec:
        rows = frame.set_index("id")
        assert not rows.loc[["t011", "c003"], "refused"].any() and (rows.loc[["t011", "c003"], "grade"] == "오답").all()


def test_report_experiment_matches_readme():
    """보고서 추가: 공통 32문항 정답률 61.5 → 62.5%, 보고서 문항 f029 는 제외 3회 오답 → 포함 3회 정확."""
    dec, kb3 = ed.common_questions(ed.setting_frames(RESULTS, _setting("kb2-dec2"), "all"),
                                   ed.setting_frames(RESULTS, _setting("kb3"), "all"))
    assert ed.mean_scores(ed.setting_summary(dec))["accuracy"] == pytest.approx(0.615, abs=1e-3)
    assert ed.mean_scores(ed.setting_summary(kb3))["accuracy"] == pytest.approx(0.625, abs=1e-3)
    assert ed.grade_matrix({"보고서": kb3, "분해": dec})["score_gap"].sum() == pytest.approx(0)
    reports = ed.setting_frames(RESULTS, _setting("kb3"), "report")
    without = ed.setting_frames(RESULTS, _setting("kb3-noreport"), "report")
    assert [list(f.set_index("id").loc[["f028", "f029"], "grade"]) for f in reports] == [["정확", "정확"]] * 3
    assert [list(f.set_index("id").loc[["f028", "f029"], "grade"]) for f in without] == [["정확", "오답"]] * 3
    assert ed.mean_scores(ed.setting_summary(ed.setting_frames(RESULTS, _setting("kb3"), "all")))["accuracy"] \
        == pytest.approx(0.647, abs=1e-3)

def test_answer_latency_is_weighted_by_items():
    latency = ed.answer_latency(RESULTS, _setting("kb2-p2"))
    assert 2000 < latency["mean"] < 3000 and latency["p95"] > latency["mean"]


def test_grade_shares_sum_to_one_per_type():
    shares = ed.grade_shares_by_type(ed.setting_frames(RESULTS, _setting("kb2"), "all"))
    assert list(dict.fromkeys(shares["type"])) == ["사실", "시점", "비교", "종합"]
    assert shares.groupby("type")["share"].sum().round(6).eq(1).all()


def test_context_comparison_matches_readme():
    table = ed.context_comparison_table(RESULTS, CONTEXT_CHUNK_COMPARISON)
    assert list(table["correct"]) == [12, 14]
    assert table["citation_accuracy"].round(3).tolist() == [0.857, 0.952]


def test_comparison_tables():
    hybrid = next(c for c in RETRIEVAL_COMPARISONS if c.key == "hybrid")
    metrics = ed.comparison_metrics(RESULTS, hybrid).set_index(["role", "metric"])["value"]
    assert metrics[("baseline", "Hit@3")] == pytest.approx(0.667, abs=1e-3)
    assert metrics[("candidate", "MRR")] == pytest.approx(0.854, abs=1e-3)
    by_type = ed.comparison_by_type(RESULTS, hybrid)
    assert set(by_type["type"]) == {"사실", "시점", "비교", "종합"}
    latency = ed.latency_points(RESULTS, hybrid)
    assert set(latency["role"]) == {"baseline", "candidate"} and latency["latency_ms"].notna().all()


def test_threshold_points_follow_harness_grouping():
    points = ed.threshold_points(RESULTS, THRESHOLD_RUNS[0].run_id)
    summary = json.loads((RESULTS / THRESHOLD_RUNS[0].run_id / "summary.json").read_text(encoding="utf-8"))
    counts = points["group"].value_counts()
    assert counts["정답 1위"] == summary["top_score"]["answerable_hit"]["count"]
    assert counts["정답 못 찾음"] == summary["top_score"]["answerable_miss"]["count"]
    assert counts["답변 불가"] == summary["unanswerable_count"]


def test_status_and_monthly_tables():
    labels = {"news": "뉴스", "press_release": "보도자료"}
    status = ed.status_table([("press_release", "kept", 3), ("news", "kept", 10), ("news", "duplicate", 2),
                              ("news", None, 1)], labels)
    assert status.to_dict("records") == [
        {"문서 종류": "뉴스", "수집": 13, "인덱싱 대상": 10, "중복": 2, "품질 제외": 0, "처리 전": 1},
        {"문서 종류": "보도자료", "수집": 3, "인덱싱 대상": 3, "중복": 0, "품질 제외": 0, "처리 전": 0},
    ]
    assert ed.status_table([], labels).empty

    monthly, undated = ed.monthly_table([("press_release", "2026-01", 2), ("news", "2025-12", 5),
                                         ("press_release", None, 1)], labels)
    assert monthly["doc_type"].tolist() == ["뉴스", "보도자료"]
    assert monthly["month"].dtype.kind == "M"
    assert undated.to_dict("records") == [{"doc_type": "보도자료", "count": 1}]


def test_setting_frames_rejects_unknown_question_set():
    with pytest.raises(ValueError):
        ed.setting_frames(RESULTS, ANSWER_SETTINGS[0], "everything")


def test_law_experiment_matches_readme():
    """법령 추가: 기존 34문항 67.6% (3회 같음, 등급이 낮아진 문항 없음), 법령 문항 정확 0/12 → 6/12 (f032·f033)."""
    kb3, kb4 = ed.common_questions(ed.setting_frames(RESULTS, _setting("kb3"), "all"),
                                   ed.setting_frames(RESULTS, _setting("kb4"), "all"))
    assert [ed.setting_summary([f])["accuracy"].iloc[0] for f in kb4] == [pytest.approx(23 / 34, abs=1e-3)] * 3
    assert (ed.grade_matrix({"법령": kb4, "보고서": kb3})["score_gap"] >= 0).all()
    laws = ed.setting_frames(RESULTS, _setting("kb4"), "law")
    without = ed.setting_frames(RESULTS, _setting("kb4-nolaw"), "law")
    assert [list(f.set_index("id").loc[["f032", "f033"], "grade"]) for f in laws] == [["정확", "정확"]] * 3
    assert all((f["grade"] != "정확").all() for f in without)
