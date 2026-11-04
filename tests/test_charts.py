from pathlib import Path

import pandas as pd

from app import charts
from app import eval_data as ed
from app.experiments import ANSWER_SETTINGS, RETRIEVAL_COMPARISONS, THRESHOLD_RUNS
from app.theme import COLORS

RESULTS = Path(__file__).resolve().parent.parent / "eval" / "results"


def test_comparison_bars_emphasize_candidate_and_label_values():
    frame = ed.comparison_metrics(RESULTS, RETRIEVAL_COMPARISONS[1])
    fig = charts.comparison_bars(frame)
    baseline, candidate = fig.data
    assert baseline.marker.color == COLORS["chip_gray"] and candidate.marker.color == COLORS["primary"]
    assert list(candidate.text) == [format(v, ".3f") for v in frame[frame["role"] == "candidate"]["value"]]
    assert fig.layout.barcornerradius == 4
    # 행 높이로 막대 두께를 정한다: 지표 4행
    assert fig.layout.height == 4 * charts.ROW_PX + charts.LEGEND_PX + charts.AXIS_PX


def test_grade_stack_uses_semantic_colors_in_fixed_order():
    shares = ed.grade_shares_by_type(ed.setting_frames(RESULTS, ANSWER_SETTINGS[0], "news"))
    fig = charts.grade_stack(shares)
    assert [t.name for t in fig.data] == list(ed.GRADES)
    assert [t.marker.color for t in fig.data] == [COLORS["approve"], COLORS["elective"], COLORS["reject"]]
    assert fig.layout.barmode == "stack" and fig.layout.uniformtext.mode == "hide"


def test_run_dots_separates_identical_runs():
    settings = ANSWER_SETTINGS[:2]
    summaries = {s.label: ed.setting_summary(ed.setting_frames(RESULTS, s, "news")) for s in settings}
    fig = charts.run_dots(summaries, settings[0].label)
    runs = fig.data[0]
    assert len(set(runs.y)) == len(runs.y)          # 같은 값의 실행도 다른 자리에
    assert runs.marker.color == COLORS["primary"] and fig.data[2].marker.color == charts.MUTED


def test_threshold_strip_draws_line_and_escapes_question():
    points = ed.threshold_points(RESULTS, THRESHOLD_RUNS[0].run_id)
    points.loc[0, "q"] = "<b>질문</b>"
    fig = charts.threshold_strip(points, 0.5)
    assert any(shape.x0 == 0.5 for shape in fig.layout.shapes)
    assert "&lt;b&gt;" in fig.data[0].customdata[0][3]


def test_monthly_multiples_one_row_per_type():
    monthly = pd.DataFrame({"doc_type": ["뉴스", "뉴스", "보도자료"],
                            "month": pd.to_datetime(["2025-01", "2025-02", "2025-01"]), "count": [10, 12, 3]})
    fig = charts.monthly_multiples(monthly, ["뉴스", "보도자료", "서울시 공고"])
    assert len(fig.data) == 2 and [a.text for a in fig.layout.annotations] == ["뉴스 (월별 건수)", "보도자료 (월별 건수)"]
