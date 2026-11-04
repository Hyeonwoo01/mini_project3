"""평가 결과 시각화 (가이드 4-8, Plotly).

eval/results/ 의 실험별 결과 파일을 읽어서 표시한다. 어떤 실험을 싣는지는 app/experiments.py.
- 요약: 채택 설정의 정답률(대표 숫자)과 핵심 지표, 실험별 결정 기록
- 검색 실험: 기준 대비 개선안의 Hit@k·MRR, 유형별 Hit@3, 문항별 검색 시간
- 답변 품질: 반복 실행 정답률, 유형별 등급(취약 유형), 문항별 등급 일관성
- 거절 임계값: 검색 1위 유사도 분포 — 임계값 설정 근거
- 지식베이스: 문서 종류별 수집·중복·인덱싱 건수, 발행 월 분포 — 구성과 편향
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from app import charts
from app import components as ui
from app import eval_data as ed
from app.experiments import (
    ADOPTED_ANSWER_SETTING,
    ANSWER_COMPARISONS,
    ANSWER_SETTINGS,
    CONTEXT_CHUNK_COMPARISON,
    RETRIEVAL_COMPARISONS,
    THRESHOLD_RUNS,
    AnswerComparison,
    AnswerSetting,
    RetrievalComparison,
)
from app.sources import DOC_TYPE_LABELS
from app.theme import set_page_width
from common.config import get_settings
from common.docstore import document_status_counts, kept_documents_by_month, open_db

PAGE_TITLE = "평가 결과"
PAGE_CAPTION = "평가셋 v2.0 · 답변 가능 32문항 (뉴스 문항 24 + 1차 자료 문항 8) · 답변 불가 6문항 · 답변은 직접 채점"
PAGE_MAX_WIDTH = 1440
VIEWS = ("요약", "검색 실험", "답변 품질", "거절 임계값", "지식베이스")
CACHE_TTL = 600  # 결과 파일은 실험을 돌릴 때만 바뀐다


def _pct(value: float | None, digits: int = 1) -> str:
    return "-" if value is None else f"{value:.{digits}%}"


def _setting(key: str) -> AnswerSetting:
    return next(s for s in ANSWER_SETTINGS if s.key == key)


def _comparison(key: str) -> RetrievalComparison:
    return next(c for c in RETRIEVAL_COMPARISONS if c.key == key)


def _answer_comparison(key: str) -> AnswerComparison:
    return next(c for c in ANSWER_COMPARISONS if c.key == key)


def _results_dir() -> Path:
    return get_settings().eval_results_dir


# ---- 데이터 (캐시: 인자는 문자열만) ----

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def answer_frames(results_dir: str, setting_key: str, question_set: str) -> list[pd.DataFrame]:
    return ed.setting_frames(Path(results_dir), _setting(setting_key), question_set)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def summary(results_dir: str, run_id: str) -> dict:
    return ed.load_summary(Path(results_dir), run_id)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def composition() -> tuple[list[tuple[str, str | None, int]], list[tuple[str, str | None, int]]]:
    with open_db(read_only=True) as conn:
        return document_status_counts(conn), kept_documents_by_month(conn)


# ---- 화면 조각 ----

def chart_card(key: str, title: str, caption: str, figure, *, alt: str, table: pd.DataFrame | None = None) -> None:
    """흰 카드 안의 차트 + 같은 값을 담은 표 (색·툴팁 없이도 값을 읽을 수 있게)."""
    with st.container(key=f"card-{key}"):
        st.html(ui.card_header_html(title, caption))
        st.plotly_chart(figure, theme=None, config=charts.PLOTLY_CONFIG, key=f"chart-{key}", alt=alt)
        if table is not None:
            with st.expander("표로 보기"):
                st.dataframe(table, hide_index=True)


def html_card(key: str, title: str, caption: str, body_html: str) -> None:
    with st.container(key=f"card-{key}"):
        st.html(ui.card_header_html(title, caption) + body_html)


def render_subnav() -> str:
    with st.container(key="subnav"):
        title_col, nav_col = st.columns([2, 3], vertical_alignment="center")
        with title_col:
            st.html(ui.subnav_title_html(PAGE_TITLE, PAGE_CAPTION))
        with nav_col, st.container(horizontal=True, horizontal_alignment="right"):
            view = st.segmented_control("보기", VIEWS, default=VIEWS[0], required=True, key="dashboard_view",
                                        label_visibility="collapsed")
    return view or VIEWS[0]


# ---- 요약 ----

def render_overview(results_dir: Path) -> None:
    root = str(results_dir)
    adopted = _setting(ADOPTED_ANSWER_SETTING)
    all_scores = ed.mean_scores(ed.setting_summary(answer_frames(root, adopted.key, "all")))
    news_frames = answer_frames(root, adopted.key, "news")
    news_scores = ed.mean_scores(ed.setting_summary(news_frames))
    refused_ids = sorted({row.id for frame in answer_frames(root, adopted.key, "all")
                          for row in frame.itertuples() if row.type != "unanswerable" and row.refused})
    latency = ed.answer_latency(results_dir, adopted)
    news_retrieval = summary(root, _comparison("primary").candidate.run_id)["overall"]
    primary_retrieval = summary(root, _comparison("primary8").candidate.run_id)["overall"]

    st.html(ui.hero_html(
        "답변 정답률",
        _pct(all_scores["accuracy"]),
        f"채택 설정(하이브리드 검색 + 1차 자료, 문서당 청크 2개, 프롬프트 v1) · 답변 가능 32문항 · 3회 평균 "
        f"(실행별 {_pct(all_scores['accuracy_min'])} ~ {_pct(all_scores['accuracy_max'])}) · "
        f"부분 정답 포함 {_pct(all_scores['accuracy_with_partial'])}",
    ))
    st.html(ui.stat_tiles_html([
        ui.StatTile("검색 Hit@3 · 뉴스 문항", f"{news_retrieval['hit@3']:.3f}",
                    f"평가셋 v1.3 · {news_retrieval['count']}문항 · MRR {news_retrieval['mrr']:.3f}"),
        ui.StatTile("검색 Hit@3 · 1차 자료 문항", f"{primary_retrieval['hit@3']:.3f}",
                    f"평가셋 v2.0 · {primary_retrieval['count']}문항 · MRR {primary_retrieval['mrr']:.3f}"),
        ui.StatTile("출처 정확도", _pct(all_scores["citation_accuracy"]), "인용한 근거 번호가 맞은 비율 · 3회 평균"),
        ui.StatTile("거절 정확도", _pct(news_scores["refusal_accuracy"]), "답변 불가 6문항 × 3회"),
        ui.StatTile("잘못된 거절", _pct(all_scores["false_refusal_rate"]), " · ".join(refused_ids) or "없음"),
        ui.StatTile("평균 응답 시간", f"{(latency['mean'] or 0) / 1000:.1f}초",
                    f"검색 + 답변 생성 · p95 {(latency['p95'] or 0) / 1000:.1f}초"),
    ]))
    html_card("decisions", "실험별 결정", "같은 평가셋 버전에서 한 가지만 바꿔 비교했다. 자세한 수치는 각 보기에서 확인",
              ui.table_html(["실험", "평가셋", "비교", "지표", "결정"], _decision_rows(results_dir)))
    st.html(ui.callout_html("측정의 한계", [
        "답변 가능 문항이 24 + 8개라 1문항이 약 3~4%p 에 해당한다.",
        "temperature 0 에서도 실행마다 등급이 바뀌는 문항이 있어(뉴스 문항 24개 중 5~6개), 답변 비교는 3회 평균과 문항별 일관성으로 판단한다.",
        "검색 지표는 정답 문서 목록 기준이라, 목록 밖 관련 문서는 판정(정답 보정)으로 추가해야 공정하게 비교된다.",
    ], tone="info"))


def _decision_rows(results_dir: Path) -> list[list[str]]:
    root = str(results_dir)
    rows = []
    for comparison in RETRIEVAL_COMPARISONS:
        base = summary(root, comparison.baseline.run_id)["overall"]
        cand = summary(root, comparison.candidate.run_id)["overall"]
        rows.append([
            f"검색 · {comparison.title}", comparison.eval_set,
            f"{comparison.baseline.label} → {comparison.candidate.label}",
            f"Hit@3 {base['hit@3']:.3f} → {cand['hit@3']:.3f}, MRR {base['mrr']:.3f} → {cand['mrr']:.3f}",
            comparison.decision,
        ])
    ctx = ed.context_comparison_table(results_dir, CONTEXT_CHUNK_COMPARISON).to_dict("records")
    rows.append([
        "답변 · 근거 청크 수", "뉴스 문항", f"{ctx[0]['설정']} → {ctx[1]['설정']}",
        f"정답률 {_pct(ctx[0]['accuracy'])} → {_pct(ctx[1]['accuracy'])}, "
        f"출처 정확도 {_pct(ctx[0]['citation_accuracy'])} → {_pct(ctx[1]['citation_accuracy'])}",
        "같은 기사의 세부 청크가 근거에 들어가 문서당 2개 채택",
    ])
    means = {s.key: ed.mean_scores(ed.setting_summary(answer_frames(root, s.key, "all"))) for s in ANSWER_SETTINGS}
    for comparison in ANSWER_COMPARISONS:
        base, cand = means[comparison.baseline], means[comparison.candidate]
        rows.append([
            f"답변 · {comparison.title}", "v2.0",
            f"{_setting(comparison.baseline).label} → {_setting(comparison.candidate).label} (3회 평균, 32문항)",
            f"정답률 {_pct(base['accuracy'])} → {_pct(cand['accuracy'])}, "
            f"부분 포함 {_pct(base['accuracy_with_partial'])} → {_pct(cand['accuracy_with_partial'])}",
            comparison.decision,
        ])
    return rows


# ---- 검색 실험 ----

def render_retrieval(results_dir: Path) -> None:
    root = str(results_dir)
    key = st.segmented_control("비교", [c.key for c in RETRIEVAL_COMPARISONS], default=RETRIEVAL_COMPARISONS[0].key,
                               required=True, format_func=lambda k: _comparison(k).title, key="retrieval_comparison")
    comparison = _comparison(key or RETRIEVAL_COMPARISONS[0].key)
    count = summary(root, comparison.candidate.run_id)["overall"]["count"]
    st.caption(f"평가셋 {comparison.eval_set} · 답변 가능 {count}문항 · k=5 · 결정: {comparison.decision}")

    metrics = ed.comparison_metrics(results_dir, comparison)
    by_type = ed.comparison_by_type(results_dir, comparison)
    left, right = st.columns(2, gap="medium")
    with left:
        chart_card(
            "retrieval-metrics", "검색 지표", f"{comparison.baseline.label}(회색) → {comparison.candidate.label}(파랑)",
            charts.comparison_bars(metrics), alt=f"{comparison.title} 비교: Hit@1, Hit@3, Hit@5, MRR 막대 그래프",
            table=metrics.pivot(index="metric", columns="setting", values="value").reset_index(),
        )
    with right:
        chart_card(
            "retrieval-types", "유형별 Hit@3", "문항 유형마다 상위 3건 안에 정답 문서가 있는 비율",
            charts.comparison_bars(by_type, value_format=".2f", category_column="type"),
            alt=f"{comparison.title} 비교: 질문 유형별 Hit@3 막대 그래프",
            table=by_type.pivot(index=["type", "count"], columns="setting", values="value").reset_index(),
        )
    latency = ed.latency_points(results_dir, comparison)
    chart_card(
        "retrieval-latency", "문항별 검색 시간", "점 = 문항 1개 (답변 불가 포함), 세로 눈금 = 평균. 답변 생성 시간은 빠져 있다",
        charts.latency_strip(latency), alt=f"{comparison.title} 비교: 문항별 검색 시간 분포",
        table=latency.pivot(index="id", columns="setting", values="latency_ms").reset_index(),
    )
    with st.container(key="card-retrieval-all"):
        st.html(ui.card_header_html("전체 검색 실험", "대시보드에 실은 실험의 지표 (평가셋 버전이 다르면 직접 비교하지 않는다)"))
        runs = []
        for c in RETRIEVAL_COMPARISONS:
            runs += [c.baseline, c.candidate]
        table = ed.retrieval_table(results_dir, runs)
        table.insert(1, "평가셋", [c.eval_set for c in RETRIEVAL_COMPARISONS for _ in (0, 1)])
        st.dataframe(table, hide_index=True, column_config={
            m: st.column_config.NumberColumn(format="%.3f") for m in ("Hit@1", "Hit@3", "Hit@5", "MRR")})


# ---- 답변 품질 ----

def _delta(candidate: float | None, baseline: float | None) -> str:
    if candidate is None or baseline is None:
        return ""
    return f"{(candidate - baseline) * 100:+.1f}%p"


def render_answers(results_dir: Path) -> None:
    root = str(results_dir)
    with st.container(horizontal=True, gap="medium"):
        comparison_key = st.segmented_control(
            "비교", [c.key for c in ANSWER_COMPARISONS], default=ANSWER_COMPARISONS[0].key, required=True,
            format_func=lambda k: _answer_comparison(k).title, key="answer_comparison")
        question_set = st.segmented_control("문항", list(ed.QUESTION_SETS), default="all", required=True,
                                            format_func=lambda k: ed.QUESTION_SETS[k], key="answer_question_set")
    comparison = _answer_comparison(comparison_key or ANSWER_COMPARISONS[0].key)
    question_set = question_set or "all"
    baseline, candidate = _setting(comparison.baseline), _setting(comparison.candidate)
    base_frames = answer_frames(root, baseline.key, question_set)
    cand_frames = answer_frames(root, candidate.key, question_set)
    base_table, cand_table = ed.setting_summary(base_frames), ed.setting_summary(cand_frames)
    base, cand = ed.mean_scores(base_table), ed.mean_scores(cand_table)
    n = int(cand_table["answerable"].iloc[0])
    st.caption(f"비교: {baseline.label} → {candidate.label} · 답변 가능 {n}문항 · 각 {len(cand_frames)}회 · "
               f"결정: {comparison.decision}")

    def tile(label: str, metric: str, note: str) -> ui.StatTile:
        delta = _delta(cand[metric], base[metric])
        return ui.StatTile(label, _pct(cand[metric]), f"기준 {_pct(base[metric])} · 차이 {delta} · {note}" if delta else note)

    tiles = [
        tile("정답률", "accuracy", f"실행별 {_pct(cand['accuracy_min'])} ~ {_pct(cand['accuracy_max'])}"),
        tile("부분 정답 포함", "accuracy_with_partial", "정확 + 부분"),
        tile("출처 정확도", "citation_accuracy", "답한 문항 중 근거 번호가 맞은 비율"),
        tile("잘못된 거절", "false_refusal_rate", "답변 가능한데 거절한 비율"),
    ]
    if cand["refusal_accuracy"] is not None:
        tiles.append(tile("거절 정확도", "refusal_accuracy", "답변 불가 문항을 거절한 비율"))
    st.html(ui.stat_tiles_html(tiles))

    summaries = {baseline.label: base_table, candidate.label: cand_table}
    shares = ed.grade_shares_by_type(cand_frames)
    left, right = st.columns(2, gap="medium")
    with left:
        chart_card(
            "answer-runs", "실행별 정답률", f"점 = 실행 1회, 세로 눈금 = 평균. {baseline.label}(회색) → {candidate.label}(파랑)",
            charts.run_dots(summaries, candidate.label), alt=f"{comparison.title}: 설정별 반복 실행 정답률 점 그래프",
            table=pd.concat([t.assign(설정=label) for label, t in summaries.items()], ignore_index=True)[
                ["설정", "실행", "correct", "partial", "wrong", "accuracy", "accuracy_with_partial", "citation_accuracy"]],
        )
    with right:
        chart_card(
            "answer-types", "유형별 등급", f"{candidate.label} · 문항 × {len(cand_frames)}회를 합친 비율. 오답이 많은 유형이 취약 유형",
            charts.grade_stack(shares), alt=f"{candidate.label}: 질문 유형별 정확·부분·오답 비율 누적 막대",
            table=shares.pivot(index=["type", "total"], columns="grade", values="count").reset_index(),
        )

    matrix = ed.grade_matrix({candidate.short: cand_frames, baseline.short: base_frames})
    grade_columns = [c for c in matrix.columns if c.endswith("회")]
    gap_column = "점수 차"
    matrix[gap_column] = matrix["score_gap"].map(lambda g: "-" if g != g else ("0" if abs(g) < 1e-9 else f"{g:+.2f}"))
    filters = {
        "differs": f"설정 간 점수가 다른 문항 {int(matrix['differs'].sum())}개",
        "varies": f"실행마다 달라진 문항 {int(matrix['varies'].sum())}개",
        "all": f"모든 문항 {len(matrix)}개",
    }
    with st.container(key="card-answer-grid"):
        st.html(ui.card_header_html(
            "문항별 등급",
            f"{candidate.short} = {candidate.label}, {baseline.short} = {baseline.label} (각 {len(cand_frames)}회). "
            f"점수 차 = {candidate.short} 평균 − {baseline.short} 평균 (정확 1 · 부분 0.5 · 오답 0)"))
        choice = st.segmented_control("문항 거르기", list(filters), default="differs", required=True,
                                      format_func=filters.get, key="answer_grid_filter", label_visibility="collapsed")
        rows = matrix if choice == "all" else matrix[matrix[choice or "differs"]]
        st.html(ui.grade_grid_html(rows.to_dict("records"), grade_columns, [gap_column]) if len(rows)
                else ui.callout_html("해당하는 문항이 없습니다", tone="info"))
        if choice != "all" and len(rows):
            st.caption(f"표시한 문항의 점수 차 합계 {rows['score_gap'].sum():+.2f}점 · 전체 {matrix['score_gap'].sum():+.2f}점")

    context = ed.context_comparison_table(results_dir, CONTEXT_CHUNK_COMPARISON)
    html_card("answer-context", "근거 청크 수 비교", "뉴스 지식베이스 · 뉴스 문항 24개 · 각 1회 (반복 측정 전 실험)",
              ui.table_html(
                  ["설정", "정확 / 부분 / 오답", "정답률", "부분 포함", "출처 정확도", "평균 응답 시간"],
                  [[r["설정"], f"{r['correct']} / {r['partial']} / {r['wrong']}", _pct(r["accuracy"]),
                    _pct(r["accuracy_with_partial"]), _pct(r["citation_accuracy"]),
                    f"{(r['평균 응답 시간(ms)'] or 0) / 1000:.1f}초"] for r in context.to_dict("records")],
                  numeric=(1, 2, 3, 4, 5)))


# ---- 거절 임계값 ----

def render_threshold(results_dir: Path) -> None:
    run_id = st.segmented_control("실험", [r.run_id for r in THRESHOLD_RUNS], default=THRESHOLD_RUNS[0].run_id,
                                  required=True, format_func=lambda r: next(x.label for x in THRESHOLD_RUNS if x.run_id == r),
                                  key="threshold_run")
    run_id = run_id or THRESHOLD_RUNS[0].run_id
    threshold = get_settings().rag_min_score
    points = ed.threshold_points(results_dir, run_id)
    unanswerable = points[points["group"] == "답변 불가"]
    answerable = points[points["group"] != "답변 불가"]
    below_unanswerable = int((unanswerable["top_score"] < threshold).sum())
    below_answerable = int((answerable["top_score"] < threshold).sum())

    chart_card(
        "threshold", "검색 1위 유사도 분포",
        f"점 = 문항 1개. 1위 유사도가 {threshold:.2f} 미만이면 LLM 을 부르지 않고 거절한다 (RAG_MIN_SCORE)",
        charts.threshold_strip(points, threshold), alt="검색 결과 구분별 1위 유사도 점 그래프와 거절 임계값",
        table=points.rename(columns={"q": "질문", "top_score": "1위 유사도", "gold_rank": "정답 순위", "group": "구분"}),
    )
    stats = points.groupby("group")["top_score"].agg(["count", "min", "median", "max"])
    stats = stats.reindex([g for g in ed.THRESHOLD_GROUPS if g in stats.index])
    below = points.assign(below=points["top_score"] < threshold).groupby("group")["below"].sum()
    html_card("threshold-stats", "구분별 요약", f"임계값 {threshold:.2f} 기준",
              ui.table_html(["구분", "문항", "최소", "중앙값", "최대", "임계값 미만"],
                            [[g, str(int(r["count"])), f"{r['min']:.3f}", f"{r['median']:.3f}", f"{r['max']:.3f}",
                              str(int(below[g]))] for g, r in stats.iterrows()],
                            numeric=(1, 2, 3, 4, 5)))
    st.html(ui.callout_html("해석", [
        f"답변 불가 {len(unanswerable)}문항 중 {below_unanswerable}개만 임계값 아래다. 답변 불가 질문도 주제가 비슷한 기사가 "
        "검색되어 유사도가 높게 나오므로, 유사도만으로는 거절을 가를 수 없다.",
        "나머지는 생성 단계의 프롬프트 규칙(근거가 없으면 '제공된 자료에서 확인할 수 없습니다')으로 거절한다. "
        "채택 설정의 거절 정확도는 3회 모두 100% 다.",
        f"답변 가능 문항 중 임계값 아래는 {below_answerable}개다. 임계값을 더 올리면 이런 문항이 검색 단계에서 잘못 거절된다.",
    ], tone="info"))


# ---- 지식베이스 ----

def render_knowledge_base() -> None:
    try:
        status_rows, month_rows = composition()
    except FileNotFoundError:
        st.html(ui.callout_html("원문 DB(store/docs.db)가 없습니다", [
            "수집·인덱싱(python -m collector …, python -m indexer prepare)을 실행한 PC 에서 확인할 수 있습니다.",
            "원문 DB 는 용량·재배포 문제로 저장소에 포함하지 않습니다.",
        ], tone="elective"))
        return
    status = ed.status_table(status_rows, DOC_TYPE_LABELS)
    monthly, undated = ed.monthly_table(month_rows, DOC_TYPE_LABELS)
    if status.empty:
        st.html(ui.callout_html("원문 DB 에 문서가 없습니다", tone="elective"))
        return

    kept_total = int(status["인덱싱 대상"].sum())
    st.html(ui.stat_tiles_html(
        [ui.StatTile("인덱싱 대상 문서", f"{kept_total:,}건", f"수집 {int(status['수집'].sum()):,}건에서 중복·품질 제외 후")]
        + [ui.StatTile(row["문서 종류"], f"{row['인덱싱 대상']:,}건", f"전체의 {row['인덱싱 대상'] / kept_total:.1%}")
           for row in status.to_dict("records") if row["인덱싱 대상"]]
    ))
    html_card("kb-status", "수집 → 인덱싱", "중복: 같은 기사의 재배포본 등 (대표 문서만 남김) · 품질 제외: 범위 밖·본문 부족 등",
              ui.table_html(list(status.columns),
                            [[str(v) if i == 0 else f"{v:,}" for i, v in enumerate(row)] for row in status.itertuples(index=False)],
                            numeric=(1, 2, 3, 4, 5)))
    doc_types = list(dict.fromkeys(monthly["doc_type"]))
    caption = "종류마다 세로 축 범위가 다르다 (뉴스가 훨씬 많아 한 축에 그리면 1차 자료가 보이지 않음)"
    if not undated.empty:
        caption += " · 발행일 미상: " + ", ".join(f"{r.doc_type} {r.count}건" for r in undated.itertuples())
    chart_card("kb-monthly", "발행 월 분포", caption, charts.monthly_multiples(monthly, doc_types),
               alt="문서 종류별 발행 월 건수 막대 그래프",
               table=monthly.assign(month=monthly["month"].dt.strftime("%Y-%m")))


VIEW_RENDERERS = {
    "요약": render_overview,
    "검색 실험": render_retrieval,
    "답변 품질": render_answers,
    "거절 임계값": render_threshold,
}


def render() -> None:
    set_page_width(PAGE_MAX_WIDTH)
    view = render_subnav()
    if view == "지식베이스":
        render_knowledge_base()
        return
    try:
        VIEW_RENDERERS[view](_results_dir())
    except ed.ResultsMissing as exc:
        st.html(ui.callout_html("평가 결과 파일을 찾을 수 없습니다", [
            str(exc), "app/experiments.py 의 실험 목록과 eval/results 폴더를 맞춰 주세요."], tone="reject"))
