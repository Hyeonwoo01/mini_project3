"""대시보드 차트 (Plotly). 입력은 app.eval_data 의 표, 출력은 go.Figure (Streamlit 비의존).

색은 디자인 시스템 안에서만 쓴다 (base.md: 차트도 Action Blue · Ink · 회색 · 시맨틱 색).
- 비교(기준 → 개선안): 강조 형식. 개선안 Action Blue, 기준 회색. 회색 막대는 대비가 낮아(1.5:1) 값을 막대 끝에 적는다
- 채점 등급: 정확/부분/오답 = approve/elective/reject (등급 자체가 좋고 나쁨을 뜻하는 상태 값)
- 그 밖에는 위치로 구분하고 한 가지 색만 쓴다
막대는 가로로 그려 행 높이로 두께를 정한다 (24px 이하). 끝은 4px 둥글게, 맞닿는 칸 사이는 2px 흰 틈.
"""

from collections.abc import Sequence
from html import escape

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.eval_data import GRADES, THRESHOLD_GROUPS, TYPE_LABELS
from app.theme import COLORS

FONT = '"Pretendard Variable", Pretendard, -apple-system, BlinkMacSystemFont, system-ui, sans-serif'
INK = COLORS["ink"]
MUTED = COLORS["ink_muted_48"]
MUTED_80 = COLORS["ink_muted_80"]
GRID = COLORS["divider_soft"]
HAIRLINE = COLORS["hairline"]
SURFACE = COLORS["canvas"]
ROLE_COLORS = {"baseline": COLORS["chip_gray"], "candidate": COLORS["primary"]}
GRADE_COLORS = {"정확": COLORS["approve"], "부분": COLORS["elective"], "오답": COLORS["reject"]}
ROW_PX = 44           # 가로 막대 한 칸 높이: 막대 2개가 들어가도 각각 24px 이하
LEGEND_PX = 40
AXIS_PX = 36
PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}


def _base_layout(fig: go.Figure, *, height: int, legend: bool) -> go.Figure:
    fig.update_layout(
        template="none",   # Plotly 기본 테마(회색 배경·색 순환)를 쓰지 않는다
        height=height,
        margin={"l": 8, "r": 16, "t": LEGEND_PX if legend else 12, "b": 8},
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font={"family": FONT, "size": 13, "color": INK},
        showlegend=legend,
        legend={"orientation": "h", "x": 0, "xanchor": "left", "y": 1.0, "yanchor": "bottom",
                "font": {"size": 13, "color": MUTED_80}, "title": {"text": ""}, "traceorder": "normal",
                "itemclick": False, "itemdoubleclick": False},
        hoverlabel={"bgcolor": SURFACE, "bordercolor": HAIRLINE, "font": {"family": FONT, "size": 13, "color": INK},
                    "align": "left"},
        barcornerradius=4,
    )
    # template="none" 은 축 여백 자동 계산을 끄므로 automargin 을 켠다 (축 글자가 잘리지 않게)
    fig.update_xaxes(gridcolor=GRID, gridwidth=1, linecolor=HAIRLINE, zeroline=False, ticks="", automargin=True,
                     tickfont={"size": 12, "color": MUTED}, title={"font": {"size": 12, "color": MUTED}})
    fig.update_yaxes(gridcolor=GRID, gridwidth=1, linecolor=HAIRLINE, zeroline=False, ticks="", automargin=True,
                     tickfont={"size": 13, "color": MUTED_80}, showgrid=False)
    return fig


def _bar_height(n_rows: int, *, legend: bool = True, axis: bool = True) -> int:
    return n_rows * ROW_PX + (LEGEND_PX if legend else 12) + (AXIS_PX if axis else 8)


def comparison_bars(frame: pd.DataFrame, *, value_format: str = ".3f", tick_format: str = ".1f",
                    x_range: tuple[float, float] = (0, 1.12), category_column: str = "metric") -> go.Figure:
    """기준 vs 개선안 가로 막대 (행: 지표 또는 유형). frame 열: setting, role, value, <category_column>."""
    categories = list(dict.fromkeys(frame[category_column]))
    fig = go.Figure()
    for role in ("baseline", "candidate"):
        subset = frame[frame["role"] == role]
        if subset.empty:
            continue
        setting = subset["setting"].iloc[0]
        fig.add_bar(
            y=subset[category_column], x=subset["value"], orientation="h", name=setting,
            marker={"color": ROLE_COLORS[role], "line": {"color": SURFACE, "width": 2}},
            text=[format(v, value_format) for v in subset["value"]], textposition="outside", cliponaxis=False,
            textfont={"size": 12, "color": INK if role == "candidate" else MUTED_80},
            hovertemplate=f"<b>%{{x:{value_format}}}</b><br>%{{y}} · {setting}<extra></extra>",
        )
    fig.update_layout(barmode="group", bargap=0.38, bargroupgap=0.06)
    fig.update_yaxes(categoryorder="array", categoryarray=categories, autorange="reversed")
    fig.update_xaxes(range=list(x_range), showgrid=True, tickformat=tick_format)
    return _base_layout(fig, height=_bar_height(len(categories)), legend=True)


def grade_stack(frame: pd.DataFrame) -> go.Figure:
    """유형별 등급 비율 누적 가로 막대. frame 열: type, grade, count, total, share."""
    types = list(dict.fromkeys(frame["type"]))
    fig = go.Figure()
    for grade in GRADES:
        subset = frame[frame["grade"] == grade]
        fig.add_bar(
            y=subset["type"], x=subset["share"], orientation="h", name=grade,
            marker={"color": GRADE_COLORS[grade], "line": {"color": SURFACE, "width": 2}},
            text=[f"{s:.0%}" if s > 0 else "" for s in subset["share"]], textposition="inside",
            insidetextanchor="middle", textangle=0, textfont={"size": 12, "color": "#ffffff"},
            customdata=subset[["count", "total"]].to_numpy(),
            hovertemplate=f"<b>%{{x:.0%}}</b> (%{{customdata[0]}}/%{{customdata[1]}})<br>%{{y}} · {grade}<extra></extra>",
        )
    # 칸 안에 들어가지 않는 값은 숨기고 툴팁·표로 본다 (잘린 글자를 보이지 않게)
    fig.update_layout(barmode="stack", bargap=0.45, uniformtext={"minsize": 11, "mode": "hide"})
    fig.update_yaxes(categoryorder="array", categoryarray=types, autorange="reversed")
    fig.update_xaxes(range=[0, 1], tickformat=".0%", showgrid=True)
    return _base_layout(fig, height=_bar_height(len(types)), legend=True)


def run_dots(summaries: dict[str, pd.DataFrame], adopted_label: str, metric: str = "accuracy") -> go.Figure:
    """설정별 반복 실행 정답률 (점 = 실행 1회, 세로 눈금 = 평균). 고른 설정만 Action Blue, 평균은 행 이름에 적는다.
    같은 값의 실행이 겹쳐 하나로 보이지 않도록 실행마다 세로 위치를 조금씩 다르게 둔다."""
    labels = list(summaries)
    fig = go.Figure()
    for row, label in enumerate(labels):
        summary = summaries[label].reset_index(drop=True)
        color = ROLE_COLORS["candidate"] if label == adopted_label else MUTED
        offsets = [(i - (len(summary) - 1) / 2) * 0.16 for i in range(len(summary))]
        fig.add_scatter(
            x=summary[metric], y=[row + o for o in offsets], mode="markers", name=label, showlegend=False,
            marker={"size": 12, "color": color, "line": {"color": SURFACE, "width": 2}},
            customdata=summary[["실행", "correct", "answerable"]].to_numpy(),
            hovertemplate=f"<b>%{{x:.1%}}</b> (정확 %{{customdata[1]}}/%{{customdata[2]}})<br>{label} · %{{customdata[0]}}"
                          "<extra></extra>",
        )
        fig.add_scatter(
            x=[float(summary[metric].mean())], y=[row], mode="markers", showlegend=False,
            marker={"symbol": "line-ns", "size": 30, "line": {"color": INK, "width": 2}},
            hovertemplate=f"<b>평균 %{{x:.1%}}</b><br>{label}<extra></extra>",
        )
    fig.update_yaxes(tickmode="array", tickvals=list(range(len(labels))),
                     ticktext=[f"{label}<br>평균 {summaries[label][metric].mean():.1%}" for label in labels],
                     range=[len(labels) - 0.5, -0.5])
    fig.update_xaxes(range=[0, 1], tickformat=".0%", showgrid=True)
    return _base_layout(fig, height=len(labels) * 72 + 12 + AXIS_PX, legend=False)


def _jitter(index: int) -> float:
    """겹치는 점을 세로로 흩뜨린다. 결정적 (다시 그려도 같은 자리)."""
    return ((index * 7) % 9 - 4) * 0.055


def threshold_strip(points: pd.DataFrame, threshold: float) -> go.Figure:
    """검색 1위 유사도 분포 (행: 검색 결과 구분, 점: 문항) + 거절 임계값 세로선."""
    groups = [g for g in THRESHOLD_GROUPS if g in set(points["group"])]
    position = {g: i for i, g in enumerate(groups)}
    ys = [position[g] + _jitter(i) for i, g in enumerate(points["group"])]
    fig = go.Figure()
    fig.add_scatter(
        x=points["top_score"], y=ys, mode="markers", showlegend=False,
        marker={"size": 11, "color": COLORS["primary"], "line": {"color": SURFACE, "width": 2}},
        customdata=list(zip(points["id"], points["type"].map(TYPE_LABELS).fillna(points["type"]),
                            points["group"], points["q"].map(_short))),
        hovertemplate="<b>%{x:.3f}</b><br>%{customdata[0]} · %{customdata[1]} · %{customdata[2]}"
                      "<br>%{customdata[3]}<extra></extra>",
    )
    fig.add_vline(x=threshold, line={"color": INK, "width": 1, "dash": "dash"})
    fig.add_annotation(x=threshold, y=1.0, yref="paper", yanchor="bottom", showarrow=False, xanchor="left",
                       text=f" 거절 임계값 {threshold:.2f}", font={"size": 12, "color": INK})
    low = min(float(points["top_score"].min()), threshold) - 0.03
    high = float(points["top_score"].max()) + 0.03
    fig.update_xaxes(range=[low, high], tickformat=".2f", showgrid=True, title={"text": "검색 1위 유사도 (코사인)"})
    fig.update_yaxes(tickmode="array", tickvals=list(range(len(groups))), ticktext=groups,
                     range=[len(groups) - 0.5, -0.5])
    fig = _base_layout(fig, height=len(groups) * 64 + 28 + AXIS_PX + 12, legend=False)
    fig.update_layout(margin={"t": 28})
    return fig


def latency_strip(points: pd.DataFrame) -> go.Figure:
    """문항별 검색 시간 (행: 설정). 기준 회색, 개선안 Action Blue, 세로 눈금 = 평균 (행 이름에도 적음)."""
    settings = list(dict.fromkeys(points["setting"]))
    means = {s: float(points.loc[points["setting"] == s, "latency_ms"].mean()) for s in settings}
    position = {s: i for i, s in enumerate(settings)}
    fig = go.Figure()
    for setting in settings:
        subset = points[points["setting"] == setting].reset_index(drop=True)
        role = subset["role"].iloc[0]
        color = ROLE_COLORS["candidate"] if role == "candidate" else MUTED
        fig.add_scatter(
            x=subset["latency_ms"], y=[position[setting] + _jitter(i) for i in range(len(subset))], mode="markers",
            showlegend=False, marker={"size": 9, "color": color, "line": {"color": SURFACE, "width": 2}},
            customdata=subset[["id"]].to_numpy(),
            hovertemplate=f"<b>%{{x:,.0f}}ms</b><br>%{{customdata[0]}} · {setting}<extra></extra>",
        )
        fig.add_scatter(
            x=[means[setting]], y=[position[setting]], mode="markers", showlegend=False,
            marker={"symbol": "line-ns", "size": 28, "line": {"color": INK, "width": 2}},
            hovertemplate=f"<b>평균 %{{x:,.0f}}ms</b><br>{setting}<extra></extra>",
        )
    fig.update_xaxes(showgrid=True, tickformat=",.0f", ticksuffix="ms", rangemode="tozero")
    fig.update_yaxes(tickmode="array", tickvals=list(range(len(settings))),
                     ticktext=[f"{s}<br>평균 {means[s]:,.0f}ms" for s in settings], range=[len(settings) - 0.5, -0.5])
    return _base_layout(fig, height=len(settings) * 72 + 12 + AXIS_PX, legend=False)


def monthly_multiples(monthly: pd.DataFrame, doc_types: Sequence[str]) -> go.Figure:
    """문서 종류별 발행 월 분포 (작은 배수: 종류마다 행, 세로 축은 종류마다 따로)."""
    doc_types = [t for t in doc_types if t in set(monthly["doc_type"])]
    rows = max(len(doc_types), 1)
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.12 if rows > 1 else 0.0,
                        subplot_titles=[f"{t} (월별 건수)" for t in doc_types])
    for row, doc_type in enumerate(doc_types, start=1):
        subset = monthly[monthly["doc_type"] == doc_type]
        fig.add_bar(
            x=subset["month"], y=subset["count"], name=doc_type, showlegend=False, row=row, col=1,
            marker={"color": COLORS["primary"], "line": {"color": SURFACE, "width": 2}},
            hovertemplate=f"<b>%{{y:,}}건</b><br>%{{x|%Y-%m}} · {doc_type}<extra></extra>",
        )
        fig.update_yaxes(showgrid=True, gridcolor=GRID, tickformat=",d", rangemode="tozero", row=row, col=1)
    fig.update_xaxes(dtick="M3", tickformat="%Y-%m")
    fig.update_layout(bargap=0.55)   # 월 20여 개를 카드 폭에 그릴 때 막대가 24px 안팎
    fig = _base_layout(fig, height=rows * 150 + 60, legend=False)
    fig.update_layout(margin={"t": 32})
    fig.update_yaxes(showgrid=True)
    fig.update_annotations(font={"size": 13, "color": MUTED_80}, xanchor="left", x=0)
    return fig


def _short(text: str, limit: int = 48) -> str:
    """툴팁용 질문 요약. Plotly 툴팁은 일부 HTML 태그를 해석하므로 escape 한다."""
    text = str(text)
    return escape(text if len(text) <= limit else text[:limit] + "…")
