"""대화 UI.

답변 화면 (가이드 4-6)
- 답변 본문과 번호로 연결된 출처 카드 (제목, 발행처, 발행일, 출처 종류, 원문 링크)
- 원문 확인 창: 원문 전체에서 답변 근거 문단을 하이라이트 (common.docstore.get_document, 읽기 전용)
- 검색된 청크의 유사도 점수·검색 순위 (필터 메뉴의 '유사도 점수 표시', 디버깅용)

UI 요건 (가이드 4-7)
- 스트리밍 출력, 검색 진행 상태 표시 (st.status)
- st.session_state 로 멀티턴 대화 유지 (체인에는 최근 대화만 넘어간다: rag.chain.MAX_HISTORY_TURNS)
- 기간·출처 종류 필터 (서브 내비게이션의 필터 메뉴 → rag.retriever.SearchFilters)
- 근거가 없거나 오류가 나면 원인과 다음 행동을 안내한다
"""

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import streamlit as st

from app import components as ui
from app.sources import (
    DOC_TYPE_LABELS,
    FILTERABLE_DOC_TYPES,
    SourceView,
    build_filters,
    describe_filters,
    format_answer_markdown,
    split_sources,
    tier_summary,
    to_source_views,
)
from app.theme import set_page_width
from common.docstore import get_document, open_db
from rag.chain import RagChain, cited_numbers, get_chain, is_refusal, retrieval_query
from rag.retriever import SearchFilters

logger = logging.getLogger(__name__)

PAGE_TITLE = "정책 근거 질의응답"
PAGE_CAPTION = "뉴스 · 정부 보도자료 · 서울시 공고에서 근거를 찾아 출처 번호와 함께 답합니다"
PAGE_MAX_WIDTH = 980
INPUT_PLACEHOLDER = "정책·사업명, 기관, 시기를 넣어 질문하세요"
ANSWER_CURSOR = " ▍"

# README '대표 질문' (평가셋 문항, 채택 설정 3회 모두 정확)
EXAMPLE_QUESTIONS = (
    "2026년 상반기 우대수수료율이 적용되는 영세·중소 신용카드가맹점은 몇 개이고 전체의 몇 %인가?",
    "한국신용데이터 분석에서 민생회복 소비쿠폰 지급 첫 주 소상공인 카드 매출은 어떻게 변했고, 가장 크게 늘어난 업종은?",
    "영세·중소가맹점 신용카드 우대수수료율은 2025년에 어떻게 바뀌었나?",
    "중기부와 서울시의 자영업자 고용보험료 지원은 환급 비율과 기간이 어떻게 다른가?",
    "민생회복 소비쿠폰의 소비 진작 효과에 대해 어떤 긍정적 평가와 비판이 있나?",
)
FOOTER_LINES = (
    "답변은 수집한 뉴스·정부 보도자료·서울시 공고에 근거한 분석 참고용 요약입니다. 수치와 기준은 원문을 확인한 뒤 인용하세요.",
    "자료에 없는 내용(특정 상권·점포의 카드 매출 등)은 답하지 않습니다. 출처별 이용 조건은 각 원문 페이지를 따릅니다.",
)

# 세션 상태 키
TURNS = "turns"
PENDING_QUESTION = "pending_question"
FILTER_TYPES = "filter_doc_types"
FILTER_DATES = "filter_dates"
SHOW_SCORES = "show_scores"


@dataclass
class Turn:
    """질문 하나와 그 답변. 다시 그릴 때 검색·생성을 반복하지 않도록 결과를 그대로 보관한다."""

    question: str
    answer: str = ""
    sources: list[SourceView] = field(default_factory=list)
    refused: bool = False
    retrieval_refused: bool = False
    filters_label: str = "전체 자료"
    details: list[str] = field(default_factory=list)   # 검색 상태 펼침 영역에 보이는 검색 조건
    elapsed_s: float = 0.0
    error: str | None = None

    @property
    def status_label(self) -> str:
        if self.error:
            return "오류가 발생했습니다"
        if self.retrieval_refused:
            return "기준을 넘는 근거 문서 없음"
        parts = [f"근거 {len(self.sources)}건 확인", tier_summary(self.sources), f"{self.elapsed_s:.1f}초"]
        return " · ".join(part for part in parts if part)


def chat_history(turns: Sequence[Turn]) -> list[tuple[str, str]]:
    """체인에 넘길 대화 이력. 오류로 끝난 턴은 뺀다."""
    history: list[tuple[str, str]] = []
    for turn in turns:
        if turn.error is None:
            history += [("user", turn.question), ("assistant", turn.answer)]
    return history


def _turns() -> list[Turn]:
    return st.session_state.setdefault(TURNS, [])


def _reset_conversation() -> None:
    st.session_state[TURNS] = []


def _ask_example(question: str) -> None:
    st.session_state[PENDING_QUESTION] = question


def _load_chain() -> tuple[RagChain | None, str | None]:
    """설정 오류(API 키·컬렉션 없음, 임베딩 모델 불일치)는 화면에 원인을 보여 준다."""
    try:
        return get_chain(), None
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        return None, str(exc)


@st.cache_data(show_spinner=False, max_entries=64, ttl="1h")
def load_document_text(doc_id: str) -> str | None:
    """원문 확인 창용 본문. 인덱서 정제 본문이 있으면 그것을 (청크가 이 본문에서 잘렸다), 없으면 수집 원문."""
    with open_db(read_only=True) as conn:
        record = get_document(conn, doc_id)
    if record is None:
        return None
    return record.get("clean_text") or record.get("raw_text") or ""


# ---- 화면 조각 ----

def render_subnav(has_turns: bool) -> tuple[SearchFilters | None, bool]:
    """제목과 필터 메뉴·새 대화 버튼. (검색 필터, 유사도 점수 표시 여부)를 돌려준다."""
    # 위젯 값은 실행이 시작될 때 이미 세션 상태에 반영돼 있어, 메뉴 제목에 현재 범위를 바로 쓸 수 있다
    filters = build_filters(st.session_state.get(FILTER_TYPES), st.session_state.get(FILTER_DATES))
    with st.container(key="subnav"):
        title_col, action_col = st.columns([3, 2], vertical_alignment="center")
        with title_col:
            st.html(ui.subnav_title_html(PAGE_TITLE, PAGE_CAPTION))
        with action_col, st.container(horizontal=True, horizontal_alignment="right", gap="small"):
            with st.popover(f"검색 범위: {describe_filters(filters)}"):
                st.pills(
                    "출처 종류", FILTERABLE_DOC_TYPES, selection_mode="multi", key=FILTER_TYPES,
                    format_func=lambda t: DOC_TYPE_LABELS[t], help="선택하지 않으면 모든 종류에서 찾습니다",
                )
                st.date_input(
                    "발행일", value=[], key=FILTER_DATES, format="YYYY.MM.DD",
                    help="기간을 지정하면 발행일을 알 수 없는 문서는 검색에서 빠집니다",
                )
                st.toggle("유사도 점수 표시 (디버깅)", key=SHOW_SCORES)
            st.button("새 대화", on_click=_reset_conversation, disabled=not has_turns, key="reset")
    return filters, bool(st.session_state.get(SHOW_SCORES))


def render_empty_state() -> None:
    st.html(ui.empty_state_html(
        "무엇을 확인할까요?",
        "상권 지표 변화의 배경이 되는 소상공인 정책과 상권 이슈를 찾아 드립니다. "
        "답변에 붙은 번호는 아래 출처 카드와 연결되고, 원문에서 근거 문단을 바로 확인할 수 있습니다.",
    ))
    st.html(ui.section_label_html("예시 질문"))
    with st.container(key="examples", horizontal=True, gap="small"):
        for i, question in enumerate(EXAMPLE_QUESTIONS):
            st.button(question, key=f"example-{i}", on_click=_ask_example, args=(question,))


@st.dialog("원문 확인", width="large")
def show_document(source: SourceView) -> None:
    st.html(ui.document_meta_html(source))
    try:
        text = load_document_text(source.doc_id)
    except FileNotFoundError as exc:
        st.html(ui.callout_html("원문 DB 를 열 수 없습니다", [str(exc)], tone="reject"))
        text = None
    if text:
        sheet, highlighted = ui.highlighted_document_html(text, source.passage)
        st.caption(
            "노란색 부분이 답변 생성에 쓰인 근거 문단입니다." if highlighted
            else "근거 문단의 위치를 원문에서 찾지 못했습니다. 아래 카드의 근거 문단과 대조해 주세요."
        )
        st.html(sheet)
    else:
        st.html(ui.callout_html("저장된 원문 텍스트가 없습니다", ["답변 생성에 쓰인 근거 문단만 표시합니다."], tone="elective"))
        st.html(ui.quote_html(source.passage))
    if source.url:
        st.link_button("원문 페이지 열기 ↗", source.url)


def render_source_card(turn_index: int, source: SourceView, show_scores: bool) -> None:
    with st.container(key=f"src-{turn_index}-{source.number}"):
        st.html(ui.source_card_html(source, show_scores=show_scores))
        with st.container(horizontal=True, gap="small", vertical_alignment="center"):
            if st.button("원문에서 근거 확인", key=f"open-{turn_index}-{source.number}"):
                show_document(source)
            if source.url:
                st.link_button("원문 페이지 ↗", source.url)


def render_sources(turn_index: int, turn: Turn, show_scores: bool) -> None:
    shown, folded = split_sources(turn.sources, turn.refused)
    if shown:
        st.html(ui.section_label_html(f"출처 {len(shown)}건"))
        for source in shown:
            render_source_card(turn_index, source, show_scores)
    if folded:
        label = (f"검색됐지만 답변에 쓰이지 않은 문서 {len(folded)}건" if turn.refused
                 else f"답변에 인용되지 않은 근거 {len(folded)}건")
        with st.expander(label):
            for source in folded:
                render_source_card(turn_index, source, show_scores)


def render_answer_body(turn: Turn) -> None:
    if turn.error:
        st.html(ui.callout_html("답변을 만들지 못했습니다", [turn.error, "잠시 후 다시 시도해 주세요."], tone="reject"))
        return
    st.markdown(format_answer_markdown(turn.answer, len(turn.sources)))
    if turn.retrieval_refused:
        tips = ["정책·사업명, 기관명, 연도처럼 구체적인 표현을 넣어 다시 질문해 보세요."]
        if turn.filters_label != describe_filters(None):
            tips.append(f"검색 범위를 넓혀 보세요 (현재: {turn.filters_label}).")
        tips.append("특정 상권·점포의 매출처럼 수집 자료에 없는 수치는 답하지 않습니다.")
        st.html(ui.callout_html("기준을 넘는 근거 문서를 찾지 못했습니다", tips))
    elif turn.refused:
        st.html(ui.callout_html("검색된 문서에 질문에 답할 내용이 없었습니다", [
            "아래 '검색됐지만 답변에 쓰이지 않은 문서'에서 무엇이 검색됐는지 확인할 수 있습니다.",
            "질문을 더 구체적으로 바꾸거나 검색 범위를 넓혀 보세요.",
        ]))


def render_turn(turn_index: int, turn: Turn, show_scores: bool) -> None:
    st.html(ui.question_html(turn.question))
    with st.container(key=f"answer-{turn_index}"):
        with st.status(turn.status_label, state="error" if turn.error else "complete", expanded=False):
            for line in turn.details:
                st.caption(line)
        render_answer_body(turn)
    render_sources(turn_index, turn, show_scores)


# ---- 질문 처리 ----

def _search_details(chain: RagChain, question: str, history: Sequence[tuple[str, str]], filters_label: str) -> list[str]:
    query = retrieval_query(question, history)
    lines = [f"검색 질의: {query}" + (" (직전 질문을 덧붙임)" if query != question else "")]
    lines.append(f"검색 범위: {filters_label}")
    lines.append(f"근거 기준: 유사도 {chain.config.min_score:.2f} 이상, 상위 {chain.config.top_k}건")
    lines.append(f"검색 방식: {chain.retriever.mode} · 컬렉션 {chain.retriever.collection}")
    return lines


def answer_question(chain: RagChain, question: str, filters: SearchFilters | None, turns: list[Turn]) -> Turn:
    """검색 → 스트리밍 생성을 화면에 그리면서 진행하고, 결과를 Turn 으로 남긴다."""
    turn_index = len(turns)
    history = chat_history(turns)
    turn = Turn(question=question, filters_label=describe_filters(filters))
    turn.details = _search_details(chain, question, history, turn.filters_label)
    started = time.perf_counter()

    st.html(ui.question_html(question))
    with st.container(key=f"answer-{turn_index}"):
        with st.status("근거 문서를 찾는 중…", expanded=False) as status:
            for line in turn.details:
                st.caption(line)
        answer_slot = st.empty()
        try:
            chunks = chain.retrieve_context(question, history, filters=filters)
            turn.retrieval_refused = not chunks
            status.update(label="답변을 작성하는 중…" if chunks else "기준을 넘는 근거 문서 없음")
            parts: list[str] = []
            for piece in chain.stream_answer(question, chunks, history):
                parts.append(piece)
                answer_slot.markdown(format_answer_markdown("".join(parts), len(chunks)) + ANSWER_CURSOR)
            turn.answer = "".join(parts).strip()
            turn.refused = not chunks or is_refusal(turn.answer)
            turn.sources = to_source_views(chunks, cited_numbers(turn.answer, len(chunks)))
        except Exception as exc:  # UI 경계: 검색·생성 실패를 대화 안에 보여 주고 다음 질문을 받는다
            logger.exception("질문 처리 실패: %s", question)
            turn.error = f"{type(exc).__name__}: {exc}"
        turn.elapsed_s = time.perf_counter() - started
        status.update(label=turn.status_label, state="error" if turn.error else "complete")
        answer_slot.empty()
        render_answer_body(turn)
    turns.append(turn)
    render_sources(turn_index, turn, bool(st.session_state.get(SHOW_SCORES)))
    return turn


def render() -> None:
    set_page_width(PAGE_MAX_WIDTH)
    turns = _turns()
    chain, chain_error = _load_chain()
    # 입력창은 호출 위치와 관계없이 화면 아래에 고정된다. 서브 내비게이션보다 먼저 읽어야
    # 첫 질문을 처리하는 실행에서도 '새 대화' 버튼이 활성화된다.
    question = st.chat_input(INPUT_PLACEHOLDER, disabled=chain is None) or st.session_state.pop(PENDING_QUESTION, None)
    question = question.strip() if question and chain is not None else None

    filters, show_scores = render_subnav(has_turns=bool(turns or question))
    if chain_error:
        st.html(ui.callout_html("챗봇을 시작할 수 없습니다", [chain_error], tone="reject"))
    if not turns and not question:
        render_empty_state()
    for index, turn in enumerate(turns):
        render_turn(index, turn, show_scores)
    if question and chain is not None:
        answer_question(chain, question, filters, turns)
    st.html(ui.footer_html(FOOTER_LINES))
