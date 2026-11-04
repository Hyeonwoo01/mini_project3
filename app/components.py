"""화면 조각 HTML (st.html 로 넣는다). Streamlit 비의존, 단위 테스트 대상.

모든 동적 값(제목·본문·URL 등 수집 자료에서 온 문자열)은 html.escape 를 거친다.
스타일은 app/theme.py 의 cb-* 클래스로 준다 (인라인 스타일 없음).
"""

from collections.abc import Sequence
from html import escape
from typing import Literal

from app.sources import SourceView, find_highlight_spans
from common.schema import SourceTier

Tone = Literal["neutral", "approve", "elective", "reject", "info"]

# 출처 신뢰도 → 배지 색 (base.md 시맨틱 상태 토큰: 버튼이 아닌 읽기 전용 배지에만 쓴다)
TIER_TONES: dict[str, Tone] = {
    SourceTier.PRIMARY: "approve",
    SourceTier.PRESS: "neutral",
    SourceTier.USER_GENERATED: "elective",
}
QUOTE_PREVIEW_CHARS = 360


def badge(label: str, tone: Tone = "neutral") -> str:
    return f'<span class="cb-badge cb-badge--{tone}">{escape(label)}</span>'


def _meta_line(source: SourceView) -> str:
    items = [source.publisher, source.published or "발행일 미상", source.doc_type_label]
    return " · ".join(escape(item) for item in items if item)


def _score_badges(source: SourceView) -> str:
    """디버그용: 코사인 유사도와 (하이브리드면) 의미·키워드 검색 순위."""
    parts = [badge(f"유사도 {source.score:.3f}")]
    ranks = []
    if source.vector_rank is not None:
        ranks.append(f"의미 {source.vector_rank}위")
    if source.keyword_rank is not None:
        ranks.append(f"키워드 {source.keyword_rank}위")
    if ranks:
        parts.append(badge(" · ".join(ranks)))
    return "".join(parts)


def _preview(text: str, limit: int = QUOTE_PREVIEW_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def source_card_html(source: SourceView, *, show_scores: bool = False) -> str:
    """출처 카드 머리 (번호·제목·발행 정보·배지) + 근거 문단 미리보기."""
    badges = [badge(source.tier, TIER_TONES.get(source.tier, "neutral"))] if source.tier else []
    if show_scores:
        badges.append(_score_badges(source))
    quote = quote_html(_preview(source.passage)) if source.passage else ""
    return (
        '<div class="cb-source">'
        '<div class="cb-source__head">'
        f'<span class="cb-num">{source.number}</span>'
        f'<span class="cb-source__title">{escape(source.title)}</span>'
        "</div>"
        f'<div class="cb-source__meta"><span>{_meta_line(source)}</span>{"".join(badges)}</div>'
        f"{quote}"
        "</div>"
    )


def quote_html(text: str) -> str:
    """evidence-quote-block: 근거 문단 인용."""
    return f'<blockquote class="cb-quote">{escape(text)}</blockquote>'


def highlighted_document_html(text: str, passage: str) -> tuple[str, int]:
    """원문 전체에서 근거 문단을 <mark> 로 표시한 문서 시트 HTML 과 하이라이트 수."""
    spans = find_highlight_spans(text, passage)
    pieces, cursor = [], 0
    for start, end in spans:
        pieces.append(escape(text[cursor:start]))
        pieces.append(f'<mark class="cb-mark">{escape(text[start:end])}</mark>')
        cursor = end
    pieces.append(escape(text[cursor:]))
    return f'<div class="cb-doc">{"".join(pieces)}</div>', len(spans)


def document_meta_html(source: SourceView) -> str:
    badges = badge(source.tier, TIER_TONES.get(source.tier, "neutral")) if source.tier else ""
    return (
        f'<div class="cb-doc-title">{escape(source.title)}</div>'
        f'<div class="cb-source__meta"><span>{_meta_line(source)}</span>{badges}</div>'
    )


def callout_html(title: str, lines: Sequence[str] = (), tone: Tone = "elective") -> str:
    items = "".join(f"<li>{escape(line)}</li>" for line in lines)
    body = f"<ul>{items}</ul>" if items else ""
    return f'<div class="cb-callout cb-callout--{tone}"><strong>{escape(title)}</strong>{body}</div>'


def subnav_title_html(title: str, caption: str = "") -> str:
    caption_html = f'<span class="cb-subnav__caption">{escape(caption)}</span>' if caption else ""
    return f'<div class="cb-subnav__title">{escape(title)}{caption_html}</div>'


def question_html(question: str) -> str:
    return f'<div class="cb-question">{escape(question)}</div>'


def section_label_html(label: str) -> str:
    return f'<div class="cb-section-label">{escape(label)}</div>'


def empty_state_html(title: str, body: str) -> str:
    return f'<div class="cb-empty"><div class="cb-empty__title">{escape(title)}</div><p>{escape(body)}</p></div>'


def footer_html(lines: Sequence[str]) -> str:
    return '<div class="cb-footer">' + "".join(f"<p>{escape(line)}</p>" for line in lines) + "</div>"
