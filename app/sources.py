"""챗봇 화면의 순수 로직 (Streamlit 비의존, 단위 테스트 대상).

- 검색 청크 → 출처 카드 표시용 SourceView (답변의 [n] 은 chunks[n-1])
- 답변 마크다운 정리: [n] 을 번호 배지로 바꾸고, 한국어 범위 표기(9~11월)가 취소선으로 렌더링되지 않게 한다
- 사이드 필터 값 → rag.retriever.SearchFilters
- 원문 확인 창: 근거 문단이 원문 어디에 있는지 찾아 하이라이트 구간을 만든다
"""

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date

from common.schema import DocType, SourceTier
from rag.retriever import RetrievedChunk, SearchFilters

DOC_TYPE_LABELS: dict[str, str] = {
    DocType.NEWS: "뉴스",
    DocType.PRESS_RELEASE: "보도자료",
    DocType.ANNOUNCEMENT: "서울시 공고",
    DocType.POLICY_REPORT: "연구 보고서",
    DocType.LAW: "법령",
    DocType.BLOG: "블로그",
}
# 필터에 내놓는 문서 종류 = 현재 지식베이스에 있는 종류 (README '수집 현황')
FILTERABLE_DOC_TYPES: tuple[DocType, ...] = (DocType.NEWS, DocType.PRESS_RELEASE, DocType.ANNOUNCEMENT,
                                              DocType.POLICY_REPORT, DocType.LAW)

# 근거 확장(rag.retriever.merge_chunk_texts)이 떨어진 청크 사이에 넣는 구분자
PASSAGE_GAP = "\n…\n"
_CITATION = re.compile(r"\[(\d{1,2})\]")
# Streamlit 마크다운에서 ~ 는 취소선, $ 는 수식으로 해석된다 (예: '9~11월 … 3~5%' 사이가 취소선이 됨)
_MARKDOWN_SPECIAL = re.compile(r"(?<!\\)([~$])")
# 굵게(**…**): CommonMark 는 닫는 ** 앞이 문장부호이고 뒤가 글자이면 강조로 보지 않는다.
# 한국어는 조사가 바로 붙어서('**95.8%**입니다') 별표가 그대로 보이므로, 바깥쪽 문장부호를 강조 밖으로 옮긴다.
_EMPHASIS = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*")
_MOVABLE_LEADING = "\"'“‘(「『"
_MOVABLE_TRAILING = "%.,!?·:;\"'”’)」』"
MIN_PIECE_CHARS = 20  # 이보다 짧은 근거 조각은 원문 곳곳에 우연히 일치할 수 있어 하이라이트하지 않는다


@dataclass(frozen=True)
class SourceView:
    """출처 카드 한 장. 세션 상태에 그대로 저장한다 (검색 결과 객체를 화면에 묶어 두지 않음)."""

    number: int               # 답변의 [n]
    doc_id: str
    title: str
    publisher: str
    published: str            # YYYY-MM-DD, 미상이면 ""
    doc_type: str
    tier: str                 # common.schema.SourceTier 값
    url: str
    passage: str              # 답변 생성에 들어간 근거 문단 (제목 머리말 제외)
    score: float              # 코사인 유사도
    vector_rank: int | None
    keyword_rank: int | None
    cited: bool               # 답변에 [n] 으로 실제 인용됐는지

    @property
    def doc_type_label(self) -> str:
        return DOC_TYPE_LABELS.get(self.doc_type, self.doc_type)


def strip_title_prefix(text: str, title: str) -> str:
    """청크 본문 앞에 붙인 '제목\\n\\n' 머리말을 뗀다 (원문에는 없는 부분)."""
    prefix = f"{title}\n\n"
    return text[len(prefix):] if title and text.startswith(prefix) else text


def to_source_views(chunks: Sequence[RetrievedChunk], cited: Iterable[int]) -> list[SourceView]:
    cited_set = set(cited)
    views = []
    for number, chunk in enumerate(chunks, start=1):
        meta = chunk.metadata
        title = str(meta.get("title", "")).strip()
        views.append(SourceView(
            number=number,
            doc_id=chunk.doc_id,
            title=title or "(제목 없음)",
            publisher=str(meta.get("source", "")).strip(),
            published=str(meta.get("published_at", "") or "")[:10],
            doc_type=str(meta.get("doc_type", "")),
            tier=str(meta.get("source_tier", "")),
            url=str(meta.get("url", "")).strip(),
            passage=strip_title_prefix(chunk.text, title).strip(),
            score=float(chunk.score),
            vector_rank=meta.get("vector_rank"),
            keyword_rank=meta.get("keyword_rank"),
            cited=number in cited_set,
        ))
    return views


def split_sources(sources: Sequence[SourceView], refused: bool) -> tuple[list[SourceView], list[SourceView]]:
    """(먼저 보여 줄 출처, 접어 둘 출처).
    인용된 번호가 있으면 인용된 출처만 펼친다. 거절했으면 모두 접는다 (답변의 근거가 아니므로).
    답변이 번호를 하나도 달지 않았으면 판단 근거를 숨기지 않도록 모두 펼친다."""
    if refused:
        return [], list(sources)
    if not any(s.cited for s in sources):
        return list(sources), []
    return [s for s in sources if s.cited], [s for s in sources if not s.cited]


def _is_punctuation(char: str) -> bool:
    return unicodedata.category(char)[0] in "PS"


def fix_korean_emphasis(text: str) -> str:
    """'**95.8%**입니다' → '**95.8**%입니다', '**'상생'**은' → ''**상생**'은'. 렌더링되는 강조는 그대로 둔다."""
    def repair(match: re.Match[str]) -> str:
        inner = match.group(1)
        before = text[match.start() - 1] if match.start() > 0 else " "
        after = text[match.end()] if match.end() < len(text) else " "
        lead = trail = ""
        if _is_punctuation(inner[0]) and not (before.isspace() or _is_punctuation(before)):
            core = inner.lstrip(_MOVABLE_LEADING)
            lead, inner = inner[:len(inner) - len(core)], core
        if inner and _is_punctuation(inner[-1]) and not (after.isspace() or _is_punctuation(after)):
            core = inner.rstrip(_MOVABLE_TRAILING)
            trail, inner = inner[len(core):], core
        if not inner or inner[0].isspace() or inner[-1].isspace():
            return match.group(0)
        return f"{lead}**{inner}**{trail}"

    return _EMPHASIS.sub(repair, text)


def format_answer_markdown(text: str, n_sources: int) -> str:
    """답변을 Streamlit 마크다운으로. 범위 안의 [n] 은 번호 배지, 범위 밖 [n] 은 문자 그대로 보인다.
    '[1](…)' 이 링크로 해석되는 것도 막는다."""
    escaped = _MARKDOWN_SPECIAL.sub(r"\\\1", fix_korean_emphasis(text))

    def citation(match: re.Match[str]) -> str:
        number = int(match.group(1))
        return f":blue-badge[{number}]" if 1 <= number <= n_sources else rf"\[{number}\]"

    return _CITATION.sub(citation, escaped)


def build_filters(doc_types: Sequence[str] | None, dates: Sequence[date] | None) -> SearchFilters | None:
    """필터 위젯 값 → SearchFilters. 아무것도 고르지 않으면 None (전체 검색).
    날짜를 하나만 고른 상태(범위 선택 중)는 시작일만 적용한다."""
    types = tuple(DocType(t) for t in (doc_types or ()))
    picked = sorted(d for d in (dates or ()) if d is not None)
    date_from = picked[0] if picked else None
    date_to = picked[-1] if len(picked) >= 2 else None
    if not types and date_from is None:
        return None
    return SearchFilters(doc_types=types, date_from=date_from, date_to=date_to)


def describe_filters(filters: SearchFilters | None) -> str:
    """'보도자료·서울시 공고 · 2025.01.01 ~ 2025.12.31' 형태. 필터가 없으면 '전체 자료'."""
    if filters is None:
        return "전체 자료"
    parts = []
    if filters.doc_types:
        parts.append("·".join(DOC_TYPE_LABELS.get(t, t) for t in filters.doc_types))
    if filters.date_from or filters.date_to:
        start = filters.date_from.strftime("%Y.%m.%d") if filters.date_from else ""
        end = filters.date_to.strftime("%Y.%m.%d") if filters.date_to else ""
        parts.append(f"{start} ~ {end}".strip())
    return " · ".join(parts) or "전체 자료"


def _flexible_pattern(piece: str) -> re.Pattern[str] | None:
    words = piece.split()
    return re.compile(r"\s+".join(map(re.escape, words))) if words else None


def find_highlight_spans(text: str, passage: str) -> list[tuple[int, int]]:
    """근거 문단(떨어진 청크는 PASSAGE_GAP 으로 이어져 있음)의 각 조각이 원문에서 차지하는 구간.
    정확히 일치하지 않으면 공백 차이를 무시하고 찾는다. 찾지 못한 조각은 건너뛴다.
    반환: 겹치지 않게 합친 (시작, 끝) 목록, 시작 순."""
    spans: list[tuple[int, int]] = []
    for raw_piece in passage.split(PASSAGE_GAP):
        piece = raw_piece.strip()
        if len(piece) < MIN_PIECE_CHARS:
            continue
        start = text.find(piece)
        if start >= 0:
            spans.append((start, start + len(piece)))
            continue
        pattern = _flexible_pattern(piece)
        match = pattern.search(text) if pattern else None
        if match:
            spans.append(match.span())
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def tier_summary(sources: Sequence[SourceView]) -> str:
    """'1차 자료 2 · 언론 보도 3' (SourceTier 정의 순서)."""
    counts = Counter(s.tier for s in sources if s.tier)
    return " · ".join(f"{tier} {counts[tier]}" for tier in SourceTier if counts[tier])
