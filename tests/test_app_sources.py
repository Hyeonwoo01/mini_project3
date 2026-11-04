from datetime import date

import pytest

from app.sources import (
    SourceView,
    build_filters,
    describe_filters,
    find_highlight_spans,
    fix_korean_emphasis,
    format_answer_markdown,
    split_sources,
    strip_title_prefix,
    tier_summary,
    to_source_views,
)
from common.schema import DocType
from rag.retriever import RetrievedChunk, SearchFilters


def _chunk(doc_id: str, title: str, body: str, **meta) -> RetrievedChunk:
    metadata = {"title": title, "source": "금융위원회", "published_at": "2025-12-30T09:00:00",
                "doc_type": "press_release", "source_tier": "1차 자료", "url": f"https://example.com/{doc_id}"}
    metadata.update(meta)
    return RetrievedChunk(chunk_id=f"{doc_id}#0", doc_id=doc_id, text=f"{title}\n\n{body}", score=0.71,
                          metadata=metadata)


def _source(number: int, cited: bool, tier: str = "언론 보도") -> SourceView:
    return SourceView(number=number, doc_id=f"d{number}", title="t", publisher="p", published="2025-01-01",
                      doc_type="news", tier=tier, url="", passage="본문", score=0.6, vector_rank=None,
                      keyword_rank=None, cited=cited)


def test_to_source_views_numbers_chunks_and_strips_title_prefix():
    chunks = [
        _chunk("kr_1", "우대수수료율 적용 대상 확정", "영세·중소 가맹점은 310.2만개다.", vector_rank=1, keyword_rank=2),
        _chunk("news_1", "기사 제목", "기사 본문", published_at="", doc_type="news", source_tier="언론 보도", url=" "),
    ]

    views = to_source_views(chunks, cited=[2, 9])

    assert [v.number for v in views] == [1, 2]
    assert views[0].passage == "영세·중소 가맹점은 310.2만개다."
    assert views[0].published == "2025-12-30"
    assert (views[0].vector_rank, views[0].keyword_rank) == (1, 2)
    assert views[0].doc_type_label == "보도자료"
    assert [v.cited for v in views] == [False, True]
    assert views[1].published == "" and views[1].url == ""


def test_strip_title_prefix_only_removes_exact_header():
    assert strip_title_prefix("제목\n\n본문", "제목") == "본문"
    assert strip_title_prefix("본문 제목\n\n", "제목") == "본문 제목\n\n"
    assert strip_title_prefix("본문", "") == "본문"


@pytest.mark.parametrize(
    ("cited", "refused", "shown", "folded"),
    [
        ([True, False, True], False, [1, 3], [2]),   # 인용된 것만 펼침
        ([False, False], False, [1, 2], []),         # 번호를 안 달았으면 모두 펼침
        ([True, False], True, [], [1, 2]),           # 거절이면 모두 접음
    ],
)
def test_split_sources(cited, refused, shown, folded):
    sources = [_source(i, c) for i, c in enumerate(cited, start=1)]
    first, rest = split_sources(sources, refused)
    assert [s.number for s in first] == shown
    assert [s.number for s in rest] == folded


def test_format_answer_markdown_turns_citations_into_badges():
    text = "총 310.2만개다[1][2]. 범위 밖 번호[7]는 그대로."
    assert format_answer_markdown(text, 2) == r"총 310.2만개다:blue-badge[1]:blue-badge[2]. 범위 밖 번호\[7\]는 그대로."


def test_format_answer_markdown_escapes_strikethrough_math_and_links():
    formatted = format_answer_markdown("9~11월 누적, 3~5% 인상, $10 [1](중기부)", 1)
    assert formatted == r"9\~11월 누적, 3\~5% 인상, \$10 :blue-badge[1](중기부)"
    assert format_answer_markdown(r"이미 \~ 처리", 0) == r"이미 \~ 처리"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("전체의 **95.8%**입니다", "전체의 **95.8**%입니다"),         # 닫는 ** 앞 문장부호 + 뒤 글자
        ("**총 310.2만개**로", "**총 310.2만개**로"),                 # 이미 렌더링되는 강조
        ("사업명은 '**\"상생\"**'이다", "사업명은 '**\"상생\"**'이다"),  # 앞뒤가 문장부호면 그대로 렌더링됨
        ("이번**'상생페이백'**은", "이번'**상생페이백**'은"),          # 여는 쪽도 옮김
        ("비율은 **95.8%**.", "비율은 **95.8%**."),
        ("**%**이다", "**%**이다"),                                   # 옮기면 비는 경우 그대로
    ],
)
def test_fix_korean_emphasis(text, expected):
    assert fix_korean_emphasis(text) == expected


def test_build_filters():
    assert build_filters(None, None) is None
    assert build_filters([], ()) is None
    assert build_filters([DocType.PRESS_RELEASE], None) == SearchFilters(doc_types=(DocType.PRESS_RELEASE,))
    # 범위 선택 중(날짜 하나)은 시작일만, 순서가 뒤집혀도 정렬한다
    assert build_filters(None, (date(2025, 3, 1),)) == SearchFilters(date_from=date(2025, 3, 1))
    assert build_filters(["news"], (date(2025, 12, 31), date(2025, 1, 1))) == SearchFilters(
        doc_types=(DocType.NEWS,), date_from=date(2025, 1, 1), date_to=date(2025, 12, 31))


def test_describe_filters():
    assert describe_filters(None) == "전체 자료"
    filters = SearchFilters(doc_types=(DocType.PRESS_RELEASE, DocType.ANNOUNCEMENT),
                            date_from=date(2025, 1, 1), date_to=date(2025, 12, 31))
    assert describe_filters(filters) == "보도자료·서울시 공고 · 2025.01.01 ~ 2025.12.31"
    assert describe_filters(SearchFilters(date_from=date(2025, 1, 1))) == "2025.01.01 ~"


DOCUMENT = (
    "금융위원회는 우대수수료율 적용 대상을 확정했다.\n\n"
    "연 매출 30억원 이하 가맹점 중 영세·중소 신용카드가맹점은 총 310.2만개로 전체의 95.8%다.\n\n"
    "신규 가맹점은 업종 평균 수수료율을 적용받은 뒤 차액을 환급받는다.\n\n"
    "끝."
)


def test_find_highlight_spans_exact_and_gapped_pieces():
    first = "연 매출 30억원 이하 가맹점 중 영세·중소 신용카드가맹점은 총 310.2만개로 전체의 95.8%다."
    second = "신규 가맹점은 업종 평균 수수료율을 적용받은 뒤 차액을 환급받는다."
    spans = find_highlight_spans(DOCUMENT, f"{first}\n…\n{second}")
    assert [DOCUMENT[s:e] for s, e in spans] == [first, second]


def test_find_highlight_spans_tolerates_whitespace_differences_and_merges_overlaps():
    passage = "연 매출 30억원 이하 가맹점 중\n영세·중소   신용카드가맹점은 총 310.2만개로"
    overlapping = "신용카드가맹점은 총 310.2만개로 전체의 95.8%다."
    spans = find_highlight_spans(DOCUMENT, f"{passage}\n…\n{overlapping}")
    assert len(spans) == 1
    assert DOCUMENT[slice(*spans[0])].startswith("연 매출") and DOCUMENT[slice(*spans[0])].endswith("95.8%다.")


def test_find_highlight_spans_skips_short_or_missing_pieces():
    assert find_highlight_spans(DOCUMENT, "끝.") == []
    assert find_highlight_spans(DOCUMENT, "원문에 없는 문장이 여기에 들어 있습니다. 찾을 수 없다.") == []
    assert find_highlight_spans(DOCUMENT, "") == []


def test_tier_summary_follows_tier_order():
    sources = [_source(1, True, "언론 보도"), _source(2, True, "1차 자료"), _source(3, False, "언론 보도")]
    assert tier_summary(sources) == "1차 자료 1 · 언론 보도 2"
    assert tier_summary([]) == ""
