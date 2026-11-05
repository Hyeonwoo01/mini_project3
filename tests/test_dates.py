from dataclasses import replace

import pytest

from indexer.chunk import chunk_document, get_chunk_config
from common.dates import annotate_relative_dates
from tests.test_chunk_embed import _source


@pytest.mark.parametrize(("text", "published_at", "expected"), [
    ("내년 예산 16조8천억원", "2025-09-02T10:00:00", "내년(2026년) 예산 16조8천억원"),
    ("내년도 예산안", "2025-09-02", "내년도(2026년) 예산안"),
    ("재작년보다 늘고 작년보다 줄었다", "2025-09-02", "재작년(2023년)보다 늘고 작년(2024년)보다 줄었다"),
    ("내일부터 지급한다", "2025-02-20T15:00:00", "내일(2025년 2월 21일)부터 지급한다"),
    ("오늘 발표", "2025-12-31", "오늘(2025년 12월 31일) 발표"),
    ("내일 시작", "2025-12-31", "내일(2026년 1월 1일) 시작"),
    ("다음 달부터 신청", "2025-12-15", "다음 달(2026년 1월)부터 신청"),
    ("지난달 매출", "2026-01-08", "지난달(2025년 12월) 매출"),
])
def test_relative_dates_are_resolved_from_publication_date(text, published_at, expected):
    annotated, count = annotate_relative_dates(text, published_at)
    assert annotated == expected and count == expected.count("(")


@pytest.mark.parametrize("text", [
    "청년내일채움공제와 내일배움카드",          # 다른 낱말·사업명의 일부
    "오늘날 골목상권",
    "내년(2026년) 예산",                       # 이미 날짜가 있다
    "올해 2025년 예산은",
    "전년 대비 3% 증가",                       # 기준 시점이 문맥에 따라 다르다
])
def test_words_that_are_not_relative_dates_are_left_alone(text):
    assert annotate_relative_dates(text, "2025-09-02") == (text, 0)


@pytest.mark.parametrize("published_at", [None, "", "2025.09.02", "2025-02-30"])
def test_unknown_publication_date_leaves_text_unchanged(published_at):
    assert annotate_relative_dates("내년 예산", published_at) == ("내년 예산", 0)


def test_dates_config_annotates_chunk_text_but_keeps_metadata_title():
    source = replace(_source(text="중기부는 내일부터 신청을 받는다. " * 3), title="내년 예산 확정")
    [chunk] = chunk_document(source, get_chunk_config("c500-dates"))
    assert chunk.text.startswith("내년(2027년) 예산 확정\n\n중기부는 내일(2026년 1월 9일)부터")
    assert chunk.metadata["title"] == "내년 예산 확정"
    [plain] = chunk_document(source, get_chunk_config("c500"))
    assert "(2027년)" not in plain.text and "(2026년 1월 9일)" not in plain.text
