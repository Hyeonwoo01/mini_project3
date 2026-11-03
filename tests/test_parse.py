import pytest

from indexer.parse import clean_text


def test_normalizes_special_whitespace():
    assert clean_text("첫 줄 둘째 셋째 끝​") == "첫 줄\n둘째\n셋째 끝"


@pytest.mark.parametrize(
    "boilerplate",
    [
        "Copyright ⓒ 뉴스1. All rights reserved.",
        "ⓒ 한국경제, 무단전재 및 재배포 금지",
        "무단 전재 및 재배포, AI학습 이용 금지.",
    ],
)
def test_removes_copyright_notices(boilerplate):
    cleaned = clean_text(f"소상공인 정책자금이 확대된다. {boilerplate}")
    assert cleaned.startswith("소상공인 정책자금이 확대된다.")
    assert "무단" not in cleaned and "Copyright" not in cleaned and "AI학습" not in cleaned


def test_keeps_body_sentences_that_mention_ai_or_copyright_topics():
    text = "AI 학습 데이터 활용 지원사업이 소상공인 대상으로 열린다."
    assert clean_text(text) == text


def test_collapses_spaces_and_blank_lines():
    assert clean_text("  가   나\n\n\n\n다  ") == "가 나\n\n다"
