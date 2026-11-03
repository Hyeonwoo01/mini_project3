import pymupdf
import pytest

from indexer.parse import ParseError, clean_text, extract_text


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


def test_extract_text_reads_pdf_and_drops_page_number_lines(tmp_path):
    pdf = pymupdf.open()
    for n in (1, 2):
        page = pdf.new_page()
        page.insert_text((72, 72), f"page {n} body", fontname="helv")
        page.insert_text((72, 100), "소상공인 지원", fontname="korea")
        page.insert_text((72, 128), "3", fontname="helv")             # 표의 숫자 셀은 남긴다
        page.insert_text((280, 800), f"- {n} -", fontname="helv")
    path = tmp_path / "press.pdf"
    pdf.save(path)

    lines = [line for line in extract_text(str(path)).splitlines() if line.strip()]
    assert lines == ["page 1 body", "소상공인 지원", "3", "page 2 body", "소상공인 지원", "3"]


@pytest.mark.parametrize("name, content", [("broken.pdf", b"not a pdf"), ("missing.pdf", None), ("doc.hwp", b"x")])
def test_extract_text_raises_parse_error(tmp_path, name, content):
    path = tmp_path / name
    if content is not None:
        path.write_bytes(content)
    with pytest.raises(ParseError):
        extract_text(str(path))
