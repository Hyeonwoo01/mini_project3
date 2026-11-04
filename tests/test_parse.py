import zipfile

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


HP = "http://www.hancom.co.kr/hwpml/2011/paragraph"


def _hwpx(path, sections):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("mimetype", "application/hwp+zip")
        for name, body in sections.items():
            archive.writestr(name, f'<hs:sec xmlns:hs="http://www.hancom.co.kr/hwpml/2011/section" '
                                   f'xmlns:hp="{HP}">{body}</hs:sec>')


def test_extract_text_reads_hwpx_paragraphs_in_order_including_tables(tmp_path):
    path = tmp_path / "press.hwpx"
    table = ('<hp:tbl><hp:tr><hp:tc><hp:subList><hp:p><hp:run><hp:t>구분</hp:t></hp:run></hp:p></hp:subList></hp:tc>'
             '<hp:tc><hp:subList><hp:p><hp:run><hp:t>3조 4,582억원</hp:t></hp:run></hp:p></hp:subList></hp:tc>'
             '</hp:tr></hp:tbl>')
    _hwpx(path, {
        "Contents/section1.xml": "<hp:p><hp:run><hp:t>둘째 구역</hp:t></hp:run></hp:p>",
        "Contents/section0.xml": (
            "<hp:p><hp:run><hp:t>소상공인 </hp:t></hp:run><hp:run><hp:t>지원사업 시행</hp:t></hp:run></hp:p>"
            f"<hp:p><hp:run><hp:t>표 앞</hp:t>{table}<hp:t>표 뒤</hp:t></hp:run></hp:p>"
            "<hp:p><hp:run><hp:t>  </hp:t></hp:run></hp:p>"),
    })

    assert extract_text(str(path)).splitlines() == [
        "소상공인 지원사업 시행", "표 앞", "구분", "3조 4,582억원", "표 뒤", "둘째 구역"]


def test_extract_text_rejects_hwpx_without_sections(tmp_path):
    path = tmp_path / "empty.hwpx"
    _hwpx(path, {})
    with pytest.raises(ParseError, match="구역"):
        extract_text(str(path))
    (tmp_path / "broken.hwpx").write_bytes(b"PK not really")
    with pytest.raises(ParseError):
        extract_text(str(tmp_path / "broken.hwpx"))


def test_extract_text_records_mupdf_repairs_as_warnings_without_printing(tmp_path, capfd):
    pdf = pymupdf.open()
    pdf.new_page().insert_text((72, 72), "body")
    data = pdf.tobytes()
    at = data.find(b"/Type/Pages")
    path = tmp_path / "repaired.pdf"
    path.write_bytes(data[:at] + b"7 " + data[at:])     # 사전에 이름이 아닌 키 → MuPDF 가 복구하며 오류 메시지

    warnings: list[str] = []
    extract_text(str(path), warnings)

    assert len(warnings) == 1 and "invalid key in dict" in warnings[0]
    assert "MuPDF error" not in capfd.readouterr().err


def _write_pdf(path, width, height, texts):
    pdf = pymupdf.open()
    page = pdf.new_page(width=width, height=height)
    for (x, y), text in texts:
        page.insert_text((x, y), text, fontname="korea")
    pdf.save(path)
    return str(path)


def test_extract_text_reads_two_up_pages_left_then_right(tmp_path):
    # A4 두 쪽을 가로 한 장에 담은 보도자료: 같은 높이의 줄이 왼쪽·오른쪽 쪽에 따로 있다
    path = _write_pdf(tmp_path / "2up.pdf", 842, 595, [
        ((50, 100), "왼쪽 첫 문단"), ((471, 100), "오른쪽 첫 문단"),
        ((50, 130), "왼쪽 둘째 문단"), ((471, 130), "오른쪽 둘째 문단"),
        ((200, 570), "- 1 -"), ((620, 570), "- 2 -"),
    ])
    lines = [line for line in extract_text(path).splitlines() if line.strip()]
    assert lines == ["왼쪽 첫 문단", "왼쪽 둘째 문단", "오른쪽 첫 문단", "오른쪽 둘째 문단"]


def test_extract_text_keeps_reading_order_on_landscape_page_spanning_middle(tmp_path):
    # 가운데를 가로지르는 표·문장이 있는 가로 쪽은 한 쪽으로 보고 위아래 순서대로 읽는다
    path = _write_pdf(tmp_path / "wide.pdf", 842, 595, [
        ((50, 100), "가로로 긴 표 제목이 쪽 가운데를 지나 오른쪽까지 이어지는 한 줄입니다 계속 이어집니다"),
        ((50, 130), "왼쪽 셀"), ((471, 130), "오른쪽 셀"),
    ])
    text = extract_text(path)
    # 반으로 자르면 가운데를 지나는 문장이 잘린다. 한 쪽으로 읽으면 문장이 온전하고 위아래 순서가 유지된다
    assert "가로로 긴 표 제목이 쪽 가운데를 지나 오른쪽까지 이어지는 한 줄입니다 계속 이어집니다" in text
    assert text.index("가로로") < text.index("왼쪽 셀") < text.index("오른쪽 셀")


def test_extract_text_drops_official_contact_lines_but_keeps_call_centers(tmp_path):
    path = _write_pdf(tmp_path / "contacts.pdf", 595, 842, [
        ((50, 100), "소상공인에게 최대 30만원을 지원한다."),
        ((50, 130), "(소상공인시장진흥공단 콜센터 : ☎1533-0600)"),
        ((50, 160), "책임자 과 장 안원호 (044-204-7821)"),
        ((50, 190), "사무관 김태선 (044-204-7853)"),
        ((50, 220), "부담경감 주무관 전보람 (044-204-7863)"),
    ])
    lines = [line for line in extract_text(path).splitlines() if line.strip()]
    assert lines == ["소상공인에게 최대 30만원을 지원한다.", "(소상공인시장진흥공단 콜센터 : ☎1533-0600)"]
