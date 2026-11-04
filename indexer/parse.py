"""PDF·HTML 텍스트 추출과 보일러플레이트 제거.

- HTML: 본문 영역만 추출하고 메뉴·머리글·꼬리말·저작권 문구를 제거한다
- PDF: 쪽 번호 줄을 제거한다 (보도자료 머리 표의 담당 부서·연락처 처리는 실제 추출 결과를 보고 정한다)
- 문서 종류별 제거 규칙은 EDA(notebooks/eda.ipynb)에서 확인한 구조를 근거로 정한다

제공 뉴스는 이미 정제되어 있다 (EDA 4번: 저작권 문구 0.3%, 기자 이메일 0%).
그래서 clean_text 는 공통 정규화와 소수의 저작권·AI 학습 금지 문구 제거만 한다.
원본(documents.raw_text)은 수정하지 않고 정제 본문(processed_documents.clean_text)에만 적용한다.
"""

import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import pymupdf

from common.config import PROJECT_ROOT

# U+2028(LS)·U+2029(PS)는 str.splitlines() 가 줄바꿈으로 취급해 청킹·길이 계산을 어긋나게 한다.
_CHAR_MAP = str.maketrans({" ": "\n", " ": "\n", " ": " "})
_ZERO_WIDTH = re.compile("[​‌‍⁠﻿]")

# 문장 내용을 지우지 않도록 저작권 표기 형태가 분명한 문구만 제거한다.
_BOILERPLATE = [
    re.compile(r"(?:Copyright|ⓒ|©)\s*[^\n]{0,60}?All\s+rights\s+reserved\.?", re.IGNORECASE),
    re.compile(r"(?:Copyright\s*)?(?:ⓒ|©)\s*[^\n]{0,30}?(?=무단)"),
    re.compile(r"무단\s*전재[^\n]{0,30}?금지"),
    re.compile(r"(?:,\s*)?AI\s*학습\s*(?:및\s*활용|이용|활용)?\s*금지"),
]
_SPACES = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n\s*\n+")


# 쪽 번호 줄: "- 3 -". 숫자만 있는 줄은 표의 셀일 수 있어 지우지 않는다
_PAGE_NUMBER_LINE = re.compile(r"^\s*-\s*\d+\s*-\s*$")


# HWPX(OWPML) 본문: zip 안의 Contents/section0.xml, section1.xml ... 에 문단(hp:p)과 글자(hp:t)가 있다
_HWPX_SECTION = re.compile(r"Contents/section(\d+)\.xml")
MAX_HWPX_XML_BYTES = 50 * 1024 * 1024   # 압축 해제 크기 상한 (비정상 파일 방어)


class ParseError(Exception):
    """원본 파일을 열 수 없거나 지원하지 않는 형식."""


# MuPDF 는 고칠 수 있는 PDF 구조 오류(예: 'syntax error: invalid key in dict')도 표준 오류에 바로 출력한다.
# 출력은 끄고 경고 기록으로 모아서, 호출한 쪽이 문서 단위로 보고하게 한다.
pymupdf.TOOLS.mupdf_display_errors(False)


def extract_text(raw_path: str, warnings: list[str] | None = None) -> str:
    """직접 수집한 원본 파일에서 본문을 추출한다. raw_path 는 프로젝트 루트 기준 상대 경로도 받는다.

    PDF (정책브리핑 보도자료 원문): 쪽마다 읽는 순서대로 텍스트를 뽑고 쪽 번호만 있는 줄을 뺀다.
    표는 셀 글자가 줄 단위로 나온다. 스캔 이미지 PDF 는 글자가 거의 없어 품질 필터에서 제외된다.
    HWPX (PDF 가 없는 보도자료): 구역(section) 순서대로 문단마다 한 줄. 표 안의 문단도 각각 한 줄이 된다.
    warnings 를 넘기면 MuPDF 가 복구하며 남긴 메시지를 추가한다 (텍스트는 정상 반환).
    """
    path = Path(raw_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    suffix = path.suffix.lower()
    if suffix == ".hwpx":
        return _hwpx_text(path, raw_path)
    if suffix != ".pdf":
        raise ParseError(f"지원하지 않는 원본 형식입니다: {raw_path}")
    pymupdf.TOOLS.reset_mupdf_warnings()
    try:
        with pymupdf.open(path) as doc:
            pages = [page.get_text("text", sort=True) for page in doc]
    except (pymupdf.FileDataError, FileNotFoundError, RuntimeError) as exc:
        raise ParseError(f"PDF 를 열 수 없습니다: {raw_path} ({exc})") from exc
    if warnings is not None and (messages := pymupdf.TOOLS.mupdf_warnings().strip()):
        warnings.append(" / ".join(dict.fromkeys(messages.splitlines())))
    lines = [line for page in pages for line in page.splitlines() if not _PAGE_NUMBER_LINE.match(line)]
    return "\n".join(lines)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _hwpx_text(path: Path, raw_path: str) -> str:
    try:
        with zipfile.ZipFile(path) as archive:
            sections = sorted((int(m.group(1)), name) for name in archive.namelist()
                              if (m := _HWPX_SECTION.fullmatch(name)))
            if not sections:
                raise ParseError(f"HWPX 본문 구역이 없습니다: {raw_path}")
            lines: list[str] = []
            for _, name in sections:
                if archive.getinfo(name).file_size > MAX_HWPX_XML_BYTES:
                    raise ParseError(f"HWPX 구역이 너무 큽니다: {raw_path} {name}")
                _collect_paragraphs(ElementTree.fromstring(archive.read(name)), lines)
    except (zipfile.BadZipFile, ElementTree.ParseError, FileNotFoundError) as exc:
        raise ParseError(f"HWPX 를 열 수 없습니다: {raw_path} ({exc})") from exc
    return "\n".join(lines)


def _collect_paragraphs(root: ElementTree.Element, lines: list[str]) -> None:
    """문단(p)마다 글자(t)를 한 줄로 모은다. 표가 든 문단은 표 앞 글자, 셀 문단들, 표 뒤 글자 순서가 된다."""
    current: list[str] = []

    def flush() -> None:
        if (line := "".join(current).strip()):
            lines.append(line)
        current.clear()

    def visit(element: ElementTree.Element) -> None:
        for child in element:
            name = _local(child.tag)
            if name == "p":
                flush()
                visit(child)
                flush()
            elif name == "t":
                current.append("".join(child.itertext()))
            else:
                visit(child)

    visit(root)
    flush()


def clean_text(text: str) -> str:
    """공통 정규화 후 보일러플레이트를 제거한다.

    - LS·PS 는 줄바꿈으로, 줄바꿈 없는 공백(U+00A0)은 공백으로 바꾸고 폭 없는 문자는 제거한다
    - 저작권 표기(Copyright·ⓒ … All rights reserved, 무단 전재 … 금지, AI 학습 금지)를 제거한다
    - 연속 공백과 빈 줄을 정리한다
    """
    text = _ZERO_WIDTH.sub("", text.translate(_CHAR_MAP))
    for pattern in _BOILERPLATE:
        text = pattern.sub(" ", text)
    text = _SPACES.sub(" ", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()
