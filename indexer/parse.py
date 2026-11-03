"""PDF·HTML 텍스트 추출과 보일러플레이트 제거.

- HTML: 본문 영역만 추출하고 메뉴·머리글·꼬리말·저작권 문구를 제거한다
- PDF: 쪽 번호 줄을 제거한다 (보도자료 머리 표의 담당 부서·연락처 처리는 실제 추출 결과를 보고 정한다)
- 문서 종류별 제거 규칙은 EDA(notebooks/eda.ipynb)에서 확인한 구조를 근거로 정한다

제공 뉴스는 이미 정제되어 있다 (EDA 4번: 저작권 문구 0.3%, 기자 이메일 0%).
그래서 clean_text 는 공통 정규화와 소수의 저작권·AI 학습 금지 문구 제거만 한다.
원본(documents.raw_text)은 수정하지 않고 정제 본문(processed_documents.clean_text)에만 적용한다.
"""

import re
from pathlib import Path

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


class ParseError(Exception):
    """원본 파일을 열 수 없거나 지원하지 않는 형식."""


def extract_text(raw_path: str) -> str:
    """직접 수집한 원본 파일에서 본문을 추출한다. raw_path 는 프로젝트 루트 기준 상대 경로도 받는다.

    PDF (정책브리핑 보도자료 원문): 쪽마다 읽는 순서대로 텍스트를 뽑고 쪽 번호만 있는 줄을 뺀다.
    표는 셀 글자가 줄 단위로 나온다. 스캔 이미지 PDF 는 글자가 거의 없어 품질 필터에서 제외된다.
    """
    path = Path(raw_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if path.suffix.lower() != ".pdf":
        raise ParseError(f"지원하지 않는 원본 형식입니다: {raw_path}")
    try:
        with pymupdf.open(path) as doc:
            pages = [page.get_text("text", sort=True) for page in doc]
    except (pymupdf.FileDataError, FileNotFoundError, RuntimeError) as exc:
        raise ParseError(f"PDF 를 열 수 없습니다: {raw_path} ({exc})") from exc
    lines = [line for page in pages for line in page.splitlines() if not _PAGE_NUMBER_LINE.match(line)]
    return "\n".join(lines)


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
