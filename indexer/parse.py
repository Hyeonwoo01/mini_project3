"""PDF·HTML 텍스트 추출과 보일러플레이트 제거.

- HTML: 본문 영역만 추출하고 메뉴·머리글·꼬리말·저작권 문구를 제거한다
- PDF: 페이지 머리글·꼬리말을 제거하고, 표는 본문과 분리하여 별도 처리한다
- 문서 종류별 제거 규칙은 EDA(notebooks/eda.ipynb)에서 확인한 구조를 근거로 정한다

제공 뉴스는 이미 정제되어 있다 (EDA 4번: 저작권 문구 0.3%, 기자 이메일 0%).
그래서 clean_text 는 공통 정규화와 소수의 저작권·AI 학습 금지 문구 제거만 한다.
원본(documents.raw_text)은 수정하지 않고 정제 본문(processed_documents.clean_text)에만 적용한다.
"""

import re

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


def extract_text(raw_path: str) -> str:
    """직접 수집한 원본 파일(HTML/PDF)에서 본문을 추출한다. 수집 소스가 확정되면 구현한다."""
    raise NotImplementedError(f"원본 파일 파싱은 아직 구현되지 않았습니다: {raw_path}")


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
