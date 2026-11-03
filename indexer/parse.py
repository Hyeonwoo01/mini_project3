"""PDF·HTML 텍스트 추출과 보일러플레이트 제거.

- HTML: 본문 영역만 추출하고 메뉴·머리글·꼬리말·저작권 문구를 제거한다
- PDF: 페이지 머리글·꼬리말을 제거하고, 표는 본문과 분리하여 별도 처리한다
- 문서 종류별 제거 규칙은 EDA(notebooks/eda.ipynb)에서 확인한 구조를 근거로 정한다
"""


def extract_text(raw_path: str) -> str:
    raise NotImplementedError


def clean_text(text: str) -> str:
    raise NotImplementedError
