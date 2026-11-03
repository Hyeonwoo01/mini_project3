"""PDF·HTML 텍스트 추출과 보일러플레이트 제거.

- HTML: 본문 영역만 추출하고 메뉴·머리글·꼬리말·저작권 문구를 제거한다
- PDF: 페이지 머리글·꼬리말을 제거하고, 표는 본문과 분리하여 별도 처리한다
- 문서 종류별 제거 규칙은 EDA(notebooks/eda.ipynb)에서 확인한 구조를 근거로 정한다
"""


def extract_text(raw_path: str) -> str:
    raise NotImplementedError


def clean_text(text: str) -> str:
    """공통 정규화 후 문서 종류별 보일러플레이트를 제거한다.

    공통 정규화
    - U+2028(LS)·U+2029(PS)는 일반 줄바꿈(\\n)으로 바꾼다. 제공 CSV 일부에 들어 있으며,
      str.splitlines() 등이 줄바꿈으로 취급해 청킹·길이 계산이 어긋날 수 있다.
      원본(documents.raw_text)은 수정하지 않고 정제 본문에만 적용한다.
    - 줄바꿈 없는 공백(U+00A0)은 공백으로, 폭 없는 문자(U+200B 등)는 제거한다.
    """
    raise NotImplementedError
