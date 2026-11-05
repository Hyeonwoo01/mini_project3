"""상대 날짜 표현에 발행일 기준 날짜를 덧붙인다.

청킹 설정 normalize_dates(indexer, 미채택)와 답변 근거 설정 annotate_dates(rag)가 함께 쓴다.

기사는 날짜를 '내년', '내일부터'처럼 발행일 기준으로 쓴다. 청크만 보면 언제인지 알 수 없어
'2026년 예산' 질문에 '내년 예산' 기사가 검색되지 않고 (t011), 답변도 '내일부터 지급'을 날짜로 옮기지 못한다 (t007).
원문은 바꾸지 않고 표현 뒤에 괄호로 날짜를 붙인다: '내년' → '내년(2026년)', '내일부터' → '내일(2025년 2월 21일)부터'.

- 발행일이 없거나 형식이 다르면 그대로 둔다
- 앞 글자가 한글이면 다른 낱말의 일부로 보고 건너뛴다 ('청년내일채움공제'). '내일배움카드' 같은 사업명도 건너뛴다
- 이미 괄호나 연도가 바로 뒤에 붙어 있으면 건너뛴다 ('내년(2026년)', '올해 2025년')
- '전년', '이날'은 기준 시점이 문맥에 따라 달라 다루지 않는다
"""

import re
from datetime import date, timedelta

# 긴 표현을 먼저 둔다 ('재작년'이 '작년'으로, '내년도'가 '내년'으로 잘리지 않게).
_YEAR_OFFSETS = {"재작년": -2, "내후년": 2, "내년도": 1, "지난해": -1, "작년": -1, "올해": 0, "금년": 0, "내년": 1}
_MONTH_OFFSETS = {"지난 달": -1, "지난달": -1, "이번 달": 0, "이달": 0, "다음 달": 1, "다음달": 1, "내달": 1}
_DAY_OFFSETS = {"그저께": -2, "어제": -1, "오늘": 0, "내일": 1}

_WORDS = {**_YEAR_OFFSETS, **_MONTH_OFFSETS, **_DAY_OFFSETS}
_PATTERN = re.compile(
    r"(?<![가-힣])(" + "|".join(sorted(map(re.escape, _WORDS), key=len, reverse=True)) + r")"
    r"(?!\s*\(|\s*\d{4}\s*년|날|배움|채움|키움|이음)"
)
_DATE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})")


def _publication_date(published_at: str | None) -> date | None:
    match = _DATE.match(published_at or "")
    if not match:
        return None
    try:
        return date(*map(int, match.groups()))
    except ValueError:
        return None


def _resolve(word: str, base: date) -> str:
    if word in _YEAR_OFFSETS:
        return f"{base.year + _YEAR_OFFSETS[word]}년"
    if word in _MONTH_OFFSETS:
        months = base.year * 12 + base.month - 1 + _MONTH_OFFSETS[word]
        return f"{months // 12}년 {months % 12 + 1}월"
    day = base + timedelta(days=_DAY_OFFSETS[word])
    return f"{day.year}년 {day.month}월 {day.day}일"


def annotate_relative_dates(text: str, published_at: str | None) -> tuple[str, int]:
    """(날짜를 덧붙인 텍스트, 덧붙인 개수). 발행일을 모르면 원문을 그대로 돌려준다."""
    base = _publication_date(published_at)
    if base is None:
        return text, 0
    annotated, count = _PATTERN.subn(lambda m: f"{m.group(1)}({_resolve(m.group(1), base)})", text)
    return annotated, count
