"""키워드 검색용 토큰화 — 색인(indexer)과 질의(rag)가 같은 규칙을 써야 하므로 공용 계층에 둔다.

형태소 분석기 없이 한국어를 다루는 방법으로 글자 2-gram 을 쓴다.
    "2027년도 예산안"  →  2027 / 년도 / 예산 / 산안
- 숫자는 통째로 하나의 토큰이다 (연도·금액이 질문의 핵심인 경우가 많다: 2027, 3916)
- 한글 덩어리는 2글자씩 겹쳐 자른다. 조사가 붙어도 어간 2-gram 은 남는다 (예산안의 → 예산 / 산안 / 안의)
- 한글 한 글자 덩어리(조사·단위: 년, 억, 은)는 버린다
- 영문은 소문자로 바꿔 통째로 쓴다

질의에서는 의문사·의문형 어미로 된 어절(어떻게, 얼마인가, 달라졌나 등)을 뺀다. 기사 본문에는 드물어서
BM25 의 역문서빈도가 높게 잡히고, 그 결과 내용과 무관한 칼럼·사설이 위로 올라오기 때문이다.
"""

import re

_TOKEN = re.compile(r"\d+|[가-힣]+|[a-z]+")
_EOJEOL = re.compile(r"[0-9A-Za-z가-힣]+")

# 질문에서 정보가 없는 어절. 의문사로 시작하거나 의문형 어미로 끝나는 경우
_QUESTION_WORD = re.compile(
    r"^(어떻|어떤|어땠|무엇|무슨|얼마|언제|왜|각각)"
    r"|(었나|았나|였나|졌나|했나|왔나|있나|없나|됐나|폈나|땠나|는가|인가|은가|던가|른가)$"
)


def keyword_terms(text: str) -> list[str]:
    """색인용 토큰. 순서와 중복을 유지한다 (BM25 의 단어 빈도에 쓰인다)."""
    terms: list[str] = []
    for token in _TOKEN.findall(text.lower()):
        if token.isascii():                 # 숫자·영문은 통째로
            terms.append(token)
        elif len(token) >= 2:               # 한글은 2글자씩 겹쳐서
            terms.extend(token[i:i + 2] for i in range(len(token) - 1))
    return terms


def query_terms(question: str) -> list[str]:
    """질의용 토큰. 의문사·의문형 어절을 뺀 뒤 색인과 같은 규칙으로 자르고 중복을 없앤다."""
    words = [w for w in _EOJEOL.findall(question) if not _QUESTION_WORD.search(w)]
    return list(dict.fromkeys(keyword_terms(" ".join(words))))
