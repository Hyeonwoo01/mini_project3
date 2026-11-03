from common.text import keyword_terms, query_terms


def test_keyword_terms_use_hangul_bigrams_and_whole_numbers():
    assert keyword_terms("2027년도 예산안의 3916억원, KDI") == [
        "2027", "년도", "예산", "산안", "안의", "3916", "억원", "kdi"]
    assert keyword_terms("은 년 억") == []          # 한글 한 글자 덩어리는 버린다


def test_query_terms_drop_question_words_and_duplicates():
    terms = query_terms("2026년 중기부 예산은 정부안과 국회 확정안이 각각 얼마였나?")
    assert "2026" in terms and "중기" in terms and "확정" in terms
    assert not {"각각", "얼마", "마였", "였나"} & set(terms)
    assert query_terms("코로나 이후 매출은 어떻게 달라졌나?") == ["코로", "로나", "이후", "매출", "출은"]
    assert len(terms) == len(set(terms))
