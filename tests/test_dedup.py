import pytest

from common.schema import DocType
from indexer.dedup import (
    DedupConfig,
    DedupInput,
    content_hash,
    count_duplicates,
    deduplicate,
    jaccard,
    shingles,
    similar_pairs,
)

ARTICLE = (
    "서울시는 골목상권 활성화를 위해 올해 소상공인 정책자금 500억 원을 추가 공급한다고 밝혔다. "
    "지원 대상은 매출이 감소한 영세 소상공인이며 금리는 연 2%로 고정된다. "
    "신청은 서울신용보증재단 누리집에서 다음 달 1일부터 받는다. "
) * 3
OTHER = (
    "부산시 전통시장 상인회는 온누리상품권 환급 행사를 열고 방문객 수가 크게 늘었다고 전했다. "
    "행사는 주말마다 이어지며 지역 축제와 연계한 공연도 함께 진행된다. "
) * 3
CONFIG = DedupConfig(name="test", similarity_threshold=0.8, max_day_gap=3)


def _doc(doc_id, text, published_at="2026-01-10T09:00:00", doc_type=DocType.NEWS):
    return DedupInput(doc_id=doc_id, doc_type=doc_type, text=text, published_at=published_at)


def test_shingles_are_deterministic_and_jaccard_bounds():
    assert shingles(ARTICLE) == shingles(ARTICLE)
    assert jaccard(shingles(ARTICLE), shingles(ARTICLE)) == 1.0
    assert jaccard(shingles(ARTICLE), shingles(OTHER)) < 0.1
    assert jaccard(frozenset(), shingles(ARTICLE)) == 0.0


def test_content_hash_ignores_spacing_and_punctuation():
    assert content_hash("소상공인 지원, 확대!") == content_hash("소상공인지원확대")


def test_exact_duplicates_merge_regardless_of_date():
    results = deduplicate([_doc("a", ARTICLE, "2025-01-01"), _doc("b", ARTICLE + "  ", "2026-06-01")], CONFIG)
    assert sum(r.is_duplicate for r in results.values()) == 1
    assert {r.similarity for r in results.values()} == {1.0}


def test_near_duplicate_within_window_is_merged_but_not_outside():
    reprint = ARTICLE.replace("밝혔다", "전했다")
    near = deduplicate([_doc("a", ARTICLE, "2026-01-10"), _doc("b", reprint, "2026-01-12")], CONFIG)
    far = deduplicate([_doc("a", ARTICLE, "2026-01-10"), _doc("b", reprint, "2026-03-01")], CONFIG)
    assert sum(r.is_duplicate for r in near.values()) == 1
    assert sum(r.is_duplicate for r in far.values()) == 0


def test_different_articles_are_kept():
    results = deduplicate([_doc("a", ARTICLE), _doc("b", OTHER)], CONFIG)
    assert not any(r.is_duplicate for r in results.values())


def test_canonical_prefers_primary_source_then_longer_text():
    press = _doc("press", ARTICLE, doc_type=DocType.PRESS_RELEASE)
    news_long = _doc("news_long", ARTICLE + " 추가 설명 문장이 붙은 기사.")
    results = deduplicate([news_long, press], CONFIG)
    assert results["news_long"].canonical_doc_id == "press"

    short = _doc("short", ARTICLE)
    long_ = _doc("long", ARTICLE + " 추가 설명 문장이 붙은 기사.")
    assert deduplicate([short, long_], CONFIG)["short"].canonical_doc_id == "long"


def test_empty_texts_are_not_grouped():
    results = deduplicate([_doc("a", ""), _doc("b", "")], CONFIG)
    assert not any(r.is_duplicate for r in results.values())


def test_similar_pairs_and_count_duplicates():
    docs = [_doc("a", ARTICLE), _doc("b", ARTICLE.replace("밝혔다", "전했다")), _doc("c", OTHER)]
    pairs = similar_pairs(docs, CONFIG, min_similarity=0.3)
    assert [(a, b) for a, b, _ in pairs] == [("a", "b")]
    assert count_duplicates(["a", "b", "c"], pairs, threshold=0.8) == 1
    assert count_duplicates(["a", "b", "c"], pairs, threshold=1.01) == 0


@pytest.mark.parametrize("overrides", [{"similarity_threshold": 0}, {"max_day_gap": -1}, {"shingle_size": 1}])
def test_config_validation(overrides):
    with pytest.raises(ValueError):
        DedupConfig(**{"name": "x", "similarity_threshold": 0.8, **overrides})
