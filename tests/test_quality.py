from datetime import date

import pytest

from common.schema import DocType
from indexer.quality import (
    AMBIGUOUS_DISTRICTS,
    QUALITY_CONFIGS,
    SEOUL_DISTRICTS,
    ExclusionReason,
    QualityConfig,
    QualityDecision,
    QualityFlag,
    QualityInput,
    Scope,
    assess,
    classify_scope,
    find_other_regions,
    get_quality_config,
    lead_text,
)

BODY = "소상공인 지원 정책이 확대된다. " * 20  # 주제어 포함, 약 340자


def _config(**overrides) -> QualityConfig:
    fields = dict(
        name="test",
        period_start=date(2021, 10, 16),
        min_chars={DocType.NEWS: 100},
        max_chars={DocType.NEWS: 10_000},
        lead_chars=300,
        topic_terms=("소상공인", "상권"),
    )
    fields.update(overrides)
    return QualityConfig(**fields)


def _news(title="소상공인 정책자금 확대", text=BODY, published_at="2026-01-08T09:00:00", **kw) -> QualityInput:
    return QualityInput(doc_id="news_1", doc_type=DocType.NEWS, title=title, text=text,
                        source="연합뉴스", published_at=published_at, **kw)


def test_seoul_districts():
    assert len(SEOUL_DISTRICTS) == len(set(SEOUL_DISTRICTS)) == 25
    assert AMBIGUOUS_DISTRICTS <= set(SEOUL_DISTRICTS)


def test_decision_requires_reason_only_when_excluded():
    QualityDecision(excluded=True, reason=ExclusionReason.TOO_SHORT)
    QualityDecision(excluded=False)
    with pytest.raises(ValueError):
        QualityDecision(excluded=True)
    with pytest.raises(ValueError):
        QualityDecision(excluded=False, reason=ExclusionReason.OFF_TOPIC)


@pytest.mark.parametrize(
    "overrides",
    [
        {"min_chars": {DocType.NEWS: 0}},
        {"lead_chars": 0},
        {"topic_terms": ()},
        {"max_chars": {DocType.NEWS: 50}},                 # min 보다 작음
        {"national_period_start": date(2020, 1, 1)},       # period_start 보다 이름
    ],
)
def test_config_validation(overrides):
    _config()
    with pytest.raises(ValueError):
        _config(**overrides)


def test_registered_config_and_unknown_name():
    assert get_quality_config("v1") is QUALITY_CONFIGS["v1"]
    with pytest.raises(SystemExit):
        get_quality_config("does-not-exist")


@pytest.mark.parametrize("dateline", ["(서울=연합뉴스)", "(서울 뉴스1)", "[대전 뉴시스]", "(충주 연합뉴스TV)"])
def test_lead_text_strips_dateline(dateline):
    lead = lead_text("제목", f"{dateline} 홍길동 기자 = 본문", 300)
    assert "서울" not in lead and "대전" not in lead and "충주" not in lead


def test_lead_text_keeps_place_in_parentheses():
    assert "강남구" in lead_text("제목", "(서울 강남구) 골목상권 매출", 300)


@pytest.mark.parametrize(
    "text, expected",
    [
        # 제공 데이터의 실제 형태: 발신지만 서울인 전국 기사, 발신지만 지방인 전국 기사
        ("(서울 뉴스1) 중소벤처기업부가 내년도 전통시장과 소상공인 지원을 확대한다.", Scope.NATIONAL),
        ("(대전 연합뉴스) 소상공인시장진흥공단은 폐업 소상공인 지원을 늘린다.", Scope.NATIONAL),
        ("(서울 뉴스1) 서울시는 골목상권 지원을 확대한다.", Scope.SEOUL),
    ],
)
def test_scope_ignores_dateline_without_equals_sign(text, expected):
    assert classify_scope(lead_text("소상공인 지원", text, 300)) == expected


@pytest.mark.parametrize(
    "lead, expected",
    [
        ("서울 마포구 골목상권 임대료가 올랐다", Scope.SEOUL),
        ("마포구 망원시장 상인회", Scope.SEOUL),
        ("서울경제 단독 소상공인 정책자금 확대", Scope.NATIONAL),   # 매체명은 서울 근거가 아님
        ("중구 상인회가 전통시장 축제를 열었다", Scope.NATIONAL),    # 중구는 다른 도시에도 있음
        ("경기 하남시는 골목상권 활성화를 위해", Scope.OTHER_REGION),
        ("고양특례시가 지역화폐 인센티브를 재개했다", Scope.OTHER_REGION),
        ("부산 소상공인 지원", Scope.OTHER_REGION),
        ("경기 침체로 소상공인 폐업이 늘었다", Scope.NATIONAL),      # '경기 침체'는 경기도가 아님
        ("디지털 대전환 지원사업", Scope.NATIONAL),                 # '대전환'은 대전이 아님
        ("대도시 전통시장은 신도시보다", Scope.NATIONAL),
        ("서울시와 부산시가 협약", Scope.SEOUL),                    # 서울 언급이 우선
    ],
)
def test_classify_scope(lead, expected):
    assert classify_scope(lead) == expected


def test_find_other_regions_ignores_non_places():
    assert find_other_regions("연합군이 진격했고 대도시의 신도시는") == set()
    assert find_other_regions("청주시는 인센티브를 유지") == {"청주"}


@pytest.mark.parametrize(
    "doc, reason",
    [
        (_news(text="짧은 본문"), ExclusionReason.TOO_SHORT),
        (_news(text="소상공인 " * 3000), ExclusionReason.TOO_LONG),
        (_news(title="[포토] 소상공인 폐업 급증"), ExclusionReason.LOW_INFORMATION),
        (_news(title="[게시판] 롯데호텔, 전통시장 활성화"), ExclusionReason.LOW_INFORMATION),
        (_news(title="[오늘의 주요일정] 사회(11월 4일)"), ExclusionReason.LOW_INFORMATION),
        (_news(title="[민주당 당대표 토론회 전문] 내가 최고"), ExclusionReason.TRANSCRIPT),
        (_news(title="정부 예산안 통과", text="국회는 내년도 예산안을 처리했다. " * 20), ExclusionReason.OFF_TOPIC),
        (_news(title="청주시는 소상공인 인센티브 유지"), ExclusionReason.OUTSIDE_SEOUL),
        (_news(published_at="2020-05-01"), ExclusionReason.OUT_OF_PERIOD),
    ],
)
def test_assess_exclusions(doc, reason):
    decision = assess(doc, _config())
    assert decision.excluded and decision.reason == reason


@pytest.mark.parametrize("title", ["[인사이드 스토리] 소상공인 매출", "[전문가 진단] 소상공인 대책"])
def test_assess_does_not_flag_series_names(title):
    assert not assess(_news(title=title), _config()).excluded


def test_assess_keeps_national_and_seoul_with_scope():
    national = assess(_news(), _config())
    seoul = assess(_news(title="서울 망원시장 소상공인 지원"), _config())
    assert (national.excluded, national.scope) == (False, Scope.NATIONAL)
    assert (seoul.excluded, seoul.scope) == (False, Scope.SEOUL)


def test_national_period_applies_only_to_national_scope():
    config = _config(national_period_start=date(2025, 1, 1))
    old_seoul = _news(title="서울 망원시장 소상공인 지원", published_at="2023-03-01")
    old_national = _news(published_at="2023-03-01")
    assert not assess(old_seoul, config).excluded
    assert assess(old_national, config).reason == ExclusionReason.OUT_OF_PERIOD


def test_flags_for_missing_source_and_date():
    decision = assess(_news(published_at=None, extra={"source_missing": True}), _config())
    assert not decision.excluded
    assert decision.flags == {QualityFlag.MISSING_SOURCE, QualityFlag.MISSING_DATE}


def test_non_news_types_skip_news_rules():
    press = QualityInput(doc_id="kr_1", doc_type=DocType.PRESS_RELEASE, title="청주시는 보도자료",
                         text="본문 " * 10, source="중소벤처기업부", published_at="2019-01-01")
    assert not assess(press, _config()).excluded
    blog = QualityInput(doc_id="b_1", doc_type=DocType.BLOG, title="후기", text="소정의 원고료를 받아 작성",
                        source="블로그", published_at="2026-01-01")
    config = _config(promo_markers={DocType.BLOG: ("소정의 원고료",)})
    assert assess(blog, config).reason == ExclusionReason.PROMOTIONAL
