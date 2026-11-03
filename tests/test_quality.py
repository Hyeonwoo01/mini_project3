from datetime import date

import pytest

from common.schema import DocType
from indexer.quality import (
    AMBIGUOUS_DISTRICTS,
    SEOUL_DISTRICTS,
    ExclusionReason,
    QualityConfig,
    QualityDecision,
    get_quality_config,
)


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


def _config(**overrides) -> QualityConfig:
    fields = dict(
        name="test",
        period_start=date(2021, 10, 16),
        min_chars={DocType.NEWS: 100},
        lead_chars=300,
        topic_terms=("소상공인",),
    )
    fields.update(overrides)
    return QualityConfig(**fields)


@pytest.mark.parametrize(
    "overrides",
    [{"min_chars": {DocType.NEWS: 0}}, {"lead_chars": 0}, {"topic_terms": ()}],
)
def test_config_validation(overrides):
    _config()
    with pytest.raises(ValueError):
        _config(**overrides)


def test_unknown_config_exits():
    with pytest.raises(SystemExit):
        get_quality_config("does-not-exist")
