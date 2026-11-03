"""문서 품질 필터 (indexer prepare 단계).

원칙 — "좋은 글만 남기기"가 아니라 문서를 세 부류로 나눠 다르게 처리한다.

| 부류                                        | 예                                   | 처리                          |
|---------------------------------------------|--------------------------------------|-------------------------------|
| 범위 밖                                     | 주제 무관, 서울 외 지자체 단독 사업, 기간 밖 | 제외                          |
| 품질 불량                                   | 기사형 광고, 과소·절단 본문, 정보량 부족 | 제외                          |
| 신뢰도는 낮지만 고유한 역할이 있는 문서      | 비판·분석 기사, 블로그 경험담         | 유지 + 플래그·출처 신뢰도 표시 |

재배포본·완전 중복은 이 모듈이 아니라 dedup.py 가 처리한다 (대표 문서 1건 유지).

과도한 필터의 위험
- 정답 문서가 사라지면 되살릴 수 없어 답변 불가가 늘어난다. 필터는 검색 순위 문제를 고치지,
  데이터 부재를 고치지 않는다.
- 관점이 다른 보도까지 지우면 상충 근거 제시가 불가능해진다 (가이드 4-4).
- 평가셋 문항을 보며 규칙을 고치면 과적합된다. 규칙과 임계값은 EDA 단계에서 정하고,
  이후 변경하면 사유를 기록한다.

처리 순서 (python -m indexer prepare)
    parse.extract_text → parse.clean_text(보일러플레이트 제거) → quality.assess → dedup
    → processed_documents 기록
- 제외: status='excluded', excluded_reason=ExclusionReason 값 → 사유별 처리 전후 건수가 남는다
- 유지: status='kept' (중복이면 dedup 이 'duplicate'), quality_flags 에 QualityFlag 목록
- 범위(Scope)와 플래그는 청크 메타데이터로 전달해 필터 UI·프롬프트(출처 종류 명시)에 쓴다

뉴스 판정 순서 — 앞에서 걸린 사유 하나만 기록한다 (EDA 11번 단계와 같은 순서)
    본문 길이(TOO_SHORT/TOO_LONG) → 제목 표지 → 주제 관련성(OFF_TOPIC)
    → 범위(OUTSIDE_SEOUL) → 기간(OUT_OF_PERIOD)

EDA 근거 (notebooks/eda.ipynb, 제공 뉴스 76,344건)
- 100~300자 단신에도 할인율·예산 같은 사실 정보가 있어 최소 길이를 100자로 낮게 잡는다 (3번)
- 최장 문서는 대부분 토론·회견 전문이고, 5,000~10,000자 구간에는 정상 장문 기사가 있다 (3번)
- 제목 표지는 괄호가 닫히는 위치까지 맞춘다. '[인사이드 스토리]'가 '[인사]'로,
  '[전문가 진단]'이 '[전문]'으로 잡히는 오탐을 막기 위해서다 (4번)
- 통신사 발신지 표기는 제공 데이터에서 이미 제거되어 있었지만, 직접 수집 기사에 대비해 제거한다 (5번)
- 광역 지자체명만 보면 '하남시는', '고양특례시가' 같은 기초 지자체 기사를 놓친다 (6번)
- 제목·리드에 주제어가 없는 기사는 표본 10건 중 8건이 주제 무관이었다 (7번)

문서 종류별 규칙
| 종류          | 제외                                                    | 유지·플래그                                   |
|---------------|---------------------------------------------------------|-----------------------------------------------|
| news          | 위 판정 순서 전부                                        | 지역 언급이 없는 전국 단위 기사는 범위 안.     |
|               |                                                         | 매체명 결손 → MISSING_SOURCE                   |
| press_release | 본문 과소 (행사 사진·일정만 있는 문서)                   | 1차 자료라 대부분 유지. 본문보다 첨부(HWP/PDF) |
|               |                                                         | 가 핵심인 경우가 많아 파싱 품질이 곧 품질       |
| announcement  | 정정·재공고로 대체된 원 공고 → SUPERSEDED (dedup 단계)    | 마감 공고는 비교·시점 질문용으로 유지 + EXPIRED |
| policy_report | 수집 단계에서 10~20건 정밀 선별                          | 요약·결론·정책제언 장 위주로 인덱싱 (chunk.py)  |
| law           | — (현행 조문만 수집하므로 필터 대상 아님)                | —                                             |
| blog          | 대가성 표시 문구(promo_markers), 본문 과소               | 기준은 공식성이 아니라 실제 경험               |

검증과 측정
- 제외·유지 문서를 종류별 20~30건씩 직접 읽고 오판율을 기록한다 (리포트 2·4장 근거)
- QualityConfig.name 을 실험 태그로 써서 필터 전후 Hit@3·거절 정확도를 비교한다.
  중복 처리와는 별도 실험으로 하여 한 번에 하나만 바꾼다 (가이드 4-9).
- 규칙 기반을 우선한다 (투명성·재현성·비용).
"""

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum
from functools import lru_cache
from typing import Any

from common.schema import DocType

SEOUL_DISTRICTS: tuple[str, ...] = (
    "강남구", "강동구", "강북구", "강서구", "관악구", "광진구", "구로구", "금천구", "노원구",
    "도봉구", "동대문구", "동작구", "마포구", "서대문구", "서초구", "성동구", "성북구", "송파구",
    "양천구", "영등포구", "용산구", "은평구", "종로구", "중구", "중랑구",
)

# 부산·대구·인천 등에도 같은 이름의 구가 있어 단독으로는 서울로 판정하지 않는다.
AMBIGUOUS_DISTRICTS: frozenset[str] = frozenset({"중구", "강서구"})

# '서울'이 들어간 매체명·기관명은 서울 관련 근거가 아니다.
SEOUL_PATTERN = "서울(?!경제|신문|대학|대|우유|보증)|" + "|".join(
    d for d in SEOUL_DISTRICTS if d not in AMBIGUOUS_DISTRICTS
)

# '경기'는 '경기 침체'와, '대전'은 '디지털 대전환'과 섞이므로 패턴을 좁힌다.
PROVINCE_PATTERNS: dict[str, str] = {
    "부산": "부산", "대구": "대구", "인천": "인천", "광주": "광주", "대전": "대전(?!환)",
    "울산": "울산", "세종": "세종시|세종특별자치시", "경기": "경기도", "강원": "강원",
    "충북": "충북|충청북도", "충남": "충남|충청남도", "전북": "전북|전라북도", "전남": "전남|전라남도",
    "경북": "경북|경상북도", "경남": "경남|경상남도", "제주": "제주",
}

# 'OO시는', 'OO군이', 'OO특례시가'처럼 기초 지자체가 주어로 나오는 형태 (근사 탐지)
CITY_PATTERN = (
    r"(?<![가-힣])([가-힣]{2,3})(?:특례시|시|군)"
    r"(?=는|은|가|이|와|과|의|에서|청|의회|민|장은|장이|수는|수가)"
)
NOT_PLACE: frozenset[str] = frozenset(
    {"서울", "대도", "신도", "소도", "중소도", "광역", "특별", "자치", "전통", "재래", "연합", "행정"}
)

# 통신사 기사 첫머리의 발신지 표기. 예: (서울=연합뉴스), [대구=뉴시스]
# 제공 데이터에는 '=' 가 빠진 (서울 뉴스1), (충주 연합뉴스) 형태로 들어 있다. '=' 가 없으면 '(서울 강남구)' 같은
# 본문 괄호와 구분되지 않으므로, 지명 뒤가 통신사명일 때만 발신지로 본다.
WIRE_AGENCIES: tuple[str, ...] = ("연합뉴스", "뉴스1", "뉴시스")
DATELINE_PATTERN = (
    r"[(\[]\s*[가-힣]{2,4}\s*(?:=\s*[^)\]]{1,20}|\s+(?:" + "|".join(WIRE_AGENCIES) + r")[^)\]]{0,10})[)\]]"
)

DEFAULT_TOPIC_TERMS: tuple[str, ...] = (
    "소상공인", "자영업", "상권", "골목", "전통시장", "재래시장", "상점가", "상인", "가맹점", "점포",
    "상가", "지역화폐", "지역사랑", "상품권", "온누리", "폐업", "임대료", "공실",
)


class ExclusionReason(StrEnum):
    OFF_TOPIC = "off_topic"                # 제목·리드에 주제어 없음
    OUTSIDE_SEOUL = "outside_seoul"        # 서울 외 지역만 언급 (지자체 단독 사업 등)
    OUT_OF_PERIOD = "out_of_period"        # 기간 규칙 밖
    TOO_SHORT = "too_short"                # 본문 과소 (사진 설명·빈 본문 등)
    TOO_LONG = "too_long"                  # 본문 과다 (토론·회견 전문 등, 청크 과다)
    TRANSCRIPT = "transcript"              # 제목에 전문·풀영상·녹취 표지
    TRUNCATED = "truncated"                # 본문 절단·누락
    PROMOTIONAL = "promotional"            # 기사형 광고, 대가성 블로그
    LOW_INFORMATION = "low_information"    # 사진·인사·게시판·정기 편성 기사 등
    SUPERSEDED = "superseded"              # 정정·재공고로 대체된 원 공고


class QualityFlag(StrEnum):
    MISSING_SOURCE = "missing_source"      # 매체명 결손
    MISSING_DATE = "missing_date"          # 발행일 미상 — 기간 규칙을 적용하지 않고 유지한다
    EXPIRED = "expired"                    # 마감된 공고
    FIRST_PERSON = "first_person"          # 블로그 1인칭 경험 서술 (긍정 신호)


class Scope(StrEnum):
    SEOUL = "seoul"                        # 서울 또는 서울 자치구 언급
    NATIONAL = "national"                  # 특정 지역 언급 없음 (전국 단위)
    OTHER_REGION = "other_region"          # 서울 외 지역만 언급


# 이름 → (패턴, 제외 사유). 괄호가 닫히는 위치까지 맞춰 연재명 오탐을 막는다.
TITLE_MARKERS: dict[str, tuple[str, ExclusionReason]] = {
    "포토·사진·영상": (
        r"^\s*[\[【(]\s*(?:포토|사진|화보|영상)(?:뉴스)?\s*[\]】)]", ExclusionReason.LOW_INFORMATION),
    "광고·협찬": (r"[\[【(]\s*(?:광고|AD|PR|협찬|후원)\s*[\]】)]", ExclusionReason.PROMOTIONAL),
    "인사·부고·동정·게시판": (
        r"^\s*[\[【(]\s*(?:인사|부고|동정|게시판)\s*[\]】)]", ExclusionReason.LOW_INFORMATION),
    "전문·풀영상·녹취": (
        r"[\[【(][^\]】)]*(?:전문|풀영상|녹취)(?![가-힣])[^\]】)]*[\]】)]", ExclusionReason.TRANSCRIPT),
    "정기 편성 (일정·신문 요약)": (
        r"오늘의\s*(?:국회\s*)?(?:주요\s*)?일정|주요\s*신문|톱뉴스|솎아보기|헤드라인|^\s*\[표\]",
        ExclusionReason.LOW_INFORMATION),
}


@dataclass(frozen=True)
class QualityInput:
    doc_id: str
    doc_type: DocType
    title: str
    text: str                  # parse.clean_text() 를 거친 본문
    source: str
    published_at: str | None
    extra: dict[str, Any] = field(default_factory=dict)  # 매체명 결손 표시, 공고 마감일 등


@dataclass(frozen=True)
class QualityDecision:
    excluded: bool
    reason: ExclusionReason | None = None
    flags: frozenset[QualityFlag] = frozenset()
    scope: Scope | None = None             # 뉴스만 판정한다

    def __post_init__(self) -> None:
        if self.excluded != (self.reason is not None):
            raise ValueError("제외 판정에는 사유가 있어야 하고, 유지 판정에는 사유가 없어야 합니다")


@dataclass(frozen=True)
class QualityConfig:
    """EDA 결과로 정하는 임계값과 판정어. name 은 실험 태그로 쓴다."""

    name: str
    period_start: date                       # 서울 기사 기간 시작
    min_chars: dict[DocType, int]            # 종류별 최소 본문 길이
    lead_chars: int                          # 관련성·범위 판정에 쓰는 본문 앞부분 길이
    topic_terms: tuple[str, ...]             # 제목·리드 관련성 판정어 (일반 문자열)
    max_chars: dict[DocType, int] = field(default_factory=dict)  # 종류별 최대 본문 길이
    # 전국 단위 기사 기간 시작. None 이면 period_start 와 같다.
    # 서울 상권의 장기 변화는 5년을 유지하고, 빠르게 바뀌는 전국 정책은 최신만 남길 때 쓴다.
    national_period_start: date | None = None
    promo_markers: dict[DocType, tuple[str, ...]] = field(default_factory=dict)  # 광고·대가성 표시

    def __post_init__(self) -> None:
        lengths = [*self.min_chars.values(), *self.max_chars.values(), self.lead_chars]
        if any(v <= 0 for v in lengths):
            raise ValueError(f"길이 기준은 양수여야 합니다: {self.name}")
        for doc_type, max_c in self.max_chars.items():
            if max_c <= self.min_chars.get(doc_type, 0):
                raise ValueError(f"max_chars 는 min_chars 보다 커야 합니다: {self.name} {doc_type}")
        if not self.topic_terms:
            raise ValueError(f"topic_terms 가 비어 있습니다: {self.name}")
        if self.national_period_start and self.national_period_start < self.period_start:
            raise ValueError(f"national_period_start 는 period_start 이후여야 합니다: {self.name}")


_V1_RULES = dict(
    # 보도자료는 PDF 추출 본문 기준. 200자 미만이면 추출 실패(스캔 이미지 등)나 사진 자료로 본다
    min_chars={DocType.NEWS: 100, DocType.PRESS_RELEASE: 200},
    max_chars={DocType.NEWS: 10_000},
    lead_chars=300,
    topic_terms=DEFAULT_TOPIC_TERMS,
)

QUALITY_CONFIGS: dict[str, QualityConfig] = {
    # EDA 확정값 (2026-11-03, 제공 뉴스 76,344건 → 재배포본 처리 전 18,552건).
    # 기간: 제공 기간 5년 전체. 시점·변화 질문과 시기별로 입장이 갈린 상충 근거
    # (예: 지역화폐 국비 축소와 확대)를 다루려면 이력이 필요하다.
    # 오래된 정보로 답하는 위험은 발행일 메타데이터·프롬프트의 최신 우선 규칙·기간 필터로 다루고,
    # 평가셋의 시점 문항으로 측정한다. 1차 자료가 뉴스에 묻히는 문제는 기간을 줄여도 비율이
    # 그대로이므로 검색 단계(문서 종류별 배분)에서 다룬다.
    "v1": QualityConfig(name="v1", period_start=date(2021, 10, 16), **_V1_RULES),
    # 비교 실험용: 수집 기간 끝 기준 최근 2년 (재배포본 처리 전 9,300건)
    "v1-2y": QualityConfig(name="v1-2y", period_start=date(2024, 10, 16), **_V1_RULES),
}


def get_quality_config(name: str) -> QualityConfig:
    try:
        return QUALITY_CONFIGS[name]
    except KeyError:
        available = ", ".join(QUALITY_CONFIGS) or "(없음)"
        raise SystemExit(f"알 수 없는 품질 설정: {name}. 사용 가능: {available}") from None


@lru_cache(maxsize=None)
def _compile(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern)


def _topic_pattern(terms: tuple[str, ...]) -> re.Pattern[str]:
    return _compile("|".join(map(re.escape, terms)))


def lead_text(title: str, text: str, lead_chars: int) -> str:
    """제목과 본문 앞부분. 발신지 표기는 내용과 무관하므로 제거한다."""
    return _compile(DATELINE_PATTERN).sub(" ", f"{title} {text[:lead_chars]}")


def find_other_regions(lead: str) -> set[str]:
    """서울 외 광역·기초 지자체명. 기초 지자체는 지자체가 주어로 나오는 형태만 잡는다."""
    found = {name for name, pattern in PROVINCE_PATTERNS.items() if _compile(pattern).search(lead)}
    found |= {name for name in _compile(CITY_PATTERN).findall(lead) if name not in NOT_PLACE}
    return found


def classify_scope(lead: str) -> Scope:
    """서울 언급이 있으면 다른 지역이 함께 나와도 SEOUL 로 본다."""
    if _compile(SEOUL_PATTERN).search(lead):
        return Scope.SEOUL
    return Scope.OTHER_REGION if find_other_regions(lead) else Scope.NATIONAL


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).date()
    except ValueError:
        return None


def _exclude(reason: ExclusionReason, flags: set[QualityFlag], scope: Scope | None = None) -> QualityDecision:
    return QualityDecision(excluded=True, reason=reason, flags=frozenset(flags), scope=scope)


def assess(doc: QualityInput, config: QualityConfig) -> QualityDecision:
    """규칙을 순서대로 적용해 첫 번째 제외 사유, 또는 유지 판정과 플래그를 반환한다."""
    flags: set[QualityFlag] = set()
    if doc.extra.get("source_missing"):
        flags.add(QualityFlag.MISSING_SOURCE)
    published = _parse_date(doc.published_at)
    if published is None:
        flags.add(QualityFlag.MISSING_DATE)

    length = len(doc.text)
    min_chars = config.min_chars.get(doc.doc_type)
    if min_chars is not None and length < min_chars:
        return _exclude(ExclusionReason.TOO_SHORT, flags)
    max_chars = config.max_chars.get(doc.doc_type)
    if max_chars is not None and length > max_chars:
        return _exclude(ExclusionReason.TOO_LONG, flags)

    if doc.doc_type is not DocType.NEWS:
        markers = config.promo_markers.get(doc.doc_type, ())
        if any(marker in doc.text for marker in markers):
            return _exclude(ExclusionReason.PROMOTIONAL, flags)
        return QualityDecision(excluded=False, flags=frozenset(flags))

    for pattern, reason in TITLE_MARKERS.values():
        if _compile(pattern).search(doc.title):
            return _exclude(reason, flags)

    lead = lead_text(doc.title, doc.text, config.lead_chars)
    if not _topic_pattern(config.topic_terms).search(lead):
        return _exclude(ExclusionReason.OFF_TOPIC, flags)

    scope = classify_scope(lead)
    if scope is Scope.OTHER_REGION:
        return _exclude(ExclusionReason.OUTSIDE_SEOUL, flags, scope)

    start = config.period_start
    if scope is Scope.NATIONAL and config.national_period_start:
        start = config.national_period_start
    if published is not None and published < start:
        return _exclude(ExclusionReason.OUT_OF_PERIOD, flags, scope)

    return QualityDecision(excluded=False, flags=frozenset(flags), scope=scope)
