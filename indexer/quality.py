"""문서 품질 필터 (indexer prepare 단계).

원칙 — "좋은 글만 남기기"가 아니라 문서를 세 부류로 나눠 다르게 처리한다.

| 부류                                        | 예                                   | 처리                          |
|---------------------------------------------|--------------------------------------|-------------------------------|
| 범위 밖                                     | 주제 무관, 서울 외 지자체 단독 사업, 기간 밖 | 제외                          |
| 품질 불량                                   | 기사형 광고, 과소·절단 본문, 정보량 부족 | 제외                          |
| 신뢰도는 낮지만 고유한 역할이 있는 문서      | 비판·분석 기사, 블로그 경험담         | 유지 + 플래그·출처 신뢰도 표시 |

재배포본·완전 중복은 이 모듈이 아니라 dedup.py 가 처리한다 (대표 문서 1건 유지).

제목 표지는 괄호가 닫히는 위치까지 맞춘다. '[인사이드 스토리]' 같은 연재명이 '[인사]'로
잡히는 오탐이 EDA에서 확인되었다.

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
- 플래그는 청크 메타데이터로도 전달해 프롬프트(출처 종류 명시)와 UI(신뢰도 표시)에 쓴다

문서 종류별 규칙 (임계값·판정어는 EDA 후 QualityConfig 로 확정)

| 종류          | 제외                                                    | 유지·플래그                                   |
|---------------|---------------------------------------------------------|-----------------------------------------------|
| news          | 본문 과소(사진·단신), 기사형 광고 표지, 제목·리드에      | 비판·분석 기사는 유지.                         |
|               | 주제어 없음, 서울 외 지자체 단독 사업, 기간 밖,          | 매체명 결손 → MISSING_SOURCE                   |
|               | 정기 편성 기사(국회 일정·신문 요약 → LOW_INFORMATION)    | 지역 언급이 없는 전국 정책 기사는 범위 안      |
| press_release | 행사 사진·일정만 있는 정보량 부족 문서                   | 1차 자료라 대부분 유지. 본문보다 첨부(HWP/PDF) |
|               |                                                         | 가 핵심인 경우가 많아 파싱 품질이 곧 품질       |
| announcement  | 정정·재공고로 대체된 원 공고 → SUPERSEDED                | 마감 공고는 비교·시점 질문용으로 유지 + EXPIRED |
| policy_report | 기간 밖, 주제 무관                                      | 수집 단계에서 10~20건 정밀 선별. 한 보고서가    |
|               |                                                         | 검색 결과를 독차지하지 않도록 요약·결론·정책제언 |
|               |                                                         | 장 위주로 인덱싱 (chunk.py)                    |
| law           | — (현행 조문만 수집하므로 필터 대상 아님)                | —                                             |
| blog          | 대가성 표시 문구, 본문 과소. 보도자료 복붙은 dedup 처리  | 기준은 공식성이 아니라 실제 경험.              |
|               |                                                         | 1인칭 경험 서술 → FIRST_PERSON                 |

검증과 측정
- 제외·유지 문서를 종류별 20~30건씩 직접 읽고 오판율을 기록한다 (리포트 2·4장 근거)
- QualityConfig.name 을 실험 태그로 써서 필터 전후 Hit@3·거절 정확도를 비교한다.
  중복 처리와는 별도 실험으로 하여 한 번에 하나만 바꾼다 (가이드 4-9).
  필터를 적용하지 않는 기준 실험은 config 없이 prepare 를 실행한다.
- 규칙 기반을 우선한다 (투명성·재현성·비용). 애매한 관련성 판정에 LLM 분류를 쓰는 경우
  샘플 검증용으로 한정한다 (공용 API 키 비용).
"""

from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from typing import Any

from common.schema import DocType

SEOUL_DISTRICTS: tuple[str, ...] = (
    "강남구", "강동구", "강북구", "강서구", "관악구", "광진구", "구로구", "금천구", "노원구",
    "도봉구", "동대문구", "동작구", "마포구", "서대문구", "서초구", "성동구", "성북구", "송파구",
    "양천구", "영등포구", "용산구", "은평구", "종로구", "중구", "중랑구",
)

# 부산·대구·인천 등에도 같은 이름의 구가 있어 단독으로는 서울로 판정하지 않는다.
AMBIGUOUS_DISTRICTS: frozenset[str] = frozenset({"중구", "강서구"})


class ExclusionReason(StrEnum):
    OFF_TOPIC = "off_topic"                # 제목·리드에 주제어 없음
    OUTSIDE_SEOUL = "outside_seoul"        # 서울 외 지자체 단독 사업
    OUT_OF_PERIOD = "out_of_period"        # 수집 기간(최근 5년) 밖
    TOO_SHORT = "too_short"                # 본문 과소 (사진·단신 기사 등)
    TRUNCATED = "truncated"                # 본문 절단·누락
    PROMOTIONAL = "promotional"            # 기사형 광고, 대가성 블로그
    LOW_INFORMATION = "low_information"    # 행사 사진·일정만 있는 보도자료 등
    SUPERSEDED = "superseded"              # 정정·재공고로 대체된 원 공고


class QualityFlag(StrEnum):
    MISSING_SOURCE = "missing_source"      # 매체명 결손 (제공 데이터 2026년 3월 이후 일부)
    MISSING_DATE = "missing_date"          # 발행일 미상 — 기간 필터에서의 처리를 명시해야 한다
    EXPIRED = "expired"                    # 마감된 공고
    FIRST_PERSON = "first_person"          # 블로그 1인칭 경험 서술 (긍정 신호)


@dataclass(frozen=True)
class QualityInput:
    doc_id: str
    doc_type: DocType
    title: str
    text: str                  # parse.clean_text() 를 거친 본문
    source: str
    published_at: str | None
    extra: dict[str, Any] = field(default_factory=dict)  # 공고 마감일 등 소스별 부가 정보


@dataclass(frozen=True)
class QualityDecision:
    excluded: bool
    reason: ExclusionReason | None = None
    flags: frozenset[QualityFlag] = frozenset()

    def __post_init__(self) -> None:
        if self.excluded != (self.reason is not None):
            raise ValueError("제외 판정에는 사유가 있어야 하고, 유지 판정에는 사유가 없어야 합니다")


@dataclass(frozen=True)
class QualityConfig:
    """EDA 결과로 정하는 임계값과 판정어. name 은 실험 태그로 쓴다."""

    name: str
    period_start: date
    min_chars: dict[DocType, int]            # 종류별 최소 본문 길이
    lead_chars: int                          # 관련성 판정에 쓰는 본문 앞부분 길이
    topic_terms: tuple[str, ...]             # 제목·리드 관련성 판정어
    promo_markers: dict[DocType, tuple[str, ...]] = field(default_factory=dict)  # 광고·대가성 표시

    def __post_init__(self) -> None:
        if self.lead_chars <= 0 or any(v <= 0 for v in self.min_chars.values()):
            raise ValueError(f"길이 기준은 양수여야 합니다: {self.name}")
        if not self.topic_terms:
            raise ValueError(f"topic_terms 가 비어 있습니다: {self.name}")


# EDA 후 확정한다.
QUALITY_CONFIGS: dict[str, QualityConfig] = {}


def get_quality_config(name: str) -> QualityConfig:
    try:
        return QUALITY_CONFIGS[name]
    except KeyError:
        available = ", ".join(QUALITY_CONFIGS) or "(없음)"
        raise SystemExit(f"알 수 없는 품질 설정: {name}. 사용 가능: {available}") from None


def is_outside_seoul_program(doc: QualityInput) -> bool:
    """서울 외 지자체가 주체인 지원사업 기사이면서 서울·전국 단위 정책 내용이 없으면 True.

    판정 시 주의할 오탐:
    - 통신사 기사 첫머리의 발신지 표기 '(서울=연합뉴스)' — 내용과 무관하므로 제거 후 판정한다
    - '서울'이 들어간 고유명사 (서울경제·서울신문 같은 매체명, 서울대, 서울우유 등)
    - AMBIGUOUS_DISTRICTS 의 중복 지명
    - 전국 정책을 지역 설명회 형태로 소개하는 기사 (예: 타 지역의 "중기부 지원정책 설명회 개최").
      정책 내용이 거의 없으면 OUTSIDE_SEOUL 보다 LOW_INFORMATION 으로 보는 편이 사유 기록에 정확하다.
    """
    raise NotImplementedError


def assess(doc: QualityInput, config: QualityConfig) -> QualityDecision:
    """문서 종류별 규칙을 적용해 첫 번째 제외 사유, 또는 유지 판정과 플래그를 반환한다."""
    raise NotImplementedError
