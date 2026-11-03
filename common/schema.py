"""문서·청크 메타데이터 스키마.

가이드 2-1 최소 요건: 청크에 출처, 제목, URL, 발행일, 문서 종류 정보가 포함되어야 한다.
가이드 7 점검 목록: 모든 문서의 출처 URL, 발행일, 수집 일시가 기록되어야 한다.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum
from typing import Any


class DocType(StrEnum):
    """문서 종류 (주제 A: 지역 상권·소상공인 정책)."""

    NEWS = "news"                         # 제공 데이터
    PRESS_RELEASE = "press_release"       # 중기부·소진공·금융위 등 보도자료
    ANNOUNCEMENT = "announcement"         # 서울시·자치구 상권 지원사업 공고
    POLICY_REPORT = "policy_report"       # 국책·지자체 연구기관 보고서
    LAW = "law"                           # 핵심 법령 현행 조문 (국가법령정보센터)
    BLOG = "blog"                         # 소상공인 체감 반응


class SourceTier(StrEnum):
    """출처 신뢰도 구분 (가이드 4-6: 출처 종류를 함께 명시)."""

    PRIMARY = "1차 자료"
    PRESS = "언론 보도"
    USER_GENERATED = "현장 반응"


SOURCE_TIER: dict[DocType, SourceTier] = {
    DocType.NEWS: SourceTier.PRESS,
    DocType.PRESS_RELEASE: SourceTier.PRIMARY,
    DocType.ANNOUNCEMENT: SourceTier.PRIMARY,
    DocType.POLICY_REPORT: SourceTier.PRIMARY,
    DocType.LAW: SourceTier.PRIMARY,
    DocType.BLOG: SourceTier.USER_GENERATED,
}


class CollectMethod(StrEnum):
    PROVIDED = "provided"   # 제공 데이터 적재
    CRAWL = "crawl"         # 웹 페이지 크롤링
    API = "api"             # 검색·공개 API


# Chroma 청크 메타데이터 필드. 값은 str/int/float/bool 만 허용되며 None 은 저장할 수 없다.
# Chroma 의 $gte/$lte 필터는 숫자에만 동작하므로, 기간 필터용으로 published_ymd(예: 20260115)를
# 정수로 함께 저장한다. 발행일 미상은 0 으로 저장하고 기간 필터 적용 시 제외 여부를 명시적으로 정한다.
CHUNK_METADATA_FIELDS: tuple[str, ...] = (
    "chunk_id",
    "chunk_index",
    "doc_id",
    "doc_type",
    "source",
    "title",
    "url",
    "published_at",
    "published_ymd",
    "collected_at",
    "source_tier",   # SourceTier 값 — 프롬프트·UI 에서 출처 신뢰도 구분
    "scope",         # 뉴스 범위 (seoul/national), 해당 없으면 빈 문자열 — UI 필터
)


def _validate_iso(value: str, field_name: str) -> None:
    try:
        if "T" in value or " " in value:
            datetime.fromisoformat(value)
        else:
            date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} 는 ISO 8601 형식이어야 합니다: {value!r}") from exc


@dataclass(frozen=True)
class DocumentRecord:
    """수집 계층이 docs.db 의 documents 테이블에 기록하는 단위."""

    doc_id: str
    doc_type: DocType
    source: str                    # 발행처 (매체명·부처명·블로그명 등)
    title: str
    url: str
    collected_at: str              # ISO 8601, 수집 일시
    collect_method: CollectMethod
    published_at: str | None = None  # ISO 8601, 미상이면 None
    keyword: str | None = None     # 수집 키워드 (제공 데이터·검색 API)
    raw_path: str | None = None    # 원본 파일 경로 (프로젝트 루트 기준 상대 경로)
    raw_text: str | None = None    # 수집 시점에 확보한 본문 (제공 뉴스 본문, 블로그 요약문 등)
    extra: dict[str, Any] = field(default_factory=dict)  # 소스별 부가 정보

    def __post_init__(self) -> None:
        for name in ("doc_id", "source", "title", "url"):
            if not getattr(self, name).strip():
                raise ValueError(f"DocumentRecord.{name} 가 비어 있습니다 (doc_id={self.doc_id!r})")
        # 문자열로 들어온 경우에도 허용 값인지 검증하며 Enum 으로 정규화한다.
        object.__setattr__(self, "doc_type", DocType(self.doc_type))
        object.__setattr__(self, "collect_method", CollectMethod(self.collect_method))
        _validate_iso(self.collected_at, "collected_at")
        if self.published_at is not None:
            _validate_iso(self.published_at, "published_at")


PROCESSED_STATUSES = ("kept", "duplicate", "excluded")


@dataclass(frozen=True)
class ProcessedRecord:
    """인덱서가 processed_documents 테이블에 기록하는 단위 (정제·품질 필터·중복 처리 결과)."""

    doc_id: str
    status: str                       # kept | duplicate | excluded
    processed_at: str                 # ISO 8601
    clean_text: str | None = None     # 제외 문서는 저장하지 않는다 (원문은 documents.raw_text)
    content_hash: str | None = None
    canonical_doc_id: str | None = None  # 대표 문서 id. 대표 문서는 자기 자신, 제외 문서는 None
    excluded_reason: str | None = None
    quality_flags: tuple[str, ...] = ()
    scope: str | None = None

    def __post_init__(self) -> None:
        if self.status not in PROCESSED_STATUSES:
            raise ValueError(f"알 수 없는 status: {self.status!r}")
        if (self.status == "excluded") != (self.excluded_reason is not None):
            raise ValueError(f"제외 문서에만 excluded_reason 이 있어야 합니다: {self.doc_id}")
        if self.status == "kept" and self.canonical_doc_id != self.doc_id:
            raise ValueError(f"대표 문서의 canonical_doc_id 는 자기 자신이어야 합니다: {self.doc_id}")
        if self.status == "duplicate" and self.canonical_doc_id in (None, self.doc_id):
            raise ValueError(f"중복 문서에는 다른 대표 문서 id 가 있어야 합니다: {self.doc_id}")
