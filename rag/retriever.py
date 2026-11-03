"""검색 및 재정렬.

- Chroma 를 디스크에서 열어 조회만 한다 (인덱스 쓰기 금지)
- 컬렉션 메타데이터의 임베딩 모델이 현재 설정과 다르면 즉시 오류를 낸다
- 기간·출처 종류 필터를 Chroma where 조건으로 변환한다 (기간은 published_ymd 정수 비교)
- 유사도 임계값 미달 결과는 검색 실패로 처리할 수 있도록 점수를 함께 반환한다
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from common.schema import DocType


@dataclass(frozen=True)
class SearchFilters:
    doc_types: tuple[DocType, ...] = ()
    date_from: date | None = None
    date_to: date | None = None


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    text: str
    score: float  # 높을수록 유사. Chroma 거리 값을 유사도로 변환한 값
    metadata: dict[str, Any] = field(default_factory=dict)


def retrieve(
    query: str,
    *,
    k: int,
    filters: SearchFilters | None = None,
    collection: str | None = None,
) -> list[RetrievedChunk]:
    """collection 을 생략하면 설정의 CHROMA_COLLECTION 을 사용한다."""
    raise NotImplementedError
