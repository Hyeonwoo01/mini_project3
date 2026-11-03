"""문서 종류별 청킹 (가이드 4-4).

필수 수행 사항
- 청크 본문에 문서 제목을 포함한다
- 청크 길이 분포를 확인한다 (과소·과다 청크 점검)
- 둘 이상의 설정을 구성하고 평가셋 Hit@3 으로 비교하여 선택 근거를 기록한다

청크 크기 단위(문자 수/토큰 수)와 문서 종류별 설정값은 EDA 결과를 근거로 정한다.
법령(law)은 구조가 고정되어 있으므로 크기 기준이 아니라 조문 단위로 분할하고 조문 번호를 보존한다.
"""

from dataclasses import dataclass, field

from common.schema import DocType


@dataclass(frozen=True)
class ChunkConfig:
    name: str
    chunk_size: int
    chunk_overlap: int
    include_title: bool = True
    # 문서 종류별로 다른 (chunk_size, chunk_overlap) 을 적용할 때 사용한다
    overrides: dict[DocType, tuple[int, int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.chunk_size <= 0 or not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError(f"잘못된 청킹 설정: {self}")


# EDA 후 확정한다. 실험 이름이 컬렉션 이름(docs__<name>)과 eval/results 태그로 이어진다.
CHUNK_CONFIGS: dict[str, ChunkConfig] = {}


def get_chunk_config(name: str) -> ChunkConfig:
    try:
        return CHUNK_CONFIGS[name]
    except KeyError:
        available = ", ".join(CHUNK_CONFIGS) or "(없음)"
        raise SystemExit(f"알 수 없는 청킹 설정: {name}. 사용 가능: {available}") from None
