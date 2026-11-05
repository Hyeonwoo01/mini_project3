"""문서 종류별 청킹 (가이드 4-4).

필수 수행 사항
- 청크 본문에 문서 제목을 포함한다 (청크만으로는 출처 문서를 식별할 수 없다)
- 청크 길이 분포를 확인한다 (`python -m indexer build --dry-run` 이 분포를 출력한다)
- 둘 이상의 설정을 구성하고 평가셋 Hit@3 으로 비교하여 선택 근거를 기록한다

크기 단위는 문자 수다. 정제 뉴스 본문은 중앙값 약 950자, 하위 5% 360자, 상위 1% 5,300자이므로
(EDA 3번) 500자 설정은 기사 하나를 2~3개로, 1000자 설정은 대부분 1~2개로 나눈다.
chunk_size 는 제목을 제외한 본문 기준이다.
법령(law)은 수집 단계에서 조문 하나를 문서 하나로 저장하므로(collector/collect_law.py) 조문 단위로 나뉘고,
긴 조문만 이 설정으로 다시 나뉜다.
"""

from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from common.schema import SOURCE_TIER, DocType
from common.dates import annotate_relative_dates

# 문단 → 줄 → 문장 → 쉼표 → 공백 순으로 자를 곳을 찾는다. 문장부호는 앞 문장에 붙여 둔다.
SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", ", ", " ", ""]

# 마지막 조각이 이보다 짧으면 앞 청크에 합친다. 분할 끝에 남은 '기자.' 같은 9~99자 조각은
# 검색에 걸려도 근거가 되지 못한다 (c1000 미리보기: 100자 미만 71개, 최소 9자).
# 품질 필터의 최소 본문 길이(100자)와 맞춘다.
MIN_CHUNK_CHARS = 100


@dataclass(frozen=True)
class ChunkConfig:
    name: str
    chunk_size: int
    chunk_overlap: int
    include_title: bool = True
    # 상대 날짜('내년', '내일')에 발행일 기준 날짜를 덧붙인다 (common/dates.py). 메타데이터 title 은 원문 그대로
    normalize_dates: bool = False
    # 문서 종류별로 다른 (chunk_size, chunk_overlap) 을 적용할 때 사용한다
    overrides: dict[DocType, tuple[int, int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        sizes = [(self.chunk_size, self.chunk_overlap), *self.overrides.values()]
        if any(size <= 0 or not 0 <= overlap < size for size, overlap in sizes):
            raise ValueError(f"잘못된 청킹 설정: {self}")

    def sizes_for(self, doc_type: DocType) -> tuple[int, int]:
        return self.overrides.get(doc_type, (self.chunk_size, self.chunk_overlap))


# 실험 이름이 컬렉션 이름(docs__<name>)과 eval/results 태그로 이어진다.
CHUNK_CONFIGS: dict[str, ChunkConfig] = {
    "c500": ChunkConfig(name="c500", chunk_size=500, chunk_overlap=100),
    "c1000": ChunkConfig(name="c1000", chunk_size=1000, chunk_overlap=150),
    # 제목 포함 효과를 재는 비교 실험용 (가이드 4-4: 제목 포함 여부가 검색 정확도에 유의미한 차이)
    "c500-notitle": ChunkConfig(name="c500-notitle", chunk_size=500, chunk_overlap=100, include_title=False),
    # 상대 날짜 정규화 실험용 (t007·t011)
    "c500-dates": ChunkConfig(name="c500-dates", chunk_size=500, chunk_overlap=100, normalize_dates=True),
}


def get_chunk_config(name: str) -> ChunkConfig:
    try:
        return CHUNK_CONFIGS[name]
    except KeyError:
        available = ", ".join(CHUNK_CONFIGS) or "(없음)"
        raise SystemExit(f"알 수 없는 청킹 설정: {name}. 사용 가능: {available}") from None


@dataclass(frozen=True)
class ChunkSource:
    """청킹 대상 문서 (processed_documents 의 kept 문서 + documents 메타데이터)."""

    doc_id: str
    doc_type: DocType
    title: str
    text: str
    source: str
    url: str
    published_at: str | None
    collected_at: str
    scope: str | None = None


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str                     # 임베딩·검색에 쓰는 텍스트 (제목 포함 설정이면 제목 + 본문)
    body: str                     # 제목을 뺀 본문 조각 (길이 분포 확인용)
    metadata: dict[str, str | int]


def _merge_short_tail(bodies: list[str], min_chars: int = MIN_CHUNK_CHARS) -> list[str]:
    """마지막 조각이 짧으면 앞 조각에 붙인다. 청크 겹침으로 이미 들어 있는 부분은 다시 붙이지 않는다."""
    if len(bodies) < 2 or len(bodies[-1]) >= min_chars:
        return bodies
    previous, tail = bodies[-2], bodies[-1]
    if tail in previous:
        return bodies[:-1]
    # 5자 미만 일치는 우연일 수 있어 겹침으로 보지 않는다.
    overlap = next((n for n in range(min(len(previous), len(tail)), 4, -1) if previous.endswith(tail[:n])), 0)
    merged = previous + tail[overlap:] if overlap else f"{previous} {tail}"
    return [*bodies[:-2], merged]


def _published_ymd(published_at: str | None) -> int:
    """Chroma 의 범위 필터는 숫자에만 동작하므로 YYYYMMDD 정수로 저장한다. 미상은 0."""
    if not published_at or len(published_at) < 10:
        return 0
    digits = published_at[:10].replace("-", "")
    return int(digits) if digits.isdigit() else 0


def chunk_document(doc: ChunkSource, config: ChunkConfig) -> list[Chunk]:
    size, overlap = config.sizes_for(doc.doc_type)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap, separators=SEPARATORS, keep_separator="end",
        strip_whitespace=True,
    )
    title, body_text = doc.title, doc.text
    if config.normalize_dates:
        title, _ = annotate_relative_dates(title, doc.published_at)
        body_text, _ = annotate_relative_dates(body_text, doc.published_at)
    bodies = _merge_short_tail([b for b in splitter.split_text(body_text) if b.strip()])
    chunks = []
    for index, body in enumerate(bodies):
        chunk_id = f"{doc.doc_id}#{index:03d}"
        metadata: dict[str, str | int] = {
            "chunk_id": chunk_id,
            "chunk_index": index,
            "doc_id": doc.doc_id,
            "doc_type": doc.doc_type.value,
            "source": doc.source,
            "title": doc.title,
            "url": doc.url,
            "published_at": doc.published_at or "",   # Chroma 메타데이터는 None 을 저장할 수 없다
            "published_ymd": _published_ymd(doc.published_at),
            "collected_at": doc.collected_at,
            "source_tier": SOURCE_TIER[doc.doc_type].value,
            "scope": doc.scope or "",
        }
        text = f"{title}\n\n{body}" if config.include_title else body
        chunks.append(Chunk(chunk_id=chunk_id, doc_id=doc.doc_id, text=text, body=body, metadata=metadata))
    return chunks
