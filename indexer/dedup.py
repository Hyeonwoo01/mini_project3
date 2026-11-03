"""중복 내용 처리 (가이드 4-4).

| 구분                | 판별 기준                        | 처리                                   |
|---------------------|----------------------------------|----------------------------------------|
| 완전 중복           | 정규화한 본문이 같음 (날짜 무관)        | 대표 1건만 유지                         |
| 재배포본            | 본문 유사도 ≥ similarity_threshold     | 대표 1건만 유지                         |
| 같은 보도자료 기사  | 본문 ≥ body_floor 이면서 제목도 유사    | 대표 1건만 유지 (title_threshold 설정 시) |
| 동일 사건 유사 기사 | 위 조건 미달 (제목이 다른 분석·해설 등) | 관점이 다를 수 있으므로 유지 (상충 근거) |

EDA 12번: 매체마다 같은 보도자료를 다시 쓴 기사는 본문 유사도가 0.35~0.55 에 머문다
(예: '상생페이백 11월분 3916억원 지급' 두 매체 0.35). 같은 구간에 관점이 다른 분석 기사도
섞여 있어 본문 임계값만 낮추면 둘 다 지워진다. 제목이 같은 사안을 가리키는지를 함께 본다.

유사도: 공백·기호를 뗀 본문의 문자 5-gram(shingle) 집합 간 Jaccard 유사도.
- 메모리를 줄이려고 해시값이 sample_mod 로 나누어떨어지는 shingle 만 쓴다. 해시는 crc32 라
  실행할 때마다 같은 결과가 나온다 (Python 기본 hash 는 실행마다 바뀐다).
- 재배포는 수일 안에 일어나므로 발행일 차이가 max_day_gap 이내인 쌍만 비교한다.
  매달 반복되는 정기 기사끼리 묶이는 것도 이 조건으로 막는다.
- 묶음은 연결 요소(union-find)로 만든다. A~B, B~C 이면 A·C 도 한 묶음이 되므로 임계값을
  낮게 잡으면 사건 연재 기사가 사슬처럼 묶일 수 있다. 임계값은 EDA 12번 표본을 읽고 정한다.

대표 문서: 출처 신뢰도가 높은 쪽(보도자료 원문 > 이를 옮긴 뉴스) → 본문이 긴 쪽 →
먼저 발행된 쪽 → doc_id 순.
"""

import hashlib
import re
import zlib
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime

from common.schema import SOURCE_TIER, DocType, SourceTier

_NON_WORD = re.compile(r"[\W_]+")
_TIER_RANK = {SourceTier.PRIMARY: 0, SourceTier.PRESS: 1, SourceTier.USER_GENERATED: 2}


@dataclass(frozen=True)
class DedupConfig:
    name: str
    similarity_threshold: float   # 이 이상이면 재배포본으로 묶는다
    max_day_gap: int = 3          # 발행일 차이가 이 이내인 쌍만 비교
    shingle_size: int = 5
    sample_mod: int = 4
    # 같은 보도자료 기사: 본문 유사도 ≥ body_floor 이면서 제목 유사도 ≥ title_threshold. None 이면 끈다.
    title_threshold: float | None = None
    body_floor: float = 0.3

    def __post_init__(self) -> None:
        if not 0 < self.similarity_threshold <= 1:
            raise ValueError(f"similarity_threshold 는 (0, 1] 범위여야 합니다: {self.name}")
        if self.title_threshold is not None and not (
            0 < self.title_threshold <= 1 and 0 < self.body_floor <= self.similarity_threshold
        ):
            raise ValueError(f"title_threshold·body_floor 설정이 잘못되었습니다: {self.name}")
        if self.max_day_gap < 0 or self.shingle_size < 2 or self.sample_mod < 1:
            raise ValueError(f"잘못된 중복 처리 설정: {self}")


DEDUP_CONFIGS: dict[str, DedupConfig] = {
    # 본문이 거의 같은 재배포본만 묶는다 (EDA 12번: 0.8 이상 925건 제거)
    "v1": DedupConfig(name="v1", similarity_threshold=0.8),
    # 같은 보도자료를 다시 쓴 기사까지 묶는다. title_threshold 는 EDA 13번 표본으로 확정한다.
    "v2": DedupConfig(name="v2", similarity_threshold=0.8, title_threshold=0.5, body_floor=0.3),
}


def get_dedup_config(name: str) -> DedupConfig:
    try:
        return DEDUP_CONFIGS[name]
    except KeyError:
        available = ", ".join(DEDUP_CONFIGS) or "(없음)"
        raise SystemExit(f"알 수 없는 중복 처리 설정: {name}. 사용 가능: {available}") from None


@dataclass(frozen=True)
class DedupInput:
    doc_id: str
    doc_type: DocType
    text: str
    published_at: str | None
    title: str = ""


@dataclass(frozen=True)
class DedupResult:
    doc_id: str
    canonical_doc_id: str
    similarity: float             # 대표 문서와의 유사도 (대표 문서 자신은 1.0)

    @property
    def is_duplicate(self) -> bool:
        return self.doc_id != self.canonical_doc_id


def normalize(text: str) -> str:
    return _NON_WORD.sub("", text)


def content_hash(text: str) -> str:
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def shingles(text: str, size: int = 5, sample_mod: int = 4) -> frozenset[int]:
    norm = normalize(text)
    hashes = (zlib.crc32(norm[i:i + size].encode("utf-8")) for i in range(len(norm) - size + 1))
    return frozenset(h for h in hashes if h % sample_mod == 0)


_TITLE_TAG = re.compile(r"[\[【(<][^\]】)>]{0,15}[\]】)>]")


def title_similarity(a: str, b: str) -> float:
    """제목의 문자 2-gram Jaccard. [단독]·[포토] 같은 머리말과 공백·기호는 뗀다."""
    def grams(title: str) -> set[str]:
        norm = normalize(_TITLE_TAG.sub("", title))
        return {norm[i:i + 2] for i in range(len(norm) - 1)}

    ga, gb = grams(a), grams(b)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def jaccard(a: frozenset[int], b: frozenset[int]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).date()
    except ValueError:
        return None


class _UnionFind:
    def __init__(self, items: Iterable[str]) -> None:
        self._parent = {item: item for item in items}

    def find(self, item: str) -> str:
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[item] != root:  # 경로 압축
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, a: str, b: str) -> None:
        self._parent[self.find(a)] = self.find(b)

    def groups(self) -> list[list[str]]:
        members: dict[str, list[str]] = defaultdict(list)
        for item in self._parent:
            members[self.find(item)].append(item)
        return list(members.values())


def _shingle_sets(docs: Sequence[DedupInput], config: DedupConfig) -> dict[str, frozenset[int]]:
    return {d.doc_id: shingles(d.text, config.shingle_size, config.sample_mod) for d in docs}


def _pairs(
    docs: Sequence[DedupInput], sets: dict[str, frozenset[int]], max_day_gap: int, floor: float
) -> list[tuple[str, str, float]]:
    dated = sorted(
        ((d.doc_id, day) for d in docs if (day := _parse_date(d.published_at)) is not None),
        key=lambda item: item[1],
    )
    found = []
    for i, (a, day_a) in enumerate(dated):
        for b, day_b in dated[i + 1:]:
            if (day_b - day_a).days > max_day_gap:
                break
            similarity = jaccard(sets[a], sets[b])
            if similarity >= floor:
                found.append((a, b, similarity))
    return found


def _is_duplicate_pair(a: DedupInput, b: DedupInput, body_similarity: float, config: DedupConfig) -> bool:
    if body_similarity >= config.similarity_threshold:
        return True
    return (
        config.title_threshold is not None
        and body_similarity >= config.body_floor
        and title_similarity(a.title, b.title) >= config.title_threshold
    )


def similar_pairs(
    docs: Sequence[DedupInput], config: DedupConfig, min_similarity: float | None = None
) -> list[tuple[str, str, float]]:
    """발행일이 근접한 문서 쌍 중 유사도가 기준 이상인 쌍. EDA 에서 임계값을 정할 때 쓴다."""
    floor = config.similarity_threshold if min_similarity is None else min_similarity
    return _pairs(docs, _shingle_sets(docs, config), config.max_day_gap, floor)


def count_duplicates(doc_ids: Iterable[str], pairs: Iterable[tuple[str, str, float]], threshold: float) -> int:
    """주어진 유사 쌍으로 묶었을 때 대표 문서 외에 제거되는 문서 수."""
    union_find = _UnionFind(doc_ids)
    for a, b, similarity in pairs:
        if similarity >= threshold:
            union_find.union(a, b)
    return sum(len(group) - 1 for group in union_find.groups())


def _rank(doc: DedupInput) -> tuple:
    return (_TIER_RANK[SOURCE_TIER[doc.doc_type]], -len(doc.text), doc.published_at or "9999-12-31", doc.doc_id)


def deduplicate(docs: Sequence[DedupInput], config: DedupConfig) -> dict[str, DedupResult]:
    """모든 문서에 대해 대표 문서와 유사도를 반환한다."""
    by_id = {d.doc_id: d for d in docs}
    sets = _shingle_sets(docs, config)
    union_find = _UnionFind(by_id)

    by_hash: dict[str, list[str]] = defaultdict(list)
    for d in docs:
        if normalize(d.text):  # 빈 본문끼리 묶지 않는다
            by_hash[content_hash(d.text)].append(d.doc_id)
    for same in by_hash.values():
        for other in same[1:]:
            union_find.union(same[0], other)

    floor = config.similarity_threshold if config.title_threshold is None else config.body_floor
    for a, b, similarity in _pairs(docs, sets, config.max_day_gap, floor):
        if _is_duplicate_pair(by_id[a], by_id[b], similarity, config):
            union_find.union(a, b)

    results: dict[str, DedupResult] = {}
    for group in union_find.groups():
        canonical = min(group, key=lambda doc_id: _rank(by_id[doc_id]))
        for doc_id in group:
            if doc_id == canonical or content_hash(by_id[doc_id].text) == content_hash(by_id[canonical].text):
                similarity = 1.0
            else:
                similarity = jaccard(sets[doc_id], sets[canonical])
            results[doc_id] = DedupResult(doc_id, canonical, similarity)
    return results
