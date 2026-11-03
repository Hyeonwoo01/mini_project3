"""제공 데이터(뉴스 CSV) 적재.

data/provided/ 의 CSV 를 읽어 DocumentRecord(doc_type=news, collect_method=provided)로
docs.db 에 적재한다.

CSV 형식 (제공 데이터 README 기준, UTF-8 with BOM)
    제목, 본문전문, 원문URL, 발행일, 매체명, 문서종류, 수집키워드

적재 원칙
- 원본의 모든 유효 행을 적재한다. 범위·품질 필터와 중복 처리는 indexer 가 수행해야
  처리 전후 건수가 남는다.
- 같은 기사(같은 URL·발행일)가 여러 파일에 있으면 한 건으로 합치고 수집 키워드와 출처 파일을
  모두 기록한다. 합친 건수는 통계로 출력한다.
- 매체명이 비어 있으면(2026년 3월 이후 기사 일부) source 를 "매체 미상"으로 두고
  extra["source_missing"] 을 표시한다 → 인덱서에서 QualityFlag.MISSING_SOURCE.
- 수집 일시는 파일명의 추출 기간 끝 날짜로 기록한다 (예: ..._2021-10-16~2026-10-16.csv).
"""

import csv
import fnmatch
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from sqlite3 import Connection

from collector.storage import make_doc_id
from common.config import get_settings
from common.docstore import open_db, upsert_documents
from common.schema import CollectMethod, DocType, DocumentRecord

# 사용할 제공 파일. 주제 A 의 소상공인 묶음과, 지역사랑상품권을 대신하는 공공정책 그룹의 지역화폐.
PROVIDED_FILE_PATTERNS: tuple[str, ...] = (
    "01_소상공인_묶음_*.csv",
    "05_공공정책_키워드_지역화폐_*.csv",
)

REQUIRED_COLUMNS = ("제목", "본문전문", "원문URL", "발행일", "매체명", "문서종류", "수집키워드")
UNKNOWN_SOURCE = "매체 미상"
BATCH_SIZE = 1000

_PERIOD_IN_NAME = re.compile(r"_(\d{4}-\d{2}-\d{2})~(\d{4}-\d{2}-\d{2})\.csv$")

# 본문 길이가 기본 한도(131072자)를 넘는 행도 읽을 수 있게 한다. Windows 의 C long 범위를 넘지 않는 값.
csv.field_size_limit(2**31 - 1)


@dataclass
class LoadStats:
    rows_read: Counter[str] = field(default_factory=Counter)        # 파일별 읽은 행
    skipped: Counter[str] = field(default_factory=Counter)          # 사유별 건너뛴 행
    merged_across_files: int = 0                                    # 여러 파일에 있던 같은 기사
    duplicate_rows_in_file: int = 0                                 # 한 파일 안의 같은 기사 (같은 URL·발행일)
    source_missing: int = 0
    loaded: int = 0                                                 # 적재된 고유 문서 수

    def summary(self) -> str:
        lines = ["[제공 데이터 적재 결과]"]
        lines += [f"  읽은 행  {name}: {n:,}" for name, n in self.rows_read.items()]
        lines += [f"  건너뜀  {reason}: {n:,}" for reason, n in self.skipped.items()]
        lines += [
            f"  파일 간 같은 기사 병합: {self.merged_across_files:,}",
            f"  파일 내 같은 기사 병합: {self.duplicate_rows_in_file:,}",
            f"  매체명 결손: {self.source_missing:,}",
            f"  적재된 고유 문서: {self.loaded:,}",
        ]
        return "\n".join(lines)


def find_provided_files(directory: Path, patterns: Iterable[str] = PROVIDED_FILE_PATTERNS) -> list[Path]:
    """패턴에 맞는 파일을 찾는다. macOS 의 한글 파일명(NFD)도 맞도록 NFC 로 정규화해 비교한다."""
    patterns = [unicodedata.normalize("NFC", p) for p in patterns]
    found = [
        path
        for path in sorted(directory.glob("*.csv"))
        if any(fnmatch.fnmatch(unicodedata.normalize("NFC", path.name), p) for p in patterns)
    ]
    if not found:
        raise SystemExit(
            f"{directory} 에서 제공 데이터를 찾지 못했습니다. 다음 파일을 원래 이름 그대로 넣으세요: "
            + ", ".join(patterns)
        )
    return found


def extraction_date(path: Path) -> str:
    """파일명의 추출 기간 끝 날짜를 수집 일시로 쓴다."""
    match = _PERIOD_IN_NAME.search(unicodedata.normalize("NFC", path.name))
    if match is None:
        raise ValueError(f"파일명에서 추출 기간을 찾지 못했습니다 (원래 이름을 유지하세요): {path.name}")
    return date.fromisoformat(match.group(2)).isoformat()


def _normalize_published_at(raw: str) -> str | None:
    raw = raw.strip()
    if not raw:
        return None
    try:
        datetime.fromisoformat(raw)
    except ValueError:
        return None
    return raw


def _iter_rows(path: Path) -> Iterator[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path.name}: 필수 컬럼 누락 {missing}")
        yield from reader


@dataclass
class _Seen:
    keywords: set[str]
    files: set[str]
    collected_at: str


def _to_record(row: dict[str, str], seen: _Seen) -> DocumentRecord:
    source = row["매체명"].strip()
    published_at = _normalize_published_at(row["발행일"])
    url = row["원문URL"].strip()
    return DocumentRecord(
        doc_id=make_doc_id("news", url, published_at),
        doc_type=DocType.NEWS,
        source=source or UNKNOWN_SOURCE,
        title=row["제목"].strip(),
        url=url,
        collected_at=seen.collected_at,
        collect_method=CollectMethod.PROVIDED,
        published_at=published_at,
        keyword="; ".join(sorted(seen.keywords)) or None,
        raw_text=row["본문전문"].strip(),
        extra={"provided_files": sorted(seen.files), "source_missing": not source},
    )


def _skip_reason(row: dict[str, str]) -> str | None:
    if not row["제목"].strip() or not row["원문URL"].strip():
        return "제목 또는 URL 없음"
    if row["문서종류"].strip() != DocType.NEWS.value:
        return f"문서종류가 news 아님 ({row['문서종류'].strip()!r})"
    if row["발행일"].strip() and _normalize_published_at(row["발행일"]) is None:
        return "발행일 형식 오류"
    return None


def load_files(paths: Iterable[Path], conn: Connection) -> LoadStats:
    """CSV 를 스트리밍으로 읽어 배치 단위로 적재한다. 메모리에는 doc_id 별 키워드·파일 목록만 둔다."""
    stats = LoadStats()
    seen: dict[str, _Seen] = {}
    pending: list[DocumentRecord] = []
    # 같은 기사가 다시 나와 키워드·출처 파일 목록이 늘어난 문서 (적재 후 최종 값으로 갱신)
    merged_rows: dict[str, dict[str, str]] = {}

    for path in paths:
        name = unicodedata.normalize("NFC", path.name)
        collected_at = extraction_date(path)
        for row in _iter_rows(path):
            stats.rows_read[name] += 1
            reason = _skip_reason(row)
            if reason:
                stats.skipped[reason] += 1
                continue

            doc_id = make_doc_id("news", row["원문URL"], _normalize_published_at(row["발행일"]))
            keyword = row["수집키워드"].strip()
            entry = seen.get(doc_id)
            if entry is not None:
                if name in entry.files:
                    stats.duplicate_rows_in_file += 1
                else:
                    stats.merged_across_files += 1
                if (keyword and keyword not in entry.keywords) or name not in entry.files:
                    merged_rows[doc_id] = row
                if keyword:
                    entry.keywords.add(keyword)
                entry.files.add(name)
                continue

            entry = _Seen(keywords={keyword} if keyword else set(), files={name}, collected_at=collected_at)
            seen[doc_id] = entry
            pending.append(_to_record(row, entry))
            if not row["매체명"].strip():
                stats.source_missing += 1
            if len(pending) >= BATCH_SIZE:
                upsert_documents(conn, pending)
                pending.clear()

    upsert_documents(conn, pending)
    upsert_documents(conn, (_to_record(row, seen[doc_id]) for doc_id, row in merged_rows.items()))
    stats.loaded = len(seen)
    return stats


def run(files: list[str] | None = None) -> LoadStats:
    settings = get_settings()
    paths = [Path(f) for f in files] if files else find_provided_files(settings.provided_dir)
    for path in paths:
        if not path.is_file():
            raise SystemExit(f"파일이 없습니다: {path}")
    with open_db() as conn:
        stats = load_files(paths, conn)
    print(stats.summary())
    return stats
