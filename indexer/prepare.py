"""python -m indexer prepare — 정제 → 품질 필터 → 중복 처리 후 processed_documents 와 키워드 색인을 다시 만든다.

    python -m indexer prepare                     # 품질 v1 + 중복 처리 v2
    python -m indexer prepare --quality none      # 품질 필터 없이 (개선 전 기준 실험)

실행할 때마다 processed_documents 전체를 다시 계산한다. 설정 이름이 실험 태그가 되므로
결과 요약(사유별 건수)을 리포트의 처리 전후 건수로 쓴다.
키워드 색인(하이브리드 검색용)도 인덱싱 대상(kept) 문서로 함께 다시 만들어, 두 테이블이 항상 일치한다.
"""

import json
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from sqlite3 import Connection

from common.docstore import iter_documents, open_db, replace_keyword_index, replace_processed_documents
from common.schema import DocType, ProcessedRecord
from indexer import dedup, quality
from indexer.parse import ParseError, clean_text, extract_text


@dataclass
class PrepareStats:
    quality_config: str
    dedup_config: str
    total: int = 0
    statuses: Counter[str] = field(default_factory=Counter)
    excluded_reasons: Counter[str] = field(default_factory=Counter)
    kept_by_type: Counter[str] = field(default_factory=Counter)
    kept_by_scope: Counter[str] = field(default_factory=Counter)
    keyword_indexed: int = 0
    parse_errors: list[str] = field(default_factory=list)   # 원본 파일을 읽지 못한 문서 (본문 없음으로 처리)

    def summary(self) -> str:
        lines = [f"[prepare 결과] 품질={self.quality_config} 중복={self.dedup_config}",
                 f"  전체 문서: {self.total:,}"]
        if self.parse_errors:
            lines.append(f"  원본 파싱 실패: {len(self.parse_errors):,} (본문 없음으로 처리)")
            lines += [f"    {message}" for message in self.parse_errors[:5]]
        lines += [f"  제외 ({reason}): {n:,}" for reason, n in self.excluded_reasons.most_common()]
        lines += [f"  중복 (대표 문서로 대체): {self.statuses['duplicate']:,}",
                  f"  유지: {self.statuses['kept']:,}"]
        lines += [f"    문서 종류 {doc_type}: {n:,}" for doc_type, n in self.kept_by_type.most_common()]
        lines += [f"    범위 {scope}: {n:,}" for scope, n in self.kept_by_scope.most_common()]
        lines.append(f"  키워드 색인: {self.keyword_indexed:,}건")
        return "\n".join(lines)


def _document_text(row, errors: list[str]) -> str:
    if row["raw_text"]:
        return clean_text(row["raw_text"])
    if row["raw_path"]:
        try:
            return clean_text(extract_text(row["raw_path"]))
        except ParseError as exc:
            errors.append(f"{row['doc_id']}: {exc}")   # 한 파일 때문에 전체 처리가 멈추지 않게 한다
    return ""


PROGRESS_EVERY = 10_000


def prepare(
    conn: Connection,
    quality_config: quality.QualityConfig | None,
    dedup_config: dedup.DedupConfig,
    progress: Callable[[str], None] = lambda message: None,
) -> PrepareStats:
    started = time.monotonic()
    elapsed = lambda: f"{time.monotonic() - started:.0f}초"  # noqa: E731
    processed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    stats = PrepareStats(quality_config.name if quality_config else "none", dedup_config.name)
    records: list[ProcessedRecord] = []
    survivors: list[dedup.DedupInput] = []
    pending: dict[str, dict] = {}

    progress("[1/4] 정제·품질 판정 중")
    for row in iter_documents(conn):
        stats.total += 1
        if stats.total % PROGRESS_EVERY == 0:
            progress(f"      {stats.total:,}건 처리 ({elapsed()})")
        doc_type = DocType(row["doc_type"])
        text = _document_text(row, stats.parse_errors)
        if quality_config is None:
            decision = quality.QualityDecision(excluded=False)
        else:
            decision = quality.assess(
                quality.QualityInput(doc_id=row["doc_id"], doc_type=doc_type, title=row["title"], text=text,
                                     source=row["source"], published_at=row["published_at"],
                                     extra=json.loads(row["extra"])),
                quality_config,
            )
        common = dict(doc_id=row["doc_id"], processed_at=processed_at,
                      quality_flags=tuple(sorted(f.value for f in decision.flags)),
                      scope=decision.scope.value if decision.scope else None)
        if decision.excluded:
            records.append(ProcessedRecord(status="excluded", excluded_reason=decision.reason.value, **common))
            stats.excluded_reasons[decision.reason.value] += 1
            continue
        survivors.append(dedup.DedupInput(row["doc_id"], doc_type, text, row["published_at"], row["title"]))
        pending[row["doc_id"]] = dict(common, clean_text=text, content_hash=dedup.content_hash(text))

    progress(f"[2/4] 중복 처리 중: 대상 {len(survivors):,}건 ({elapsed()})")
    results = dedup.deduplicate(survivors, dedup_config)
    kept_docs: list[tuple[str, str, str]] = []
    for doc in survivors:
        result = results[doc.doc_id]
        status = "duplicate" if result.is_duplicate else "kept"
        record = ProcessedRecord(status=status, canonical_doc_id=result.canonical_doc_id, **pending[doc.doc_id])
        records.append(record)
        if status == "kept":
            stats.kept_by_type[doc.doc_type.value] += 1
            stats.kept_by_scope[record.scope or "(종류상 해당 없음)"] += 1
            kept_docs.append((doc.doc_id, doc.title, record.clean_text or ""))

    progress(f"[3/4] processed_documents 기록 중: {len(records):,}건 ({elapsed()})")
    replace_processed_documents(conn, records)
    stats.statuses.update(r.status for r in records)
    progress(f"[4/4] 키워드 색인 생성 중: {len(kept_docs):,}건 ({elapsed()})")
    stats.keyword_indexed = replace_keyword_index(conn, kept_docs)
    progress(f"완료 ({elapsed()})")
    return stats


def run(quality_name: str, dedup_name: str) -> PrepareStats:
    quality_config = None if quality_name == "none" else quality.get_quality_config(quality_name)
    dedup_config = dedup.get_dedup_config(dedup_name)
    with open_db() as conn:
        stats = prepare(conn, quality_config, dedup_config, progress=lambda message: print(message, flush=True))
    print(stats.summary())
    return stats
