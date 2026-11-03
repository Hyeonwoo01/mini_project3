"""python -m indexer prepare — 정제 → 품질 필터 → 중복 처리 후 processed_documents 를 다시 만든다.

    python -m indexer prepare                     # 품질 v1 + 중복 처리 v1
    python -m indexer prepare --quality none      # 품질 필터 없이 (개선 전 기준 실험)

실행할 때마다 processed_documents 전체를 다시 계산한다. 설정 이름이 실험 태그가 되므로
결과 요약(사유별 건수)을 리포트의 처리 전후 건수로 쓴다.
"""

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from sqlite3 import Connection

from common.docstore import iter_documents, open_db, replace_processed_documents
from common.schema import DocType, ProcessedRecord
from indexer import dedup, quality
from indexer.parse import clean_text, extract_text


@dataclass
class PrepareStats:
    quality_config: str
    dedup_config: str
    total: int = 0
    statuses: Counter[str] = field(default_factory=Counter)
    excluded_reasons: Counter[str] = field(default_factory=Counter)
    kept_by_type: Counter[str] = field(default_factory=Counter)
    kept_by_scope: Counter[str] = field(default_factory=Counter)

    def summary(self) -> str:
        lines = [f"[prepare 결과] 품질={self.quality_config} 중복={self.dedup_config}",
                 f"  전체 문서: {self.total:,}"]
        lines += [f"  제외 ({reason}): {n:,}" for reason, n in self.excluded_reasons.most_common()]
        lines += [f"  중복 (대표 문서로 대체): {self.statuses['duplicate']:,}",
                  f"  유지: {self.statuses['kept']:,}"]
        lines += [f"    문서 종류 {doc_type}: {n:,}" for doc_type, n in self.kept_by_type.most_common()]
        lines += [f"    범위 {scope}: {n:,}" for scope, n in self.kept_by_scope.most_common()]
        return "\n".join(lines)


def _document_text(row) -> str:
    if row["raw_text"]:
        return clean_text(row["raw_text"])
    if row["raw_path"]:
        return clean_text(extract_text(row["raw_path"]))
    return ""


def prepare(
    conn: Connection, quality_config: quality.QualityConfig | None, dedup_config: dedup.DedupConfig
) -> PrepareStats:
    processed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    stats = PrepareStats(quality_config.name if quality_config else "none", dedup_config.name)
    records: list[ProcessedRecord] = []
    survivors: list[dedup.DedupInput] = []
    pending: dict[str, dict] = {}

    for row in iter_documents(conn):
        stats.total += 1
        doc_type = DocType(row["doc_type"])
        text = _document_text(row)
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

    results = dedup.deduplicate(survivors, dedup_config)
    for doc in survivors:
        result = results[doc.doc_id]
        status = "duplicate" if result.is_duplicate else "kept"
        record = ProcessedRecord(status=status, canonical_doc_id=result.canonical_doc_id, **pending[doc.doc_id])
        records.append(record)
        if status == "kept":
            stats.kept_by_type[doc.doc_type.value] += 1
            stats.kept_by_scope[record.scope or "(종류상 해당 없음)"] += 1

    replace_processed_documents(conn, records)
    stats.statuses.update(r.status for r in records)
    return stats


def run(quality_name: str, dedup_name: str) -> PrepareStats:
    quality_config = None if quality_name == "none" else quality.get_quality_config(quality_name)
    dedup_config = dedup.get_dedup_config(dedup_name)
    with open_db() as conn:
        stats = prepare(conn, quality_config, dedup_config)
    print(stats.summary())
    return stats
