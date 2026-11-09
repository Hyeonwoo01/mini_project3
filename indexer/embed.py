"""임베딩 생성 및 Chroma 적재·검증 (가이드 4-5).

    python -m indexer build --chunk-config c500 --dry-run   # 청크 수·길이 분포만 확인 (API 호출 없음)
    python -m indexer build --chunk-config c500             # 임베딩 후 docs__c500 에 적재
    python -m indexer verify --collection docs__c500        # 문서 수·청크 수·빈 청크·샘플 검색 확인

- 임베딩 모델은 common.embeddings.get_embedding_model() 로만 생성한다
- 벡터 DB 에는 청크와 메타데이터(common.schema.CHUNK_METADATA_FIELDS)만 저장한다. 원문은 docs.db
- 컬렉션 메타데이터에 임베딩 모델과 청킹 설정을 기록한다. retriever 가 모델 일치 여부를 확인한다
- 거리 계산은 코사인. 유사도 = 1 - 거리
- 이어서 적재: 청크 id 가 결정적(<doc_id>#<순번>)이므로 중단 후 같은 명령을 다시 실행하면 이미 적재된
  청크는 건너뛴다 (공용 API 키의 호출 한도로 중간에 실패할 수 있다). 같은 id 라도 청크 내용이 바뀌었으면
  (청킹 규칙 변경 등) 다시 임베딩해 덮어쓴다. 내용은 같고 메타데이터만 바뀌었으면 (품질 규칙 변경으로 scope 가
  바뀐 경우 등) 임베딩 없이 메타데이터만 갱신한다.
- 지식베이스가 바뀌어(indexer prepare 재실행) 더 이상 없는 청크는 컬렉션에서 지운다.
"""

import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from sqlite3 import Connection
from statistics import quantiles

import chromadb
from langchain_core.embeddings import Embeddings

from common.config import get_settings
from common.docstore import open_db
from common.embeddings import get_embedding_model
from common.schema import DocType
from indexer.chunk import Chunk, ChunkConfig, ChunkSource, chunk_document
from common.dates import annotate_relative_dates

ADD_BATCH = 256                 # 한 번에 임베딩·적재하는 청크 수
PROGRESS_EVERY_SEC = 30         # 진행 상황 출력 간격 (멈춤을 바로 알아챌 수 있게)
# 컬렉션 전체를 한 번에 읽으면 청크 5만 개 이상에서 Chroma 내부 SQLite 한도(too many SQL variables)에
# 걸린다 (c500, 50,713개). 나눠 읽는다.
GET_PAGE = 5_000
SAMPLE_QUERIES = ("소상공인 정책자금 지원 규모", "지역화폐 예산 삭감 논란", "서울 골목상권 매출 변화")


@dataclass
class BuildStats:
    collection: str
    documents: int = 0
    chunks: int = 0
    skipped_empty: int = 0
    added: int = 0
    updated: int = 0
    metadata_updated: int = 0
    already_present: int = 0
    removed_stale: int = 0
    dated_documents: int = 0      # normalize_dates 설정에서 날짜를 덧붙인 문서 수
    dated_expressions: int = 0
    body_lengths: list[int] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"[build 결과] 컬렉션={self.collection}",
                 f"  문서 {self.documents:,}건 → 청크 {self.chunks:,}개 (문서당 평균 {self.chunks / max(self.documents, 1):.1f}개)"]
        if self.body_lengths:
            q = quantiles(self.body_lengths, n=20) if len(self.body_lengths) > 1 else self.body_lengths * 19
            lines.append(f"  청크 본문 길이: 최소 {min(self.body_lengths)} / 5% {q[0]:.0f} / 중앙값 {q[9]:.0f} "
                         f"/ 95% {q[18]:.0f} / 최대 {max(self.body_lengths)}자")
            short = sum(length < 100 for length in self.body_lengths)
            lines.append(f"  100자 미만 청크: {short:,}개 ({short / len(self.body_lengths):.1%})")
        if self.dated_expressions:
            lines.append(f"  상대 날짜에 날짜 덧붙임: 문서 {self.dated_documents:,}건 · 표현 {self.dated_expressions:,}개")
        lines.append(f"  빈 청크로 제외: {self.skipped_empty:,}개")
        lines.append(f"  새로 적재 {self.added:,}개 · 내용 변경으로 갱신 {self.updated:,}개 "
                     f"· 메타데이터만 갱신 {self.metadata_updated:,}개 · 이미 있음 {self.already_present:,}개 "
                     f"· 오래된 청크 삭제 {self.removed_stale:,}개")
        return "\n".join(lines)


def iter_chunk_sources(conn: Connection) -> Iterator[ChunkSource]:
    rows = conn.execute(
        """
        SELECT d.doc_id, d.doc_type, d.title, d.source, d.url, d.published_at, d.collected_at,
               p.clean_text, p.scope
        FROM processed_documents AS p JOIN documents AS d USING (doc_id)
        WHERE p.status = 'kept'
        ORDER BY d.doc_id
        """
    )
    for r in rows:
        yield ChunkSource(doc_id=r["doc_id"], doc_type=DocType(r["doc_type"]), title=r["title"],
                          text=r["clean_text"] or "", source=r["source"], url=r["url"],
                          published_at=r["published_at"], collected_at=r["collected_at"], scope=r["scope"])


def _client(chroma_dir: Path | None) -> chromadb.ClientAPI:
    path = chroma_dir or get_settings().chroma_dir
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path))


def _collection_names(client: chromadb.ClientAPI) -> set[str]:
    return {c if isinstance(c, str) else c.name for c in client.list_collections()}


def _get_all(col, include: list[str]) -> dict[str, list]:
    """컬렉션 전체를 GET_PAGE 개씩 나눠 읽는다."""
    result: dict[str, list] = {"ids": [], **{key: [] for key in include}}
    offset = 0
    while True:
        page = col.get(include=include, limit=GET_PAGE, offset=offset)
        if not page["ids"]:
            return result
        result["ids"].extend(page["ids"])
        for key in include:
            result[key].extend(page[key])
        offset += len(page["ids"])


def _make_chunks(conn: Connection, config: ChunkConfig, stats: BuildStats) -> list[Chunk]:
    chunks: list[Chunk] = []
    for source in iter_chunk_sources(conn):
        stats.documents += 1
        if config.normalize_dates:
            found = sum(annotate_relative_dates(text, source.published_at)[1] for text in (source.title, source.text))
            stats.dated_documents += found > 0
            stats.dated_expressions += found
        doc_chunks = chunk_document(source, config)
        if not doc_chunks:
            stats.skipped_empty += 1
        chunks.extend(doc_chunks)
    stats.chunks = len(chunks)
    stats.body_lengths = [len(c.body) for c in chunks]
    return chunks


def build_collection(
    config: ChunkConfig,
    collection: str,
    *,
    rebuild: bool = False,
    dry_run: bool = False,
    embeddings: Embeddings | None = None,
    embedding_model_name: str | None = None,
    chroma_dir: Path | None = None,
    db_path: Path | None = None,
    progress=print,
) -> BuildStats:
    stats = BuildStats(collection=collection)
    with open_db(db_path, read_only=True) as conn:
        chunks = _make_chunks(conn, config, stats)
    if dry_run:
        return stats

    model_name = embedding_model_name or get_settings().embedding_model
    embeddings = embeddings or get_embedding_model()
    client = _client(chroma_dir)
    if rebuild and collection in _collection_names(client):
        client.delete_collection(collection)

    col = client.get_or_create_collection(
        collection,
        configuration={"hnsw": {"space": "cosine"}},
        metadata={"embedding_model": model_name, "chunk_config": config.name, "chunk_size": config.chunk_size,
                  "chunk_overlap": config.chunk_overlap, "include_title": config.include_title,
                  "normalize_dates": config.normalize_dates,
                  "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")},
    )
    stored_model = (col.metadata or {}).get("embedding_model")
    if stored_model != model_name:
        raise SystemExit(f"컬렉션 {collection} 은 {stored_model} 로 만들어졌습니다 (현재 {model_name}). "
                         "--rebuild 로 다시 만드세요.")

    current_ids = {c.chunk_id for c in chunks}
    stale = [chunk_id for chunk_id in _get_all(col, include=[])["ids"] if chunk_id not in current_ids]
    for i in range(0, len(stale), ADD_BATCH * 20):
        col.delete(ids=stale[i:i + ADD_BATCH * 20])
    stats.removed_stale = len(stale)

    started = last_report = time.monotonic()
    batches = [chunks[i:i + ADD_BATCH] for i in range(0, len(chunks), ADD_BATCH)]
    for number, batch in enumerate(batches, start=1):
        found = col.get(ids=[c.chunk_id for c in batch], include=["documents", "metadatas"])
        stored = {cid: (doc, meta) for cid, doc, meta in zip(found["ids"], found["documents"], found["metadatas"])}
        todo = [c for c in batch if c.chunk_id not in stored or stored[c.chunk_id][0] != c.text]
        changed = sum(c.chunk_id in stored for c in todo)
        meta_only = [c for c in batch if c.chunk_id in stored and stored[c.chunk_id][0] == c.text
                     and stored[c.chunk_id][1] != c.metadata]
        if meta_only:   # documents·embeddings 를 넘기지 않으면 Chroma 는 임베딩을 다시 계산하지 않는다
            col.update(ids=[c.chunk_id for c in meta_only], metadatas=[c.metadata for c in meta_only])
            stats.metadata_updated += len(meta_only)
        stats.already_present += len(batch) - len(todo) - len(meta_only)
        if todo:
            try:
                vectors = embeddings.embed_documents([c.text for c in todo])
            except Exception as exc:  # API 오류(호출 한도 등)를 이어서 적재 안내와 함께 알린다
                raise SystemExit(
                    f"임베딩 중단 ({stats.added + stats.updated + stats.metadata_updated + stats.already_present:,}"
                    f"/{len(chunks):,}개 처리): {exc}\n"
                    "같은 명령을 다시 실행하면 이미 적재된 청크는 건너뛰고 이어서 진행합니다."
                ) from exc
            col.upsert(ids=[c.chunk_id for c in todo], documents=[c.text for c in todo],
                       embeddings=vectors, metadatas=[c.metadata for c in todo])
            stats.updated += changed
            stats.added += len(todo) - changed
        now = time.monotonic()
        if now - last_report >= PROGRESS_EVERY_SEC or number == len(batches):
            last_report = now
            done = number * ADD_BATCH if number < len(batches) else len(chunks)
            elapsed = now - started
            eta = elapsed / done * (len(chunks) - done) if done else 0
            progress(f"  {done:,}/{len(chunks):,} 청크 ({elapsed:.0f}초, 남은 시간 약 {eta / 60:.0f}분)")
    return stats


@dataclass
class VerifyReport:
    collection: str
    metadata: dict
    chunks: int
    documents: int
    kept_documents: int
    missing_documents: int
    empty_chunks: int
    # 출처 추적에 필요한 메타데이터 칸별로 값이 빈 문서 수 (가이드 점검: 모든 문서의 출처 URL·발행일·수집 일시)
    missing_metadata: dict[str, int] = field(default_factory=dict)
    samples: list[tuple[str, list[tuple[float, str, str]]]] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"[verify 결과] 컬렉션={self.collection}",
                 f"  설정: {self.metadata}",
                 f"  청크 {self.chunks:,}개 · 문서 {self.documents:,}건 (인덱싱 대상 {self.kept_documents:,}건, "
                 f"누락 {self.missing_documents:,}건) · 빈 청크 {self.empty_chunks:,}개"]
        if self.missing_metadata:
            lines.append("  메타데이터 빈 값 (문서 수): "
                         + " · ".join(f"{name} {count:,}" for name, count in self.missing_metadata.items()))
        for query, hits in self.samples:
            lines.append(f"  [샘플 검색] {query}")
            lines += [f"    {score:.3f}  {date[:10]}  {title}" for score, date, title in hits]
        return "\n".join(lines)


TRACE_FIELDS = ("doc_type", "source", "title", "url", "published_at", "collected_at")


def missing_metadata_counts(metadatas: Sequence[dict]) -> dict[str, int]:
    """칸마다 값이 비어 있는 청크가 하나라도 있는 문서 수."""
    missing: dict[str, set[str]] = {name: set() for name in TRACE_FIELDS}
    for meta in metadatas:
        for name in TRACE_FIELDS:
            if not str(meta.get(name) or "").strip():
                missing[name].add(meta["doc_id"])
    return {name: len(ids) for name, ids in missing.items()}


def verify_collection(
    collection: str,
    *,
    queries: Sequence[str] = SAMPLE_QUERIES,
    embeddings: Embeddings | None = None,
    chroma_dir: Path | None = None,
    db_path: Path | None = None,
    k: int = 3,
) -> VerifyReport:
    client = _client(chroma_dir)
    if collection not in _collection_names(client):
        raise SystemExit(f"컬렉션이 없습니다: {collection}. python -m indexer build 를 먼저 실행하세요.")
    col = client.get_collection(collection)
    data = _get_all(col, include=["documents", "metadatas"])
    doc_ids = {m["doc_id"] for m in data["metadatas"]}
    with open_db(db_path, read_only=True) as conn:
        kept = {r[0] for r in conn.execute("SELECT doc_id FROM processed_documents WHERE status = 'kept'")}
    report = VerifyReport(
        collection=collection, metadata=dict(col.metadata or {}), chunks=len(data["ids"]),
        documents=len(doc_ids), kept_documents=len(kept), missing_documents=len(kept - doc_ids),
        empty_chunks=sum(not (d or "").strip() for d in data["documents"]),
        missing_metadata=missing_metadata_counts(data["metadatas"]),
    )
    if queries:
        embeddings = embeddings or get_embedding_model()
        for query in queries:
            result = col.query(query_embeddings=[embeddings.embed_query(query)], n_results=k,
                               include=["metadatas", "distances"])
            hits = [(1 - dist, meta["published_at"], meta["title"])
                    for meta, dist in zip(result["metadatas"][0], result["distances"][0])]
            report.samples.append((query, hits))
    return report
