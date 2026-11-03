"""수집 원문 저장.

- 원본 파일(HTML/PDF)은 data/raw/<doc_type>/<doc_id>.<ext> 로 저장한다
- 메타데이터는 common.docstore.upsert_documents 로 docs.db 에 등록한다
- doc_id 는 URL·발행일로부터 결정적으로 생성하여 재수집 시 중복 적재 대신 갱신되도록 한다
  (같은 URL 에는 항상 같은 발행일 값을 넘겨야 id 가 바뀌지 않는다)
"""

import hashlib
from pathlib import Path

from common.config import PROJECT_ROOT, get_settings
from common.schema import DocType


def make_doc_id(prefix: str, url: str, published_at: str | None) -> str:
    """예: make_doc_id("kr", url, "2026-01-15") -> "kr_20260115_1a2b3c4d"."""
    if not prefix.isalnum():
        raise ValueError(f"prefix 는 영숫자여야 합니다: {prefix!r}")
    if not url.strip():
        raise ValueError("url 이 비어 있습니다")
    ymd = published_at[:10].replace("-", "") if published_at else "00000000"
    digest = hashlib.sha256(url.strip().encode("utf-8")).hexdigest()[:8]
    return f"{prefix}_{ymd}_{digest}"


def save_raw(doc_type: DocType, doc_id: str, content: bytes, ext: str) -> str:
    """원본을 저장하고 docs.db 에 기록할 프로젝트 루트 기준 상대 경로를 반환한다."""
    ext = ext.lstrip(".").lower()
    if not ext.isalnum():
        raise ValueError(f"잘못된 확장자: {ext!r}")
    path: Path = get_settings().raw_dir / DocType(doc_type).value / f"{doc_id}.{ext}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path.relative_to(PROJECT_ROOT).as_posix()
