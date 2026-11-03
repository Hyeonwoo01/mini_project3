"""프로젝트 전역 설정.

경로는 프로젝트 루트 기준으로 고정하고, 모델·컬렉션·크롤링 파라미터만 .env 로 조정한다.
API 키는 Settings 에 담지 않는다 (repr·로그를 통한 노출 방지). 필요한 시점에
require_env() 로 존재 여부만 확인하고, 실제 값은 각 SDK 가 환경변수에서 직접 읽는다.
"""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MIN_CRAWL_DELAY_SEC = 1.0


@dataclass(frozen=True)
class Settings:
    # 경로
    provided_dir: Path       # 제공 데이터(뉴스) 원본
    raw_dir: Path            # 직접 수집 원본 파일 (HTML/PDF)
    docs_db_path: Path       # 원문 및 메타데이터 (SQLite)
    chroma_dir: Path         # 벡터 DB 영속 디렉터리
    eval_set_path: Path
    eval_results_dir: Path

    # 모델·인덱스
    chat_model: str
    embedding_model: str
    chroma_collection: str

    # 크롤링
    crawl_delay_sec: float
    crawler_user_agent: str


def _env(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    return value or default


def _parse_crawl_delay(raw: str) -> float:
    try:
        delay = float(raw)
    except ValueError as exc:
        raise ValueError(f"CRAWL_DELAY_SEC 는 숫자여야 합니다: {raw!r}") from exc
    if delay < MIN_CRAWL_DELAY_SEC:
        raise ValueError(
            f"CRAWL_DELAY_SEC 는 {MIN_CRAWL_DELAY_SEC}초 이상이어야 합니다: {delay}"
        )
    return delay


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    return Settings(
        provided_dir=PROJECT_ROOT / "data" / "provided",
        raw_dir=PROJECT_ROOT / "data" / "raw",
        docs_db_path=PROJECT_ROOT / "store" / "docs.db",
        chroma_dir=PROJECT_ROOT / "store" / "chroma",
        eval_set_path=PROJECT_ROOT / "eval" / "eval_set.jsonl",
        eval_results_dir=PROJECT_ROOT / "eval" / "results",
        chat_model=_env("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        embedding_model=_env("EMBEDDING_MODEL", "text-embedding-3-small"),
        chroma_collection=_env("CHROMA_COLLECTION", "docs__c500"),
        crawl_delay_sec=_parse_crawl_delay(_env("CRAWL_DELAY_SEC", "1.5")),
        crawler_user_agent=_env("CRAWLER_USER_AGENT", "mini-project3-edu-crawler/0.1"),
    )


def require_env(name: str) -> None:
    """환경변수가 비어 있으면 원인과 해결 방법을 담은 오류를 낸다. 값은 반환하지 않는다."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    if not os.getenv(name, "").strip():
        raise RuntimeError(
            f"환경변수 {name} 가 설정되지 않았습니다. "
            f".env.example 을 .env 로 복사한 뒤 값을 채우세요."
        )
