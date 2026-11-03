"""크롤링 준수 사항을 강제하는 공용 HTTP 요청 모듈 (가이드 3-3).

- robots.txt 확인 후 허용된 경로만 요청한다 (호스트별로 한 번 읽어 둔다)
- 요청 간 최소 딜레이(CRAWL_DELAY_SEC, 1초 이상)를 지키며 병렬 요청을 하지 않는다.
  robots.txt 의 Crawl-delay 가 더 길면 그 호스트에는 Crawl-delay 를 따른다
- 식별 가능한 User-Agent(CRAWLER_USER_AGENT)를 보낸다
- 로그인 이후 페이지·개인정보는 수집하지 않는다 (크롤러가 대상 URL 을 공개 목록·본문으로 한정)
- 응답과 함께 수집 일시를 반환하여 모든 문서에 기록되도록 한다

모든 크롤러(crawl_*.py, collect_*.py)는 requests 를 직접 호출하지 않고 이 모듈을 거친다.

robots.txt 응답별 처리 (표준 관례와 같다)
    200        규칙대로 판단
    401·403    전체 금지로 본다
    그 밖의 4xx 규칙이 없는 것으로 보고 허용 (404 등)
    5xx·접속 실패 확인할 수 없으므로 금지로 본다 (보수적)
"""

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import requests

from common.config import MIN_CRAWL_DELAY_SEC, get_settings

DEFAULT_TIMEOUT = 20      # 초
MAX_RETRIES = 2           # 5xx·접속 오류 재시도 횟수

ALLOWED, DISALLOWED, UNAVAILABLE = "allowed", "disallowed", "unavailable"


class RobotsDisallowed(Exception):
    """robots.txt 가 허용하지 않거나 확인할 수 없는 URL."""


@dataclass(frozen=True)
class RobotsCheck:
    status: str                       # allowed | disallowed | unavailable(확인 불가 → 금지 취급)
    detail: str                       # robots.txt 응답 코드 또는 접속 오류
    crawl_delay: float | None = None  # robots.txt 의 Crawl-delay (없으면 None)


@dataclass(frozen=True)
class FetchResult:
    url: str                # 리다이렉트 후 최종 URL
    status: int
    content: bytes
    content_type: str
    encoding: str | None
    collected_at: str       # ISO 8601 (UTC)

    @property
    def text(self) -> str:
        return self.content.decode(self.encoding or "utf-8", errors="replace")


def describe_error(exc: Exception, limit: int = 160) -> str:
    """요청 예외를 한 줄로. urllib3 메시지는 앞부분이 주소 반복이라 원인('Caused by ...')부터 보여 준다."""
    message = str(exc)
    cause = message.find("Caused by ")
    if cause >= 0:
        message = message[cause:]
    return f"{type(exc).__name__}: {message[:limit]}"


def _host(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}"


class Fetcher:
    """robots.txt·딜레이·User-Agent 를 지키는 순차 요청기. 크롤러 실행 하나에 하나를 만들어 쓴다."""

    def __init__(self, *, delay: float | None = None, user_agent: str | None = None,
                 timeout: float = DEFAULT_TIMEOUT, session: requests.Session | None = None,
                 clock=time.monotonic, sleep=time.sleep) -> None:
        settings = get_settings()
        self.delay = settings.crawl_delay_sec if delay is None else delay
        if self.delay < MIN_CRAWL_DELAY_SEC:
            raise ValueError(f"요청 간 딜레이는 {MIN_CRAWL_DELAY_SEC}초 이상이어야 합니다")
        self.user_agent = user_agent or settings.crawler_user_agent
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = self.user_agent
        self._clock, self._sleep = clock, sleep
        self._last_request: float | None = None
        # 호스트 → (규칙 파서 또는 None, 파서가 없을 때의 일괄 판단, 판단 근거)
        self._robots: dict[str, tuple[RobotFileParser | None, str, str]] = {}

    def _wait_turn(self, delay: float) -> None:
        if self._last_request is not None:
            remaining = delay - (self._clock() - self._last_request)
            if remaining > 0:
                self._sleep(remaining)
        self._last_request = self._clock()

    def _request(self, url: str, params: dict | None = None, *, delay: float | None = None) -> requests.Response:
        for attempt in range(MAX_RETRIES + 1):
            self._wait_turn(self.delay if delay is None else delay)
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException:
                if attempt == MAX_RETRIES:
                    raise
                continue
            if response.status_code < 500 or attempt == MAX_RETRIES:
                return response
        raise AssertionError("unreachable")

    def _load_robots(self, host: str) -> tuple[RobotFileParser | None, str, str]:
        try:
            response = self._request(f"{host}/robots.txt")
        except requests.RequestException as exc:
            return None, UNAVAILABLE, f"접속 실패 — {describe_error(exc)}"
        code = response.status_code
        if code == 200:
            parser = RobotFileParser()
            parser.parse(response.text.splitlines())
            return parser, "", "robots.txt 200"
        if code in (401, 403):
            return None, DISALLOWED, f"robots.txt {code} → 전체 금지로 봄"
        if 400 <= code < 500:
            return None, ALLOWED, f"robots.txt {code} → 규칙 없음으로 봄"
        return None, UNAVAILABLE, f"robots.txt {code} → 확인 불가"

    def check_robots(self, url: str) -> RobotsCheck:
        host = _host(url)
        if host not in self._robots:
            self._robots[host] = self._load_robots(host)
        parser, fixed_status, detail = self._robots[host]
        if parser is None:
            return RobotsCheck(fixed_status, detail)
        crawl_delay = parser.crawl_delay(self.user_agent)
        status = ALLOWED if parser.can_fetch(self.user_agent, url) else DISALLOWED
        return RobotsCheck(status, detail, float(crawl_delay) if crawl_delay else None)

    def get(self, url: str, params: dict | None = None) -> FetchResult:
        """robots.txt 가 허용한 URL 만 요청한다. 허용하지 않거나 확인할 수 없으면 RobotsDisallowed."""
        robots = self.check_robots(url)
        if robots.status != ALLOWED:
            raise RobotsDisallowed(f"robots.txt {robots.status} ({robots.detail}): {url}")
        delay = max(self.delay, robots.crawl_delay or 0.0)
        response = self._request(url, params=params, delay=delay)
        encoding = response.encoding
        if not encoding or encoding.lower() == "iso-8859-1":   # 헤더에 charset 이 없으면 본문에서 추정
            encoding = response.apparent_encoding
        return FetchResult(url=response.url, status=response.status_code, content=response.content,
                           content_type=response.headers.get("Content-Type", ""), encoding=encoding,
                           collected_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))


_default: Fetcher | None = None


def fetch(url: str) -> tuple[bytes, str]:
    """URL 본문과 수집 일시(ISO 8601)를 반환한다. 2xx 가 아니면 requests.HTTPError."""
    global _default
    _default = _default or Fetcher()
    result = _default.get(url)
    if not 200 <= result.status < 300:
        raise requests.HTTPError(f"{result.status} {url}")
    return result.content, result.collected_at
