"""공용 테스트 도구: 네트워크 없이 collector.fetch.Fetcher 를 시험하는 가짜 HTTP 세션과 시계."""

import pytest


class FakeResponse:
    def __init__(self, url: str, status: int = 200, body: str | bytes = "", headers: dict | None = None,
                 encoding: str | None = "utf-8", apparent_encoding: str = "utf-8") -> None:
        self.url = url
        self.status_code = status
        self.content = body.encode("utf-8") if isinstance(body, str) else body
        self.headers = headers if headers is not None else {"Content-Type": "text/html; charset=utf-8"}
        self.encoding = encoding
        self.apparent_encoding = apparent_encoding

    @property
    def text(self) -> str:
        return self.content.decode(self.encoding or "utf-8", errors="replace")


class FakeSession:
    """URL 별로 미리 정한 응답(또는 예외)을 순서대로 돌려준다. 등록하지 않은 URL 은 404."""

    def __init__(self) -> None:
        self.headers: dict[str, str] = {}
        self.calls: list[str] = []
        self._routes: dict[str, list] = {}

    def route(self, url: str, *responses) -> None:
        """응답 인자: (status, body) 튜플, (status, body, headers) 튜플, FakeResponse, 예외 인스턴스."""
        self._routes[url] = list(responses)

    def get(self, url, params=None, timeout=None):
        self.calls.append(url)
        queue = self._routes.get(url)
        if not queue:
            return FakeResponse(url, 404, "not found")
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, Exception):
            raise item
        if isinstance(item, FakeResponse):
            return item
        return FakeResponse(url, *item)


class FakeClock:
    """sleep 하면 시간이 흐르고, 요청 자체는 시간이 걸리지 않는 시계."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(round(seconds, 6))
        self.now += seconds


@pytest.fixture
def http():
    return FakeSession()


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def make_fetcher(http, clock):
    from collector.fetch import Fetcher

    def _make(delay: float = 1.5, user_agent: str = "test-bot/1.0") -> Fetcher:
        return Fetcher(delay=delay, user_agent=user_agent, session=http, clock=clock, sleep=clock.sleep)

    return _make



@pytest.fixture
def fake_response():
    return FakeResponse
