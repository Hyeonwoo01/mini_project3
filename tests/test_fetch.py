import pytest
import requests

from collector.fetch import ALLOWED, DISALLOWED, UNAVAILABLE, RobotsDisallowed, describe_error, fetch

SITE = "https://example.go.kr"
ROBOTS = f"{SITE}/robots.txt"


def test_delay_below_minimum_is_rejected(make_fetcher):
    with pytest.raises(ValueError):
        make_fetcher(delay=0.5)


def test_sends_identifiable_user_agent(make_fetcher, http):
    make_fetcher(user_agent="edu-bot/0.1")
    assert http.headers["User-Agent"] == "edu-bot/0.1"


def test_waits_between_requests_and_reads_robots_once_per_host(make_fetcher, http, clock):
    http.route(ROBOTS, (200, "User-agent: *\nDisallow: /admin\n"))
    http.route(f"{SITE}/a", (200, "<html>a</html>"))
    http.route(f"{SITE}/b", (200, "<html>b</html>"))
    fetcher = make_fetcher(delay=1.5)

    assert fetcher.get(f"{SITE}/a").content == b"<html>a</html>"
    fetcher.get(f"{SITE}/b")

    assert http.calls == [ROBOTS, f"{SITE}/a", f"{SITE}/b"]
    assert clock.sleeps == [1.5, 1.5]          # robots → a, a → b 사이마다 대기


def test_disallowed_path_is_not_requested(make_fetcher, http):
    http.route(ROBOTS, (200, "User-agent: *\nDisallow: /admin\n"))
    fetcher = make_fetcher()

    assert fetcher.check_robots(f"{SITE}/admin/x").status == DISALLOWED
    with pytest.raises(RobotsDisallowed):
        fetcher.get(f"{SITE}/admin/x")
    assert http.calls == [ROBOTS]


def test_rules_for_our_user_agent_apply(make_fetcher, http):
    http.route(ROBOTS, (200, "User-agent: test-bot\nDisallow: /\n\nUser-agent: *\nAllow: /\n"))
    assert make_fetcher(user_agent="test-bot/1.0").check_robots(f"{SITE}/news").status == DISALLOWED


@pytest.mark.parametrize("code, expected", [(404, ALLOWED), (403, DISALLOWED), (401, DISALLOWED)])
def test_robots_response_codes(make_fetcher, http, code, expected):
    http.route(ROBOTS, (code, ""))
    assert make_fetcher().check_robots(f"{SITE}/news").status == expected


def test_robots_server_error_is_unavailable_after_retries(make_fetcher, http):
    http.route(ROBOTS, (503, ""))
    check = make_fetcher().check_robots(f"{SITE}/news")
    assert check.status == UNAVAILABLE
    assert http.calls == [ROBOTS] * 3          # 첫 요청 + 재시도 2회


def test_robots_connection_failure_is_unavailable_with_reason(make_fetcher, http):
    http.route(ROBOTS, requests.exceptions.SSLError("certificate verify failed"))
    fetcher = make_fetcher()
    check = fetcher.check_robots(f"{SITE}/news")
    assert check.status == UNAVAILABLE
    assert "SSLError" in check.detail
    with pytest.raises(RobotsDisallowed):
        fetcher.get(f"{SITE}/news")


def test_longer_crawl_delay_from_robots_is_respected(make_fetcher, http, clock):
    http.route(ROBOTS, (200, "User-agent: *\nCrawl-delay: 5\nAllow: /\n"))
    http.route(f"{SITE}/a", (200, "a"))
    fetcher = make_fetcher(delay=1.5)

    assert fetcher.check_robots(f"{SITE}/a").crawl_delay == 5.0
    fetcher.get(f"{SITE}/a")
    assert clock.sleeps == [5.0]               # robots 요청 뒤 본문 요청까지 Crawl-delay 만큼 대기


def test_retries_server_error_then_succeeds(make_fetcher, http):
    http.route(ROBOTS, (404, ""))
    http.route(f"{SITE}/a", (502, ""), (200, "ok"))
    result = make_fetcher().get(f"{SITE}/a")
    assert (result.status, result.content) == (200, b"ok")
    assert http.calls.count(f"{SITE}/a") == 2


def test_client_error_is_returned_without_retry(make_fetcher, http):
    http.route(ROBOTS, (404, ""))
    http.route(f"{SITE}/missing", (404, "nope"))
    assert make_fetcher().get(f"{SITE}/missing").status == 404
    assert http.calls.count(f"{SITE}/missing") == 1


def test_encoding_falls_back_to_detected_when_header_has_no_charset(make_fetcher, http, fake_response):
    body = "보도자료".encode("euc-kr")
    http.route(ROBOTS, (404, ""))
    http.route(f"{SITE}/kr", fake_response(f"{SITE}/kr", 200, body, {"Content-Type": "text/html"},
                                           encoding="ISO-8859-1", apparent_encoding="EUC-KR"))
    result = make_fetcher().get(f"{SITE}/kr")
    assert result.encoding == "EUC-KR"
    assert result.text == "보도자료"
    assert result.collected_at.endswith("+00:00")


def test_fetch_raises_for_non_2xx(monkeypatch, make_fetcher, http):
    import collector.fetch as fetch_module

    http.route(ROBOTS, (404, ""))
    http.route(f"{SITE}/gone", (410, ""))
    monkeypatch.setattr(fetch_module, "_default", make_fetcher())
    with pytest.raises(requests.HTTPError):
        fetch(f"{SITE}/gone")


def test_describe_error_shows_cause_first():
    exc = requests.exceptions.ProxyError(
        "HTTPSConnectionPool(host='a.go.kr', port=443): Max retries exceeded with url: /robots.txt "
        "(Caused by ProxyError('Unable to connect to proxy', OSError('Tunnel connection failed: 403')))")
    assert describe_error(exc).startswith("ProxyError: Caused by ProxyError('Unable to connect to proxy'")
    assert describe_error(requests.Timeout("read timed out")) == "Timeout: read timed out"
