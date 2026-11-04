import pytest

from collector.probe import PROBE_TARGETS, ProbeTarget, analyze_html, probe_target, run

SITE = "https://board.go.kr"
LIST_HTML = """<html><head><title>보도자료 | 기관</title><script>var x = "긴 스크립트";</script></head>
<body><nav><a href="/press">보도자료</a> <a href="/press">보도자료</a> <a href="/about">기관소개</a>
<a href="javascript:openMenu()">메뉴</a></nav>
<ul>
  <li><a href="/press/101?curPage=1">소상공인 지원 확대</a></li>
  <li><a href="/press/102">카드수수료 인하</a></li>
  <li><a href="https://board.go.kr/press/101?curPage=1">중복 링크</a></li>
  <li><a href="javascript:fn_view('103')">자바스크립트 상세</a></li>
  <li><a href="#" onclick="goView(104); return false;">onclick 상세</a></li>
  <li><a name="anchor-only">링크 아님</a></li>
</ul></body></html>"""


def test_analyze_html_finds_keyword_detail_and_js_links():
    info = analyze_html(LIST_HTML, f"{SITE}/press", keywords=("보도자료",), detail_pattern=r"/press/\d+")

    assert info.title == "보도자료 | 기관"
    assert info.link_count == 9                       # href·onclick 이 없는 a 는 세지 않는다
    assert info.js_links == 3
    # 게시물 제목처럼 긴 링크의 호출 방식이 메뉴 링크보다 먼저 나온다
    assert info.js_samples == ["javascript:fn_view('103')", "goView(104); return false;", "javascript:openMenu()"]
    assert info.keyword_links == [("보도자료", f"{SITE}/press")]      # 같은 주소는 한 번만
    assert info.detail_urls == [f"{SITE}/press/101?curPage=1", f"{SITE}/press/102"]


def test_analyze_html_finds_attachment_links():
    html = """<a href="/comm/getFile?upperNo=1&fileNo=1">보도자료.hwp</a>
    <a href="/files/report.PDF">보고서</a> <a href="/files/report.PDF">보고서 중복</a>
    <a href="/pdf-guide">안내</a> <a href="/press/1">본문</a>"""
    info = analyze_html(html, SITE)
    assert info.attachments == [("보도자료.hwp", f"{SITE}/comm/getFile?upperNo=1&fileNo=1"),
                                ("보고서", f"{SITE}/files/report.PDF")]


def test_text_chars_excludes_scripts_and_styles():
    html = "<html><head><style>p{}</style><script>var a = 1;</script></head><body><p>가나 다</p></body></html>"
    assert analyze_html(html, SITE).text_chars == len("가나 다")


def _target(**kwargs):
    defaults = dict(key="board", name="시험 게시판", url=f"{SITE}/press", keywords=("보도자료",),
                    detail_pattern=r"/press/\d+", verified=True)
    return ProbeTarget(**(defaults | kwargs))


def test_probe_saves_list_and_first_detail(tmp_path, make_fetcher, http):
    http.route(f"{SITE}/robots.txt", (200, "User-agent: *\nAllow: /\n"))
    http.route(f"{SITE}/press", (200, LIST_HTML))
    http.route(f"{SITE}/press/101?curPage=1", (200, "<html><title>상세</title><body>본문 내용"
                                                   "<a href='/comm/getFile?fileNo=1'>원문.hwp</a></body></html>"))
    lines = []

    summary = probe_target(make_fetcher(), _target(), tmp_path, lines.append)

    assert summary == "200 · 링크 9 · 상세 2건, 첫 상세 200, 첨부 1"
    assert "  첨부    1개 · 첫 첨부 robots allowed" in lines
    assert f"{SITE}/comm/getFile?fileNo=1" not in http.calls       # 첨부는 내려받지 않는다
    assert (tmp_path / "board_list.html").read_text(encoding="utf-8") == LIST_HTML
    assert "본문 내용" in (tmp_path / "board_detail.html").read_text(encoding="utf-8")
    assert any(line.startswith("  키워드  보도자료 → ") for line in lines)



def test_probe_lists_attachments_of_a_detail_page_target(tmp_path, make_fetcher, http):
    """상세 페이지를 직접 확인하는 후보(detail_pattern 없음)도 그 페이지의 PDF 링크를 보여 준다. 내려받지는 않는다."""
    http.route(f"{SITE}/robots.txt", (200, "User-agent: *\nAllow: /\n"))
    http.route(f"{SITE}/report/7", (200, "<html><title>보고서</title><body>요약"
                                         "<a href='/files/r7.pdf'>원문 PDF</a></body></html>"))
    lines = []

    summary = probe_target(make_fetcher(), _target(url=f"{SITE}/report/7", detail_pattern=None), tmp_path, lines.append)

    assert summary == "200 · 링크 1 · 첨부 1"
    assert f"          원문 PDF → {SITE}/files/r7.pdf" in lines
    assert f"{SITE}/files/r7.pdf" not in http.calls

def test_probe_does_not_request_page_when_robots_disallows(tmp_path, make_fetcher, http):
    http.route(f"{SITE}/robots.txt", (200, "User-agent: *\nDisallow: /press\n"))
    lines = []

    summary = probe_target(make_fetcher(), _target(), tmp_path, lines.append)

    assert summary == "robots disallowed"
    assert http.calls == [f"{SITE}/robots.txt"]
    assert not list(tmp_path.iterdir())


def test_probe_skips_detail_disallowed_by_robots(tmp_path, make_fetcher, http):
    http.route(f"{SITE}/robots.txt", (200, "User-agent: *\nDisallow: /press/\nAllow: /press\n"))
    http.route(f"{SITE}/press", (200, LIST_HTML))

    summary = probe_target(make_fetcher(), _target(), tmp_path, lambda line: None)

    assert summary.endswith("상세 2건, 요청 실패")
    assert f"{SITE}/press/101?curPage=1" not in http.calls


def test_probe_shows_non_html_preview(tmp_path, make_fetcher, http):
    http.route(f"{SITE}/robots.txt", (404, ""))
    http.route(f"{SITE}/api", (200, "<LawSearch><totalCnt>3</totalCnt></LawSearch>",
                               {"Content-Type": "application/xml;charset=UTF-8"}))
    lines = []

    summary = probe_target(make_fetcher(), _target(url=f"{SITE}/api"), tmp_path, lines.append)

    assert summary == "200 (HTML 아님)"
    assert "  내용    <LawSearch><totalCnt>3</totalCnt></LawSearch>" in lines


def test_targets_have_unique_keys_and_valid_patterns():
    import re

    keys = [t.key for t in PROBE_TARGETS]
    assert len(keys) == len(set(keys))
    for target in PROBE_TARGETS:
        assert target.url.startswith("https://")
        if target.detail_pattern:
            re.compile(target.detail_pattern)


def test_run_rejects_unknown_target(tmp_path, make_fetcher):
    with pytest.raises(SystemExit):
        run(only=["nope"], out_dir=tmp_path, fetcher=make_fetcher())


def test_run_writes_report(tmp_path, make_fetcher, http):
    http.route("https://apis.data.go.kr/robots.txt", (404, ""))
    http.route("https://apis.data.go.kr/", (200, "Unexpected errors", {"Content-Type": "text/plain"}))

    run(only=["data_go_kr"], out_dir=tmp_path, fetcher=make_fetcher())

    report = (tmp_path / "report.txt").read_text(encoding="utf-8")
    assert "[data_go_kr]" in report
    assert "data_go_kr  200 (HTML 아님)" in report
