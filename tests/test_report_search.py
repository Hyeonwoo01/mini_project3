"""연구기관 보고서 후보 검색. 목록 HTML 은 2차 probe 표본의 구조를 줄여 옮긴 것이다."""

import csv
import re

import pytest

from collector.report_search import (
    REPORT_SOURCES,
    Candidate,
    ReportSource,
    merge,
    parse_candidates,
    parse_date,
    run,
    search_source,
    write_candidates,
)

SOURCES = {s.key: s for s in REPORT_SOURCES}

# 산업연구원: 항목(li.item) 안에 제목 링크와 발행일
KIET_LIST = """<ul class="list_box">
<li class="item"><div class="rpt_tit"><a href="./issueView?issue_no=812&skey=&sval=&pg=1&pp=10">대형마트 의무휴업 평일 전환의 상권 효과</a>
  <p>홍길동 2025.02.13</p></div><a href="./issueView?issue_no=812&skey=&sval=&pg=1&pp=10">원문 미리보기</a></li>
<li class="item"><div class="rpt_tit"><a href="./issueView?issue_no=800&skey=&sval=&pg=1&pp=10">석유 최고가격제의 효과</a>
  <p>김철수 2021.03.02</p></div></li>
</ul>"""

# 조세연: 한 칸(div)에 여러 항목, 링크 글자에 저자·발간월이 붙는다
KIPF_LIST = """<div class="col">
<a href="#" onclick="javascript: fn_search_detail('527500'); return false;">소비쿠폰의 지역경제 효과 저자 홍길동 발간월 2026-05 관련키워드 자세히보기</a>
<a href="#" onclick="javascript: fn_search_detail('527667'); return false;">재정포럼 2026년 10월호(제364호) 저자 한국조세재정연구원 발간월 2026-10 관련키워드 자세히보기</a>
</div>"""

# KDI: 목록에 발행일이 없다. 이미지 링크(글자 없음)는 건너뛴다
KDI_LIST = """<ul><li><a href="./focusView?pub_no=19187"><img src="x.png"></a>
<a href="./focusView?pub_no=19187">KDI FOCUS 의무휴업일 평일 전환이 시사하는 유통정책의 전환 방향</a> #경쟁정책 12p</li></ul>"""

SI_LIST = """<ul><li><a href="/bbs/view.do?key=2024100039&pstSn=2611030001">서울 골목상권 매출 변화 분석</a></li>
<li><a href="/bbs/view.do?key=2024100039&pstSn=2611030002">서울시 도시교통 수요 전망</a></li></ul>"""


def test_parse_kiet_list_reads_title_and_date_of_each_item():
    cands = parse_candidates(KIET_LIST, SOURCES["kiet_issue"], "의무휴업")
    assert [(c.title, c.published, c.url) for c in cands] == [
        ("대형마트 의무휴업 평일 전환의 상권 효과", "2025-02-13", "https://www.kiet.re.kr/research/issueView?issue_no=812"),
        ("석유 최고가격제의 효과", "2021-03-02", "https://www.kiet.re.kr/research/issueView?issue_no=800"),
    ]
    assert cands[0].keywords == ["의무휴업"]
    assert cands[0].title_topics == ["골목상권·전통시장", "유통 규제"] and cands[0].in_period
    assert cands[1].title_topics == [] and cands[1].in_period is False      # 2021-10-16 이전


def test_parse_kipf_list_cuts_author_tail_and_reads_month():
    cands = parse_candidates(KIPF_LIST, SOURCES["kipf_all"], "소비쿠폰")
    assert [(c.title, c.published) for c in cands] == [("소비쿠폰의 지역경제 효과", "2026-05"),
                                                       ("재정포럼 2026년 10월호(제364호)", "2026-10")]
    assert cands[0].url == "https://www.kipf.re.kr/kor/Publication/All/kiPublish/ALL/view.do?serialNo=527500"
    assert cands[1].in_period is True        # 월 단위 비교: 2026-10 은 기간 끝(2026-10-16)과 같은 달


def test_parse_kdi_list_skips_image_link_and_has_no_date():
    [cand] = parse_candidates(KDI_LIST, SOURCES["kdi_focus"], None)
    assert cand.title == "KDI FOCUS 의무휴업일 평일 전환이 시사하는 유통정책의 전환 방향"
    assert cand.published is None and cand.in_period is None and cand.keywords == []


def test_parse_date_formats():
    assert parse_date("홍길동 2026.5.21") == "2026-05-21"
    assert parse_date("발간월 2026-10") == "2026-10"
    assert parse_date("날짜 없음") is None


def test_merge_combines_keywords_of_same_report():
    a = Candidate("kdi_focus", "KDI", "제목", "u1", None, ["유통"])
    b = Candidate("kdi_focus", "KDI", "제목", "u1", "2026-05-21", ["대형마트", "유통"])
    [merged] = merge([a, b])
    assert merged.keywords == ["유통", "대형마트"] and merged.published == "2026-05-21"
    assert a.keywords == ["유통"]            # 원래 후보는 바꾸지 않는다


def test_write_candidates_puts_topic_matches_first(tmp_path):
    path = tmp_path / "c.tsv"
    write_candidates([Candidate("s", "기관", "무관한 보고서", "u1", "2026-01-01"),
                      Candidate("s", "기관", "소상공인 실태", "u2", "2024-01-01", ["소상공인"]),
                      Candidate("s", "기관", "소상공인 전망", "u3", "2025-01-01", ["소상공인"])], path)
    rows = list(csv.DictReader(path.read_text(encoding="utf-8-sig").splitlines(), delimiter="\t"))
    assert [r["url"] for r in rows] == ["u3", "u2", "u1"]
    assert rows[0]["title_topics"] == "소상공인·자영업" and rows[0]["in_period"] == "Y"


def _robots(http, host):
    http.route(f"{host}/robots.txt", (200, "User-agent: *\nAllow: /\n"))


def test_search_source_saves_samples_and_merges_keywords(tmp_path, make_fetcher, http):
    source = SOURCES["kiet_issue"]
    _robots(http, "https://www.kiet.re.kr")
    for kw in ("대형마트", "유통"):
        http.route(source.search_url(kw), (200, KIET_LIST))
    http.route("https://www.kiet.re.kr/research/issueView?issue_no=812", (200, "<html>상세</html>"))
    lines = []

    cands, stats = search_source(make_fetcher(), source, ["대형마트", "유통"], tmp_path, lines.append)

    assert [c.keywords for c in cands] == [["대형마트", "유통"], ["대형마트", "유통"]]
    assert stats.requests == 3 and not stats.failures
    assert (tmp_path / "kiet_issue_list.html").exists() and (tmp_path / "kiet_issue_detail.html").exists()
    assert lines == ["[kiet_issue] 산업연구원 · 요청 3 · 후보 2 (제목에 주제어 1)"]


def test_search_source_warns_when_site_ignores_keyword(tmp_path, make_fetcher, http):
    source = SOURCES["kiet_issue"]
    _robots(http, "https://www.kiet.re.kr")
    http.route(source.search_url("카드"), (200, KIET_LIST.replace("대형마트 의무휴업 평일 전환의 상권 효과", "반도체 수출")))
    lines = []
    search_source(make_fetcher(), source, ["카드"], tmp_path, lines.append)
    assert any(line.startswith("  경고: 제목에 주제어가 있는 후보가 없다") for line in lines)


def test_unsearchable_board_keeps_only_topic_titles(tmp_path, make_fetcher, http):
    source = SOURCES["si_report"]
    _robots(http, "https://www.si.re.kr")
    http.route(source.search_url(""), (200, SI_LIST))
    cands, stats = search_source(make_fetcher(), source, ["상권", "유통"], tmp_path, lambda line: None)
    assert [c.title for c in cands] == ["서울 골목상권 매출 변화 분석"]
    assert stats.requests == 2       # 검색어마다가 아니라 목록 1쪽 + 첫 상세 표본(없는 주소라 404 실패)


def test_failures_are_reported_and_do_not_stop_the_run(tmp_path, make_fetcher, http):
    source = SOURCES["kdi_focus"]
    http.route("https://www.kdi.re.kr/robots.txt", (200, "User-agent: *\nDisallow: /research/\n"))
    lines = []
    cands, stats = search_source(make_fetcher(), source, ["유통"], tmp_path, lines.append)
    assert cands == [] and len(stats.failures) == 1 and "RobotsDisallowed" in stats.failures[0]
    assert lines[0].endswith("후보 0 (제목에 주제어 0) · 실패 1")


def test_run_writes_candidates_and_log(tmp_path, make_fetcher, http):
    source = SOURCES["kipf_all"]
    _robots(http, "https://www.kipf.re.kr")
    http.route(source.search_url("소비쿠폰"), (200, KIPF_LIST))
    path = run(only=["kipf_all"], keywords=["소비쿠폰"], out_dir=tmp_path, fetcher=make_fetcher())
    rows = list(csv.DictReader(path.read_text(encoding="utf-8-sig").splitlines(), delimiter="\t"))
    assert [r["title"] for r in rows] == ["소비쿠폰의 지역경제 효과", "재정포럼 2026년 10월호(제364호)"]
    assert "후보 2건" in (tmp_path / "search_log.txt").read_text(encoding="utf-8")


def test_run_rejects_unknown_source(tmp_path, make_fetcher):
    with pytest.raises(SystemExit, match="없는 기관 key: nope"):
        run(only=["nope"], out_dir=tmp_path, fetcher=make_fetcher())


@pytest.mark.parametrize("source", REPORT_SOURCES, ids=lambda s: s.key)
def test_sources_have_unique_keys_and_build_urls(source):
    assert [s.key for s in REPORT_SOURCES].count(source.key) == 1
    url = source.search_url("소상공인")
    assert url.startswith("https://")
    if source.searchable:
        assert "%EC%86%8C%EC%83%81%EA%B3%B5%EC%9D%B8" in url      # 검색어가 인코딩되어 들어간다
    assert isinstance(source.item_link, re.Pattern)


def test_kosi_search_url_matches_site_script():
    """사이트 검색 스크립트(searchFormApply)가 만드는 주소와 같아야 한다 (menuFrontURL 값 안의 '?' 포함)."""
    assert SOURCES["kosi_report"].search_url("소상공인") == (
        "https://www.kosi.re.kr/front/functionDisplay?menuFrontNo=59996687477237983229272069890802427549"
        "&menuFrontURL=front/basicResearchList?limit=12&titleMain=%EC%86%8C%EC%83%81%EA%B3%B5%EC%9D%B8")


def test_custom_source_detail_url():
    source = ReportSource("x", "기관", lambda kw: "https://x.kr/list", re.compile(r"id=(\d+)"),
                          lambda m: f"https://x.kr/view?id={m[1]}")
    [cand] = parse_candidates('<a href="/v?id=7">소상공인 보고서</a>', source, None)
    assert cand.url == "https://x.kr/view?id=7"
