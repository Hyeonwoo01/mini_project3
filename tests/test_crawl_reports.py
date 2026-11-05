"""연구기관 보고서 선별 수집. 상세 화면 HTML 은 후보 검색 때 받은 표본의 구조를 줄여 옮긴 것이다."""

import pytest

from collector import crawl_reports as reports
from collector.crawl_reports import CuratedReport, parse_kdi, parse_kiet, parse_kosi
from common.docstore import get_document, open_db

LONG = "대형마트 매출은 증가했고 전통시장 매출 감소는 확인되지 않았다. " * 10   # MIN_TEXT_CHARS 이상

KDI_FOCUS = f"""<html><head><meta property="og:title" content="의무휴업일 평일 전환의 효과 - KDI 한국개발연구원 - 연구 - KDI FOCUS"></head>
<body><header><span>2026.10.23</span></header>
<section id="contents"><strong class="title">KDI FOCUS 의무휴업일 평일 전환의 효과 <span>2026.05.21</span></strong>
<div class="editor-template"><div><p>평일 전환 지역에서 <span style="color:#e67e22;">대형마트 매출</span>은 증가했다.<br/><br/>
<span><strong>I. 연구의 필요성</strong></span></p><p>{LONG}</p></div></div>
<div class="desc sumy03">KDI FOCUS 목차 I. 연구의 필요성</div>
<div class="editor-template"><p>요약 팝업</p></div></section></body></html>"""

KDI_REPORT = """<html><head><meta property="og:title" content="자영업자 고용보험 적용 - KDI 한국개발연구원 - 연구 - 연구 보고서"></head>
<body><section id="contents"><p><span>정책연구시리즈 자영업자 고용보험 적용</span><span>2022.12.31</span></p>
<div class="bundlebox desc sumy01"><button>국문요약 열기</button><div class="editor-template">본 연구는 자영업자 고용보험 적용을 앞두고 """ + "실증분석 결과를 제공한다. " * 30 + """</div></div>
<div class="bundlebox desc sumy01"><button>영문요약 열기</button><div class="editor-template">""" + "This study examines UI. " * 40 + """</div></div>
<div class="bundlebox desc sumy02">목차 열기 발간사 요 약 제1장 서 론</div></section></body></html>"""

KOSI_DETAIL = f"""<html><head><title>중소벤처기업연구원</title></head><body><table>
<tr><th>제목</th><td>소상공인 특화 생존지수 개발</td></tr><tr><th>저자</th><td>정은애</td></tr>
<tr><th>발행일</th><td><span class="table-td-text">2025-12-15</span></td></tr>
<tr><th>요약</th><td>01 연구의 배경 및 목적<br/>□ 연구의 배경<br/>ㅇ {LONG}</td></tr></table></body></html>"""

KIET_ISSUE = f"""<html><head><meta property="og:title" content="소상공인정책 인식 - KIET 산업연구원 - 연구 - 이슈 분석"></head>
<body><div class="page_view">소상공인정책 인식 2016.12.26 홍석일
<div class="p_d-cont"><div>요약</div><div>언론보도</div><p>{LONG}</p><dl><dt>첨부파일</dt><dd>e640.pdf</dd></dl></div></div></body></html>"""

KIET_REPORT = """<html><head><meta property="og:title" content="소상공인협동조합 - KIET 산업연구원 - 연구 - 심층 분석 - 연구보고서"></head>
<body><div class="page_view">소상공인협동조합 2020.12.24 <div class="p_d-cont"><div>요약</div><div>목차</div>
<p>머리말 요약 제1장 서론 1. 연구의 배경 및 목적 제2장 소상공인의 경영난</p></div></div></body></html>"""


def test_kdi_focus_takes_full_text_without_breaking_sentences_at_spans():
    page = parse_kdi(KDI_FOCUS)
    assert (page.title, page.published_at, page.text_kind) == ("의무휴업일 평일 전환의 효과", "2026-05-21", "전문")
    lines = page.text.splitlines()
    assert lines[:2] == ["평일 전환 지역에서 대형마트 매출은 증가했다.", "I. 연구의 필요성"]
    assert "KDI FOCUS 목차" not in page.text and "요약 팝업" not in page.text


def test_kdi_report_takes_korean_summary_not_english_or_toc():
    page = parse_kdi(KDI_REPORT)
    assert (page.published_at, page.text_kind) == ("2022-12-31", "요약")
    assert page.text.startswith("본 연구는 자영업자 고용보험") and "This study" not in page.text


def test_kosi_reads_table_fields_and_longest_cell():
    page = parse_kosi(KOSI_DETAIL)
    assert (page.title, page.published_at) == ("소상공인 특화 생존지수 개발", "2025-12-15")
    assert page.text.startswith("01 연구의 배경 및 목적\n□ 연구의 배경\nㅇ 대형마트")


def test_kiet_issue_summary_drops_tab_labels_and_attachments():
    page = parse_kiet(KIET_ISSUE)
    assert (page.title, page.published_at) == ("소상공인정책 인식", "2016-12-26")
    assert page.text.startswith("대형마트 매출은") and "첨부파일" not in page.text and "e640" not in page.text


def test_kiet_report_with_only_table_of_contents_has_no_summary():
    assert parse_kiet(KIET_REPORT).text == ""


@pytest.fixture
def raw_store(tmp_path, monkeypatch):
    saved = {}

    def fake_save_raw(doc_type, doc_id, content, ext):
        path = tmp_path / "raw" / f"{doc_id}.{ext}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        saved[doc_id] = (doc_type, path)
        return str(path)

    monkeypatch.setattr(reports, "save_raw", fake_save_raw)
    return saved


def _report(site: str, url: str, title: str = "후보 제목") -> CuratedReport:
    return CuratedReport(site, url, title, "유통 규제", "시험")


def test_run_saves_in_period_reports_and_records_exclusions(tmp_path, make_fetcher, http, raw_store):
    host = "https://www.kdi.re.kr"
    http.route(f"{host}/robots.txt", (404, ""))
    http.route(f"{host}/focus/1", (200, KDI_FOCUS))
    http.route(f"{host}/report/2", (200, KDI_REPORT.replace("2022.12.31", "2020.12.31")))    # 기간 밖
    http.route(f"{host}/report/3", (200, KDI_REPORT.replace('class="bundlebox desc sumy01"', 'class="x"')))
    http.route(f"{host}/report/4", (503, "busy"))
    curated = [_report("kdi", f"{host}/focus/1"), _report("kdi", f"{host}/report/2"),
               _report("kdi", f"{host}/report/3"), _report("kdi", f"{host}/report/4")]
    db, review = tmp_path / "docs.db", tmp_path / "collected.md"

    stats = reports.run(reports=curated, fetcher=make_fetcher(), db_path=db, review_path=review)

    assert [r.title for r in stats.saved] == ["의무휴업일 평일 전환의 효과"]
    assert [s.split(" — ")[1] for s in stats.skipped] == ["기간 밖 (2020-12-31)", "상세 화면에 요약 없음 (0자)"]
    assert len(stats.failed) == 1 and "HTTPError" in stats.failed[0]
    [(doc_id, (doc_type, path))] = raw_store.items()
    assert doc_id.startswith("rpt_20260521_") and doc_type.value == "policy_report" and path.suffix == ".html"
    with open_db(db, read_only=True) as conn:
        doc = get_document(conn, doc_id)
    assert (doc["source"], doc["published_at"], doc["url"]) == ("한국개발연구원", "2026-05-21", f"{host}/focus/1")
    assert doc["extra"]["text_kind"] == "전문" and doc["extra"]["topic"] == "유통 규제"
    assert doc["raw_text"].startswith("평일 전환 지역에서")
    assert "## 의무휴업일 평일 전환의 효과" in review.read_text(encoding="utf-8")

    calls = len(http.calls)
    again = reports.run(reports=curated[:1], fetcher=make_fetcher(), db_path=db, review_path=review)
    assert (len(again.saved), len(again.already)) == (0, 1)
    assert f"{host}/focus/1" not in http.calls[calls:]          # 이미 받은 보고서는 다시 요청하지 않는다


def test_dry_run_makes_no_requests(tmp_path, make_fetcher, http, capsys):
    stats = reports.run(dry_run=True, fetcher=make_fetcher(), db_path=tmp_path / "docs.db")
    assert http.calls == [] and stats.saved == []
    assert f"수집 대상 {len(reports.CURATED_REPORTS)}건" in capsys.readouterr().out


def test_curated_list_follows_selection_criteria():
    """선별 기준: 주제별 최대 4건, 총 20건 이하 (README '연구기관 보고서 선별 기준'). 같은 보고서를 두 번 넣지 않는다.
    총 10건 하한은 아직 못 채웠다 (지역화폐·카드수수료 주제 후보 0건, 서울연구원 후보로 보충 예정)."""
    curated = reports.CURATED_REPORTS
    assert len(curated) <= 20
    assert len({r.url for r in curated}) == len(curated)
    topics = [r.topic for r in curated]
    assert max(topics.count(t) for t in set(topics)) <= 4
    assert all(r.site in reports.PARSERS for r in curated)
