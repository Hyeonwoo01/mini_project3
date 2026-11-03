import pytest

from collector import crawl_korea_kr as kr
from common.docstore import get_document, open_db

ROBOTS = "https://www.korea.kr/robots.txt"


def _item(news_id, title, date, ministry):
    return f"""<li><a href="/briefing/pressReleaseView.do?newsId={news_id}&amp;pageIndex=1&amp;srchWord=&amp;period=">
      <span class="text"><strong>{title}</strong><span class="lead">본문 앞부분</span>
      <span class="source"><span>{date}</span><span>{ministry}</span></span></span></a></li>"""


def list_html(items, last_page=None, total=None):
    paging = ""
    if last_page:
        paging = (f'<div class="paging"><a class="first" href="#pageLink" onclick="pageLink(1); return false;"></a>'
                  f'<span class="num"><a href="#pageLink" onclick="pageLink(2); return false;">2</a></span>'
                  f'<a class="last" href="#pageLink" onclick="pageLink({last_page}); return false;"></a></div>')
    result = f'<div class="result">검색결과 총 <strong>{total:,}</strong>건</div>' if total is not None else ""
    return f'<html><body>{result}<div class="list_type"><ul>{"".join(items)}</ul></div>{paging}</body></html>'


def detail_html(files):
    rows = "".join(
        f"""<p><span><a href="/common/download.do?fileId={file_id}&amp;tblKey=GMN"><img alt="파일">{name}</a></span>
        <span><a class="view" href="/common/docViewer.do?fileId={file_id}&amp;tblKey=GMN">바로보기</a>
        <a class="down" href="/common/download.do?fileId={file_id}&amp;tblKey=GMN">내려받기</a></span></p>"""
        for file_id, name in files)
    return f'<html><body><div class="filedown"><dl><dt>첨부파일</dt><dd>{rows}</dd></dl></div></body></html>'


def test_parse_list_reads_items_paging_and_total():
    page = kr.parse_list(list_html([
        _item(101, "[보도자료] 소상공인 지원 확대", "2026.10.14", "중소벤처기업부"),
        _item(102, "카드수수료 인하", "2025.02.13", "금융위원회"),
    ], last_page=275, total=2741))

    assert page.items == [
        kr.ListItem("101", "소상공인 지원 확대", "2026-10-14", "중소벤처기업부"),
        kr.ListItem("102", "카드수수료 인하", "2025-02-13", "금융위원회"),
    ]
    assert page.items[0].url == "https://www.korea.kr/briefing/pressReleaseView.do?newsId=101"
    assert (page.last_page, page.total) == (275, 2741)


def test_parse_list_without_search_or_paging():
    page = kr.parse_list(list_html([_item(1, "제목", "2026.11.03", "외교부")]))
    assert (page.last_page, page.total) == (1, None)


def test_parse_attachments_keeps_file_links_only():
    attachments = kr.parse_attachments(detail_html([(11, "보도자료.hwpx"), (12, "보도자료.PDF")]))
    assert [(a.name, a.url, a.is_pdf) for a in attachments] == [
        ("보도자료.hwpx", "https://www.korea.kr/common/download.do?fileId=11&tblKey=GMN", False),
        ("보도자료.PDF", "https://www.korea.kr/common/download.do?fileId=12&tblKey=GMN", True),
    ]


@pytest.fixture
def raw_store(tmp_path, monkeypatch):
    saved = {}

    def fake_save_raw(doc_type, doc_id, content, ext):
        path = tmp_path / "raw" / doc_type.value / f"{doc_id}.{ext}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        saved[doc_id] = path
        return str(path)

    monkeypatch.setattr(kr, "save_raw", fake_save_raw)
    return saved


def _view(news_id):
    return f"https://www.korea.kr/briefing/pressReleaseView.do?newsId={news_id}"


def _download(file_id):
    return f"https://www.korea.kr/common/download.do?fileId={file_id}&tblKey=GMN"


def test_run_saves_pdf_for_wanted_ministry_and_resumes(tmp_path, make_fetcher, http, raw_store):
    db = tmp_path / "docs.db"
    http.route(ROBOTS, (404, ""))
    http.route(kr.LIST_URL, (200, list_html([
        _item(101, "[보도자료] 소상공인 지원 확대", "2026.10.14", "중소벤처기업부"),
        _item(102, "외교 일정", "2026.10.14", "외교부"),
    ], total=2)))
    http.route(_view(101), (200, detail_html([(11, "원문.hwpx"), (12, "원문.pdf")])))
    http.route(_download(12), (200, b"%PDF-1.7 fake", {"Content-Type": "application/pdf"}))

    stats = kr.run(["소상공인"], ministries=["중소벤처기업부"], fetcher=make_fetcher(), db_path=db)

    assert (stats.saved, stats.skipped_ministry, stats.failed, stats.no_pdf) == (1, 1, [], [])
    assert _view(102) not in http.calls and _download(11) not in http.calls
    doc_id = next(iter(raw_store))
    with open_db(db, read_only=True) as conn:
        doc = get_document(conn, doc_id)
    assert doc["doc_type"] == "press_release" and doc["source"] == "중소벤처기업부"
    assert doc["title"] == "소상공인 지원 확대" and doc["published_at"] == "2026-10-14"
    assert doc["url"] == _view(101) and doc["keyword"] == "소상공인" and doc["collect_method"] == "crawl"
    assert doc["raw_path"] == str(raw_store[doc_id]) and doc["raw_text"] is None
    assert doc["extra"]["attachment_name"] == "원문.pdf" and doc["extra"]["license"].startswith("공공누리 제1유형")
    assert doc["extra"]["attachment_url"] == _download(12)

    calls_before = len(http.calls)
    again = kr.run(["소상공인"], ministries=["중소벤처기업부"], fetcher=make_fetcher(), db_path=db)
    assert (again.saved, again.already) == (0, 1)
    assert _view(101) not in http.calls[calls_before:]          # 이미 받은 문서는 상세도 다시 열지 않는다


def test_run_records_missing_pdf_and_bad_download(tmp_path, make_fetcher, http, raw_store):
    http.route(ROBOTS, (404, ""))
    http.route(kr.LIST_URL, (200, list_html([
        _item(201, "한글 첨부만", "2026.01.02", "행정안전부"),
        _item(202, "PDF 대신 오류 화면", "2026.01.03", "행정안전부"),
    ])))
    http.route(_view(201), (200, detail_html([(21, "원문.hwp")])))
    http.route(_view(202), (200, detail_html([(22, "원문.pdf")])))
    http.route(_download(22), (200, "<html>오류</html>"))

    stats = kr.run(["지역사랑상품권"], fetcher=make_fetcher(), db_path=tmp_path / "docs.db")

    assert stats.saved == 0 and not raw_store
    assert len(stats.no_pdf) == 1 and "한글 첨부만" in stats.no_pdf[0]
    assert len(stats.failed) == 1 and "PDF 가 아닌 응답" in stats.failed[0]


def test_run_follows_pages_up_to_last_and_max_pages(tmp_path, make_fetcher, http, raw_store):
    http.route(ROBOTS, (404, ""))
    page = list_html([], last_page=3, total=30)
    http.route(kr.LIST_URL, (200, page), (200, page), (200, page))

    stats = kr.run(["상권"], fetcher=make_fetcher(), db_path=tmp_path / "a.db")
    assert stats.pages == 3

    limited = kr.run(["상권"], max_pages=2, fetcher=make_fetcher(), db_path=tmp_path / "b.db")
    assert limited.pages == 2


def test_count_only_reads_first_page(tmp_path, make_fetcher, http, raw_store, capsys):
    http.route(ROBOTS, (404, ""))
    http.route(kr.LIST_URL, (200, list_html([_item(1, "제목", "2026.01.01", "중소벤처기업부")], last_page=9, total=88)))

    stats = kr.run(["소상공인"], count_only=True, fetcher=make_fetcher(), db_path=tmp_path / "docs.db")

    assert (stats.pages, stats.listed) == (1, 0)
    assert http.calls.count(kr.LIST_URL) == 1
    assert "검색 결과 88건 · 9쪽" in capsys.readouterr().out
