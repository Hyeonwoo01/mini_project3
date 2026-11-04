import pytest

from collector import crawl_seoul_economy as seoul
from common.docstore import get_document, open_db

SITE = "https://news.seoul.go.kr"
LIST = seoul.DEFAULT_LIST_URL


def _post(post_id, title, date):
    url = f"{SITE}/economy/archives/{post_id}?listPage=1"
    return f"""<div id="post-{post_id}" class="child_policyDL"><div class="child_policyDL_R">
      <h3 class="tit"><a href="{url}" title="{title}" rel="bookmark">{title}</a></h3>
      <span class="time">등록일 : {date}<span class="part"><a href="#">새소식</a></span></span>
      <div class="topicCont_1"><a href="{url}">{title}</a></div></div></div>"""


def list_html(posts, last_page=None):
    paging = ""
    if last_page:
        paging = ('<div class="paging"><span aria-current="page" class="page-numbers current">1</span>'
                  '<a class="page-numbers" href="?paged=2">2</a><span class="page-numbers dots">&hellip;</span>'
                  f'<a class="page-numbers" href="?paged={last_page}">{last_page}</a>'
                  '<a class="next page-numbers" href="?paged=2"><img alt="다음페이지"></a></div>')
    return f"<html><body>{''.join(posts)}{paging}</body></html>"


# 실제 페이지처럼 #post_content 가 닫히지 않아 태그 목록·공공누리·만족도 조사가 그 안에 들어간 구조
DETAIL = """<html><head><meta property="og:title" content="제목"></head><body>
<dl class="top-row row1"><dt>담당부서</dt><dd class="dept"><span>민생노동국</span><span>소상공인정책과</span></dd>
<dt>수정일</dt><dd class="date">2026-11-03</dd></dl>
<div id="view_ct"><script>const key = "x";</script>
<div id='post_content' class="post-content">
<!-- wp본문시작 -->
<p><strong><span>2026 서울시 소상공인의 날</span></strong></p>
<p><span>📅 일시: 2026년 11월 6일(금)</span><br /><span>📍 장소: 서울광장</span></p>
<div class="tagDiv"><ul><li><a href="#">서울광장</a></li></ul></div>
<div id='sns_elem'>좋아요 인쇄</div>
<div id="content_manager_info">본 저작물은 "공공누리" 제4유형 조건에 따라 이용 할 수 있습니다.</div>
<p>페이지 만족도 평가</p>
</body></html>"""


def test_parse_list_reads_posts_and_last_page():
    items, last_page = seoul.parse_list(list_html(
        [_post(575088, "2026 서울시 소상공인의 날 기념 행사 안내", "2026-11-02"), _post(1, "날짜 이상", "미정")],
        last_page=12))
    assert items[0] == seoul.ListItem(f"{SITE}/economy/archives/575088", "2026 서울시 소상공인의 날 기념 행사 안내",
                                      "2026-11-02")
    assert items[1].published_at is None
    assert last_page == 12


def test_parse_detail_stops_before_tags_and_footer():
    detail = seoul.parse_detail(DETAIL)
    assert detail.text == "2026 서울시 소상공인의 날\n📅 일시: 2026년 11월 6일(금)\n📍 장소: 서울광장"
    assert (detail.dept, detail.modified_at) == ("민생노동국 소상공인정책과", "2026-11-03")


def test_parse_detail_requires_body():
    with pytest.raises(ValueError):
        seoul.parse_detail("<html><body><p>없음</p></body></html>")


@pytest.fixture
def raw_store(tmp_path, monkeypatch):
    saved = {}

    def fake_save_raw(doc_type, doc_id, content, ext):
        path = tmp_path / "raw" / f"{doc_id}.{ext}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        saved[doc_id] = (doc_type, path)
        return str(path)

    monkeypatch.setattr(seoul, "save_raw", fake_save_raw)
    return saved


def test_run_saves_posts_in_period_and_resumes(tmp_path, make_fetcher, http, raw_store):
    db = tmp_path / "docs.db"
    http.route(f"{SITE}/robots.txt", (404, ""))
    http.route(LIST, (200, list_html([_post(575088, "소상공인의 날 행사", "2026-11-02"),
                                      _post(100, "오래된 공지", "2020-05-01")])))
    http.route(f"{SITE}/economy/archives/575088", (200, DETAIL))

    stats = seoul.run(fetcher=make_fetcher(), db_path=db)

    assert (stats.saved, stats.out_of_period, stats.failed) == (1, 1, [])
    assert f"{SITE}/economy/archives/100" not in http.calls
    doc_id, (doc_type, path) = next(iter(raw_store.items()))
    assert doc_type.value == "announcement" and path.suffix == ".html"
    with open_db(db, read_only=True) as conn:
        doc = get_document(conn, doc_id)
    assert doc["source"] == "서울특별시" and doc["title"] == "소상공인의 날 행사"
    assert doc["url"] == f"{SITE}/economy/archives/575088" and doc["published_at"] == "2026-11-02"
    assert doc["raw_text"].startswith("2026 서울시 소상공인의 날") and "공공누리" not in doc["raw_text"]
    assert doc["extra"]["dept"] == "민생노동국 소상공인정책과" and doc["extra"]["license"].startswith("공공누리 제4유형")

    calls = len(http.calls)
    again = seoul.run(fetcher=make_fetcher(), db_path=db)
    assert (again.saved, again.already) == (0, 1)
    assert f"{SITE}/economy/archives/575088" not in http.calls[calls:]


def test_run_follows_paging_dry_run_and_reports_missing_date(tmp_path, make_fetcher, http, raw_store):
    http.route(f"{SITE}/robots.txt", (404, ""))
    page = list_html([_post(1, "공지", "미정")], last_page=3)
    http.route(LIST, (200, page), (200, page), (200, page))

    planned = seoul.run(dry_run=True, fetcher=make_fetcher(), db_path=tmp_path / "a.db")
    assert planned.pages == 3 and len(planned.planned) == 3 and not raw_store

    stats = seoul.run(max_pages=1, fetcher=make_fetcher(), db_path=tmp_path / "b.db")
    assert stats.pages == 1 and len(stats.failed) == 1 and "등록일" in stats.failed[0]


def test_run_does_not_refetch_post_with_empty_body(tmp_path, make_fetcher, http, raw_store):
    db = tmp_path / "docs.db"
    http.route(f"{SITE}/robots.txt", (404, ""))
    http.route(LIST, (200, list_html([_post(7, "포스터만 있는 공지", "2025-04-01")])))
    http.route(f"{SITE}/economy/archives/7", (200, "<html><body><div id='post_content'><p><img src='a.png'></p>"
                                                    "<div class='tagDiv'>태그</div></div></body></html>"))

    first = seoul.run(fetcher=make_fetcher(), db_path=db)
    calls = len(http.calls)
    again = seoul.run(fetcher=make_fetcher(), db_path=db)

    assert first.saved == 1 and (again.saved, again.already) == (0, 1)
    assert f"{SITE}/economy/archives/7" not in http.calls[calls:]
