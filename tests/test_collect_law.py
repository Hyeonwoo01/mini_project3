"""법령 수집. 응답 XML 은 DRF lawSearch·lawService 의 구조를 줄여 옮긴 것이다 (실제 응답으로 다시 확인할 것)."""

import pytest

from collector import collect_law as law
from collector.collect_law import LawApiError, LawTarget, article_label, parse_law, parse_search
from common.docstore import get_document, open_db

SEARCH = """<?xml version="1.0" encoding="UTF-8"?><LawSearch><target>law</target><totalCnt>2</totalCnt>
<law id="1"><법령일련번호>270001</법령일련번호><현행연혁코드>현행</현행연혁코드><법령명한글>여신전문금융업법</법령명한글>
<법령구분명>법률</법령구분명><소관부처명>금융위원회</소관부처명><시행일자>20250101</시행일자></law>
<law id="2"><법령일련번호>270002</법령일련번호><현행연혁코드>현행</현행연혁코드><법령명한글>여신전문금융업법 시행령</법령명한글>
<법령구분명>대통령령</법령구분명><시행일자>20250101</시행일자></law></LawSearch>"""

BODY = """<?xml version="1.0" encoding="UTF-8"?><법령 법령키="0001"><기본정보><법령명_한글>여신전문금융업법</법령명_한글>
<시행일자>20250101</시행일자><소관부처 소관부처코드="1">금융위원회</소관부처></기본정보><조문>
<조문단위 조문키="0001000"><조문번호>1</조문번호><조문여부>전문</조문여부><조문내용>제1장 총칙</조문내용></조문단위>
<조문단위 조문키="0001001"><조문번호>1</조문번호><조문여부>조문</조문여부><조문제목>목적</조문제목>
<조문내용>제1조(목적) 이 법은 여신전문금융업을 건전하게 육성한다.</조문내용></조문단위>
<조문단위 조문키="0005001"><조문번호>5</조문번호><조문여부>조문</조문여부><조문내용>제5조 삭제 &lt;2015. 1. 20.&gt;</조문내용></조문단위>
<조문단위 조문키="0018003"><조문번호>18</조문번호><조문가지번호>3</조문가지번호><조문여부>조문</조문여부>
<조문제목>가맹점수수료율의 차별금지 등</조문제목><조문내용>제18조의3(가맹점수수료율의 차별금지 등)</조문내용>
<항><항번호>①</항번호><항내용>① 신용카드업자는 가맹점수수료율을 정할 때 차별해서는 아니 된다.</항내용></항>
<항><항번호>③</항번호><항내용>③ 다음 각 호의 가맹점에는 우대수수료율을 적용한다.</항내용>
<호><호번호>1.</호번호><호내용>1. 영세한 중소신용카드가맹점</호내용>
<목><목번호>가.</목번호><목내용>가. 연매출 3억원 이하</목내용></목></호></항></조문단위>
</조문></법령>"""

TARGET = LawTarget("여신전문금융업법", "우대수수료율 근거", articles=("18의3",))


def test_search_picks_exact_current_law_not_decree():
    info = parse_search(SEARCH.encode(), "여신전문금융업법")
    assert (info.mst, info.kind, info.enforced_at, info.ministry) == ("270001", "법률", "2025-01-01", "금융위원회")
    assert parse_search(SEARCH.encode(), "없는 법") is None


@pytest.mark.parametrize("body", [b"<html><body>error</body></html>",
                                  "<result><msg>사용자 정보 검증에 실패하였습니다.</msg></result>".encode()])
def test_search_error_pages_raise(body):
    with pytest.raises(LawApiError):
        parse_search(body, "여신전문금융업법")


def test_body_keeps_articles_with_clauses_and_drops_headings_and_deleted():
    info, articles = parse_law(BODY.encode())
    assert info == {"name": "여신전문금융업법", "enforced_at": "2025-01-01", "ministry": "금융위원회"}
    assert [a.key for a in articles] == ["1", "18의3"]
    assert articles[1].text.splitlines() == [
        "제18조의3(가맹점수수료율의 차별금지 등)", "① 신용카드업자는 가맹점수수료율을 정할 때 차별해서는 아니 된다.",
        "③ 다음 각 호의 가맹점에는 우대수수료율을 적용한다.", "1. 영세한 중소신용카드가맹점", "가. 연매출 3억원 이하"]
    assert (article_label("18의3"), article_label("1")) == ("제18조의3", "제1조")


@pytest.fixture
def raw_store(tmp_path, monkeypatch):
    saved = {}

    def fake_save_raw(doc_type, doc_id, content, ext):
        path = tmp_path / "raw" / f"{doc_id}.{ext}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        saved[doc_id] = content
        return str(path)

    monkeypatch.setattr(law, "save_raw", fake_save_raw)
    return saved


def _routes(http, body=BODY):
    http.route("https://www.law.go.kr/robots.txt", (404, ""))
    http.route(f"{law.API_BASE}/lawSearch.do", (200, SEARCH))
    http.route(f"{law.API_BASE}/lawService.do", (200, body))


def test_run_saves_selected_articles_and_replaces_them_after_amendment(tmp_path, make_fetcher, http, raw_store, capsys):
    _routes(http, BODY.replace("<시행일자>20250101</시행일자><소관부처", "<시행일자>20250101</시행일자><비고>SECRETOC</비고><소관부처"))
    db, review = tmp_path / "docs.db", tmp_path / "collected.md"
    stats = law.run(targets=[TARGET], fetcher=make_fetcher(), db_path=db, review_path=review, oc="SECRETOC")
    assert stats.saved == {"여신전문금융업법": 1} and not stats.failed
    assert all(b"SECRETOC" not in content for content in raw_store.values())     # 저장 원본에서 인증값을 가린다
    with open_db(db, read_only=True) as conn:
        [(doc_id,)] = conn.execute("SELECT doc_id FROM documents").fetchall()
        doc = get_document(conn, doc_id)
    assert doc_id.startswith("law_20250101_") and doc["doc_type"] == "law" and doc["collect_method"] == "api"
    assert doc["title"] == "여신전문금융업법 제18조의3(가맹점수수료율의 차별금지 등)"
    assert doc["source"] == "국가법령정보센터 (금융위원회)" and doc["published_at"] == "2025-01-01"
    assert doc["extra"]["article"] == "18의3" and "%EC%A0%9C18%EC%A1%B0%EC%9D%983" in doc["url"]
    assert "## 여신전문금융업법 제18조의3" in review.read_text(encoding="utf-8")

    _routes(http, BODY.replace("20250101", "20260701"))                    # 개정: 시행일이 바뀜
    again = law.run(targets=[TARGET], fetcher=make_fetcher(), db_path=db, review_path=review, oc="SECRETOC")
    with open_db(db, read_only=True) as conn:
        ids = [row[0] for row in conn.execute("SELECT doc_id FROM documents")]
    assert again.replaced == 1 and len(ids) == 1 and ids[0].startswith("law_20260701_")
    assert "SECRETOC" not in capsys.readouterr().out


def test_api_error_is_reported_without_the_key(tmp_path, make_fetcher, http, raw_store, capsys):
    http.route("https://www.law.go.kr/robots.txt", (404, ""))
    http.route(f"{law.API_BASE}/lawSearch.do", (200, "<html>OC=SECRETOC 미등록 IP</html>"))
    stats = law.run(targets=[TARGET], fetcher=make_fetcher(), db_path=tmp_path / "docs.db",
                    review_path=tmp_path / "c.md", oc="SECRETOC")
    assert len(stats.failed) == 1 and "SECRETOC" not in stats.failed[0] and "***" in stats.failed[0]
    assert "SECRETOC" not in capsys.readouterr().out


def test_prune_removes_laws_dropped_from_the_list(tmp_path, make_fetcher, http, raw_store):
    _routes(http)
    db = tmp_path / "docs.db"
    law.run(targets=[TARGET], fetcher=make_fetcher(), db_path=db, review_path=tmp_path / "c.md", oc="K")
    with open_db(db) as conn:
        assert law.prune_unlisted(conn, [LawTarget("다른 법", "")]) == 1
        assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0


def test_dry_run_and_missing_key(tmp_path, http, capsys, monkeypatch):
    law.run(dry_run=True)
    assert http.calls == [] and f"수집 대상 {len(law.CURATED_LAWS)}개 법령" in capsys.readouterr().out
    monkeypatch.setenv("LAW_OC", "")
    monkeypatch.setattr("common.config.load_dotenv", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="LAW_OC"):
        law.run(targets=[TARGET], db_path=tmp_path / "docs.db")


def test_curated_list_is_three_to_five_laws():
    names = [t.name for t in law.CURATED_LAWS]
    assert 3 <= len(names) <= 5 and len(set(names)) == len(names)
