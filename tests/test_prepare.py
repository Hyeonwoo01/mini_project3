from datetime import date

import pymupdf

from common.docstore import get_document, open_db, upsert_documents
from common.schema import CollectMethod, DocType, DocumentRecord
from indexer.dedup import DedupConfig
from indexer.prepare import prepare
from indexer.quality import QualityConfig

BODY = (
    "서울 마포구 망원시장 소상공인들은 정책자금 지원이 확대되면서 매출 회복을 기대하고 있다. "
    "구청은 상권 활성화를 위해 이달부터 공동 마케팅을 지원한다. 무단 전재 및 재배포 금지"
) * 2
OTHER_BODY = "부산 해운대구 전통시장 상인회가 온누리상품권 환급 행사를 열어 소상공인 매출이 늘었다. " * 3

QUALITY = QualityConfig(name="t", period_start=date(2024, 10, 16), min_chars={DocType.NEWS: 100},
                        max_chars={DocType.NEWS: 10_000}, lead_chars=300, topic_terms=("소상공인", "상권"))
DEDUP = DedupConfig(name="t", similarity_threshold=0.8)


def _news(doc_id, title, text, published_at="2026-01-10T09:00:00"):
    return DocumentRecord(doc_id=doc_id, doc_type=DocType.NEWS, source="연합뉴스", title=title,
                          url=f"https://n.news.naver.com/{doc_id}", collected_at="2026-10-16",
                          collect_method=CollectMethod.PROVIDED, published_at=published_at, raw_text=text)


def test_prepare_records_every_document_with_status(tmp_path):
    db = tmp_path / "docs.db"
    with open_db(db) as conn:
        upsert_documents(conn, [
            _news("n1", "망원시장 소상공인 지원", BODY),
            _news("n2", "망원시장 소상공인 지원 확대", BODY, published_at="2026-01-11T09:00:00"),  # 재배포
            _news("n3", "부산 전통시장 행사", OTHER_BODY),                                     # 타 지역
            _news("n4", "짧은 기사", "소상공인"),                                               # 본문 과소
            _news("n5", "망원시장 소상공인 지원", BODY, published_at="2023-01-10T09:00:00"),     # 기간 밖
        ])
        stats = prepare(conn, QUALITY, DEDUP)

    assert stats.total == 5
    assert stats.statuses == {"kept": 1, "duplicate": 1, "excluded": 3}
    assert stats.excluded_reasons == {"outside_seoul": 1, "too_short": 1, "out_of_period": 1}
    assert stats.kept_by_scope == {"seoul": 1}

    with open_db(db, read_only=True) as conn:
        kept = [get_document(conn, i) for i in ("n1", "n2") if get_document(conn, i)["status"] == "kept"][0]
        dup = get_document(conn, "n2" if kept["doc_id"] == "n1" else "n1")
        excluded = get_document(conn, "n3")

    assert dup["status"] == "duplicate" and dup["canonical_doc_id"] == kept["doc_id"]
    assert "무단" not in kept["clean_text"]                      # 보일러플레이트 제거
    assert kept["scope"] == "seoul"
    assert excluded["clean_text"] is None and excluded["excluded_reason"] == "outside_seoul"


def test_prepare_without_quality_filter_and_rerun_replaces_rows(tmp_path):
    db = tmp_path / "docs.db"
    with open_db(db) as conn:
        upsert_documents(conn, [_news("n3", "부산 전통시장 행사", OTHER_BODY), _news("n4", "짧은 기사", "소상공인")])
        prepare(conn, QUALITY, DEDUP)
        stats = prepare(conn, None, DEDUP)
        assert conn.execute("SELECT COUNT(*) FROM processed_documents").fetchone()[0] == 2
    assert stats.quality_config == "none"
    assert stats.statuses == {"kept": 2}


def test_prepare_builds_keyword_index_for_kept_documents_only(tmp_path):
    db = tmp_path / "docs.db"
    with open_db(db) as conn:
        upsert_documents(conn, [
            _news("n1", "망원시장 소상공인 지원", BODY),
            _news("n2", "망원시장 소상공인 지원 확대", BODY, published_at="2026-01-11T09:00:00"),  # 중복
            _news("n3", "부산 전통시장 행사", OTHER_BODY),                                     # 제외
        ])
        stats = prepare(conn, QUALITY, DEDUP)
        indexed = {row[0] for row in conn.execute("SELECT doc_id FROM keyword_index")}
        kept = {row[0] for row in conn.execute("SELECT doc_id FROM processed_documents WHERE status = 'kept'")}
    assert indexed == kept and len(indexed) == stats.keyword_indexed == 1
    assert "키워드 색인: 1건" in stats.summary()


def _press(doc_id, raw_path):
    return DocumentRecord(doc_id=doc_id, doc_type=DocType.PRESS_RELEASE, source="중소벤처기업부",
                          title=f"보도자료 {doc_id}", url=f"https://www.korea.kr/briefing/pressReleaseView.do?newsId={doc_id}",
                          collected_at="2026-11-03T08:00:00+00:00", collect_method=CollectMethod.CRAWL,
                          published_at="2026-10-14", raw_path=raw_path)


def test_prepare_reads_press_release_pdf_and_survives_broken_file(tmp_path):
    pdf = pymupdf.open()
    page = pdf.new_page()
    for i in range(6):
        page.insert_text((72, 72 + 24 * i), "소상공인 정책자금 지원 확대 안내 보도자료 본문입니다", fontname="korea")
    good = tmp_path / "good.pdf"
    pdf.save(good)
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"not a pdf")
    quality = QualityConfig(name="t", period_start=date(2024, 10, 16),
                            min_chars={DocType.NEWS: 100, DocType.PRESS_RELEASE: 100},
                            lead_chars=300, topic_terms=("소상공인",))

    db = tmp_path / "docs.db"
    with open_db(db) as conn:
        upsert_documents(conn, [_press("p1", str(good)), _press("p2", str(broken))])
        stats = prepare(conn, quality, DEDUP)

    assert stats.statuses == {"kept": 1, "excluded": 1}
    assert stats.excluded_reasons == {"too_short": 1}
    assert len(stats.parse_errors) == 1 and stats.parse_errors[0].startswith("p2: ")
    assert "원본 파싱 실패: 1" in stats.summary()
    with open_db(db, read_only=True) as conn:
        assert "정책자금 지원 확대" in get_document(conn, "p1")["clean_text"]
