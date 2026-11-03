from datetime import date

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
