import pytest

from common.docstore import open_db, replace_processed_documents, upsert_documents
from common.schema import CollectMethod, DocType, DocumentRecord, ProcessedRecord
from eval.find_docs import search


def _doc(doc_id, title, published_at):
    return DocumentRecord(doc_id=doc_id, doc_type=DocType.NEWS, source="연합뉴스", title=title,
                          url=f"https://x/{doc_id}", collected_at="2026-10-16",
                          collect_method=CollectMethod.PROVIDED, published_at=published_at)


def _processed(doc_id, status, text, scope="national", canonical=None):
    return ProcessedRecord(doc_id=doc_id, status=status, processed_at="2026-11-03",
                           clean_text=text if status != "excluded" else None,
                           canonical_doc_id=None if status == "excluded" else (canonical or doc_id),
                           excluded_reason="off_topic" if status == "excluded" else None, scope=scope)


@pytest.fixture
def conn(tmp_path):
    with open_db(tmp_path / "docs.db") as c:
        upsert_documents(c, [
            _doc("a", "상생페이백 11월분 3916억원 지급", "2025-12-16T09:00:00"),
            _doc("b", "상생페이백 11월분 지급 완료", "2025-12-16T10:00:00"),
            _doc("c", "서울 골목상권 매출 회복", "2023-10-16T09:00:00"),
            _doc("d", "상생페이백 무관 기사", "2025-01-01T09:00:00"),
        ])
        replace_processed_documents(c, [
            _processed("a", "kept", "중기부는 상생페이백 11월분으로 3916억원을 지급했다."),
            _processed("b", "duplicate", "상생페이백 11월분 지급", canonical="a"),
            _processed("c", "kept", "서울 골목상권 외식업 매출이 14% 늘었다.", scope="seoul"),
            _processed("d", "excluded", ""),
        ])
        yield c


def test_search_returns_only_kept_documents_matching_all_terms(conn):
    found = search(conn, ["상생페이백", "11월"])
    assert [d.doc_id for d in found] == ["a"]          # 중복(b)·제외(d)는 검색되지 않음
    assert "3916억원" in found[0].snippet


def test_search_filters_by_scope_and_year(conn):
    assert [d.doc_id for d in search(conn, ["매출"], scope="seoul")] == ["c"]
    assert search(conn, ["매출"], year=2025) == []


def test_search_requires_terms(conn):
    with pytest.raises(ValueError):
        search(conn, [])


def test_show_document_points_duplicates_to_canonical(conn):
    from eval.find_docs import show_document
    assert "대표 문서 a" in show_document(conn, "b")
    assert "정답 문서로 쓸 수 없습니다" in show_document(conn, "d")
    assert "3916억원" in show_document(conn, "a")
