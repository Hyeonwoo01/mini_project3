import csv
import unicodedata

import pytest

from collector.load_provided import extraction_date, find_provided_files, load_files
from common.docstore import get_document, open_db

HEADER = ["제목", "본문전문", "원문URL", "발행일", "매체명", "문서종류", "수집키워드"]
BUNDLE = "01_소상공인_묶음_소상공인_3건_2021-10-16~2026-10-16.csv"
LOCAL_CURRENCY = "05_공공정책_키워드_지역화폐_2건_2021-10-16~2026-10-16.csv"


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(rows)


def _row(title, url, published="2026-01-08T00:07:00", source="강원일보", keyword="골목상권",
         body="본문 첫 줄\n\"인용\" 포함 둘째 줄", doc_type="news"):
    return [title, body, url, published, source, doc_type, keyword]


@pytest.fixture
def files(tmp_path):
    bundle = tmp_path / BUNDLE
    _write_csv(bundle, [
        _row("기사 A", "https://n.news.naver.com/a"),
        _row("기사 B", "https://n.news.naver.com/b", source=""),
        _row("", "https://n.news.naver.com/c"),                      # 제목 없음
        _row("기사 A 재수록", "https://n.news.naver.com/a"),          # 같은 파일 안의 같은 기사
    ])
    local = tmp_path / LOCAL_CURRENCY
    _write_csv(local, [
        _row("기사 A", "https://n.news.naver.com/a", keyword="지역화폐"),  # 파일 간 같은 기사
        _row("기사 D", "https://n.news.naver.com/d", published="2026/01/08"),  # 발행일 형식 오류
    ])
    return tmp_path, [bundle, local]


def test_load_files_stats_and_records(files):
    tmp_path, paths = files
    with open_db(tmp_path / "docs.db") as conn:
        stats = load_files(paths, conn)

    assert stats.rows_read[BUNDLE] == 4
    assert stats.rows_read[LOCAL_CURRENCY] == 2
    assert stats.skipped == {"제목 또는 URL 없음": 1, "발행일 형식 오류": 1}
    assert stats.duplicate_rows_in_file == 1
    assert stats.merged_across_files == 1
    assert stats.source_missing == 1
    assert stats.loaded == 2

    with open_db(tmp_path / "docs.db", read_only=True) as conn:
        assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 2
        rows = {r["title"]: dict(r) for r in conn.execute("SELECT * FROM documents")}
        doc_a = get_document(conn, rows["기사 A"]["doc_id"])
        doc_b = get_document(conn, rows["기사 B"]["doc_id"])

    assert doc_a["keyword"] == "골목상권; 지역화폐"
    assert doc_a["extra"]["provided_files"] == sorted([BUNDLE, LOCAL_CURRENCY])
    assert doc_a["collected_at"] == "2026-10-16"
    assert doc_a["published_at"] == "2026-01-08T00:07:00"
    assert doc_a["raw_text"] == "본문 첫 줄\n\"인용\" 포함 둘째 줄"
    assert doc_b["source"] == "매체 미상"
    assert doc_b["extra"]["source_missing"] is True


def test_reload_is_idempotent(files):
    tmp_path, paths = files
    for _ in range(2):
        with open_db(tmp_path / "docs.db") as conn:
            load_files(paths, conn)
    with open_db(tmp_path / "docs.db", read_only=True) as conn:
        assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 2


def test_missing_column_fails(tmp_path):
    path = tmp_path / BUNDLE
    path.write_text("제목,본문전문\n가,나\n", encoding="utf-8-sig")
    with open_db(tmp_path / "docs.db") as conn, pytest.raises(ValueError, match="필수 컬럼"):
        load_files([path], conn)


def test_find_provided_files_matches_nfd_names(tmp_path):
    (tmp_path / unicodedata.normalize("NFD", BUNDLE)).touch()
    (tmp_path / "02_AI반도체_묶음_AI반도체_1건_2021-10-16~2026-10-16.csv").touch()
    found = find_provided_files(tmp_path)
    assert len(found) == 1
    with pytest.raises(SystemExit):
        find_provided_files(tmp_path / "empty_dir_does_not_exist")


def test_extraction_date_requires_original_name(tmp_path):
    assert extraction_date(tmp_path / BUNDLE) == "2026-10-16"
    with pytest.raises(ValueError):
        extraction_date(tmp_path / "renamed.csv")
