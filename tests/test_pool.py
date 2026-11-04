from pathlib import Path
import json

from eval.pool import (PoolEntry, answer_numbers, apply_judgments, collect_pool, parse_judgments, snippet,
                       write_pool)


def test_answer_numbers_skip_years_and_short_numbers():
    key = "2026-01-20 발표 누적 1조3060억원, 1170만명, 평균 11만1570원, 폐업률 8.64%"
    assert answer_numbers(key) == ["3060", "1170", "1570", "8.64"]


def test_snippet_centers_on_answer_number():
    text = "가" * 300 + " 폐업률은 8.64%로 하락했다. " + "나" * 300
    body, found = snippet(text, ["8.64", "9.04"])
    assert "8.64%" in body and found == ["8.64"] and body.startswith("…")
    lead, none = snippet("짧은 본문", ["9.04"])
    assert lead == "짧은 본문" and none == []


def _results(tmp_path, name, rows):
    d = tmp_path / name
    d.mkdir()
    (d / "retrieval.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return d


def test_collect_pool_skips_gold_judged_and_unanswerable(tmp_path):
    hit = lambda doc_id, rank: {"doc_id": doc_id, "rank": rank}  # noqa: E731
    a = _results(tmp_path, "20261103-1615_base", [
        {"id": "f001", "retrieved": [hit("g", 1), hit("x", 2), hit("y", 3), hit("z", 4)]},
        {"id": "u001", "retrieved": [hit("w", 1)]},
    ])
    b = _results(tmp_path, "20261104-0900_hybrid", [{"id": "f001", "retrieved": [hit("x", 1), hit("n", 2)]}])
    pool = collect_pool([a, b], top=3, gold={"f001": {"g"}, "u001": set()}, judged={"f001": ["y"]})
    assert set(pool) == {"f001"}                                     # 답변 불가 문항은 판정하지 않음
    entries = {e.doc_id: e.hits for e in pool["f001"]}
    assert entries == {"x": ["base 2위", "hybrid 1위"], "n": ["hybrid 2위"]}   # g 는 정답, y 는 판정됨, z 는 4위


def test_write_parse_and_apply_judgments(tmp_path):
    from common.docstore import open_db, replace_processed_documents, upsert_documents
    from common.schema import CollectMethod, DocType, DocumentRecord, ProcessedRecord
    from eval.run_eval import EvalItem
    db = tmp_path / "docs.db"
    with open_db(db) as conn:
        upsert_documents(conn, [DocumentRecord(doc_id=d, doc_type=DocType.NEWS, source="연합뉴스", title=f"{d} 기사",
                                               url=f"https://x/{d}", collected_at="2026-10-16",
                                               collect_method=CollectMethod.PROVIDED, published_at="2026-06-30T09:00:00")
                                for d in ("x", "n")])
        replace_processed_documents(conn, [ProcessedRecord(doc_id=d, status="kept", processed_at="t", clean_text=t,
                                                           canonical_doc_id=d)
                                           for d, t in (("x", "폐업률은 8.64%로 0.40%포인트 하락"), ("n", "무관한 기사"))])
    item = EvalItem(id="f018", q="폐업률은?", type="fact", gold_doc_ids=("g",), answer_key="폐업률 8.64%로 하락")
    out = tmp_path / "pool.md"
    assert write_pool(out, {"f018": [PoolEntry("x", ["base 2위"]), PoolEntry("n", ["base 3위"])]}, [item],
                      [tmp_path / "20261103-1615_base"], top=3, db_path=db) == 2
    text = out.read_text(encoding="utf-8")
    assert "본문의 answer_key 수치: 8.64" in text and "본문의 answer_key 수치: 없음" in text

    judgments = parse_judgments(text.replace("- [ ] `x`", "- [x] `x`"))
    assert judgments == {"f018": (["x"], ["n"])}

    eval_set, judged = tmp_path / "eval_set.jsonl", tmp_path / "pool_judged.json"
    rows = [{"id": "f018", "q": "폐업률은?", "type": "fact", "gold_doc_ids": ["g"], "answer_key": "8.64%"},
            {"id": "u001", "q": "전망은?", "type": "unanswerable", "gold_doc_ids": [], "answer_key": "거절"}]
    eval_set.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    assert apply_judgments(eval_set, judged, judgments) == {"f018": 1}
    updated = [json.loads(line) for line in eval_set.read_text(encoding="utf-8").splitlines()]
    assert updated[0]["gold_doc_ids"] == ["g", "x"] and updated[1]["gold_doc_ids"] == []
    assert json.loads(judged.read_text(encoding="utf-8")) == {"f018": ["n"]}
    assert apply_judgments(eval_set, judged, judgments) == {}            # 다시 반영해도 중복 추가 없음


def test_apply_keeps_gold_documents_out_of_judged(tmp_path):
    eval_set, judged = tmp_path / "eval_set.jsonl", tmp_path / "pool_judged.json"
    eval_set.write_text(json.dumps({"id": "t007", "q": "?", "type": "temporal", "gold_doc_ids": ["g"],
                                    "answer_key": "30만원"}, ensure_ascii=False), encoding="utf-8")
    # 체크하지 않은 판정표를 먼저 반영해 정답 문서 g 가 '정답 아님'에 남은 상태
    judged.write_text(json.dumps({"t007": ["g", "n"], "f001": ["z"]}), encoding="utf-8")

    # 처음엔 정답 아님으로 판정했던 n 을 다시 판정해 정답으로 바꾼다
    assert apply_judgments(eval_set, judged, {"t007": (["n"], ["m"])}) == {"t007": 1}

    assert json.loads(judged.read_text(encoding="utf-8")) == {"f001": ["z"], "t007": ["m"]}


def test_resolve_result_dir_accepts_tag(tmp_path):
    from eval.pool import resolve_result_dir

    for name in ("20261104-1100_primary8-kb2", "20261104-1200_primary8-kb2", "20261104-1100_primary8-news"):
        (tmp_path / name).mkdir()
    assert resolve_result_dir(Path("primary8-kb2"), tmp_path) == tmp_path / "20261104-1200_primary8-kb2"
    assert resolve_result_dir(tmp_path / "20261104-1100_primary8-news", tmp_path).name == "20261104-1100_primary8-news"
    assert resolve_result_dir(Path("nothing"), tmp_path) == Path("nothing")
