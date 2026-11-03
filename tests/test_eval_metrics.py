import json

import pytest

from eval.run_eval import hit_at_k, load_eval_set, reciprocal_rank


def test_hit_at_k():
    ranked = ["a", "b", "c", "d"]
    assert hit_at_k(ranked, {"c"}, k=3)
    assert not hit_at_k(ranked, {"d"}, k=3)
    assert hit_at_k(ranked, {"x", "a"}, k=1)


def test_reciprocal_rank():
    assert reciprocal_rank(["a", "b", "c"], {"b"}) == pytest.approx(0.5)
    assert reciprocal_rank(["a", "b", "c"], {"c", "b"}) == pytest.approx(0.5)
    assert reciprocal_rank(["a", "b"], {"z"}) == 0.0


def test_metrics_reject_unanswerable_and_bad_k():
    with pytest.raises(ValueError):
        hit_at_k(["a"], set(), k=3)
    with pytest.raises(ValueError):
        reciprocal_rank(["a"], [])
    with pytest.raises(ValueError):
        hit_at_k(["a"], {"a"}, k=0)


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")


def test_load_eval_set(tmp_path):
    path = tmp_path / "eval_set.jsonl"
    _write_jsonl(path, [
        {"id": "f001", "q": "질문", "type": "fact", "gold_doc_ids": ["d1"], "answer_key": "답"},
        {"id": "u001", "q": "질문", "type": "unanswerable", "gold_doc_ids": [], "answer_key": "거절"},
    ])
    items = load_eval_set(path)
    assert [i.id for i in items] == ["f001", "u001"]
    assert items[0].gold_doc_ids == ("d1",)


@pytest.mark.parametrize(
    "row",
    [
        {"id": "f001", "q": "질문", "type": "fact", "gold_doc_ids": [], "answer_key": "답"},
        {"id": "u001", "q": "질문", "type": "unanswerable", "gold_doc_ids": ["d1"], "answer_key": "거절"},
        {"id": "x001", "q": "질문", "type": "opinion", "gold_doc_ids": ["d1"], "answer_key": "답"},
        {"id": "f001", "q": "질문", "type": "fact", "gold_doc_ids": "d1", "answer_key": "답"},
        {"q": "질문", "type": "fact", "gold_doc_ids": ["d1"], "answer_key": "답"},
    ],
)
def test_load_eval_set_rejects_invalid_rows(tmp_path, row):
    path = tmp_path / "eval_set.jsonl"
    _write_jsonl(path, [row])
    with pytest.raises(ValueError):
        load_eval_set(path)


def test_load_eval_set_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "eval_set.jsonl"
    row = {"id": "f001", "q": "질문", "type": "fact", "gold_doc_ids": ["d1"], "answer_key": "답"}
    _write_jsonl(path, [row, row])
    with pytest.raises(ValueError):
        load_eval_set(path)


def test_check_eval_set_reports_missing_unanswerable(tmp_path, monkeypatch, capsys):
    from eval import run_eval
    path = tmp_path / "eval_set.jsonl"
    _write_jsonl(path, [{"id": "f001", "q": "질문", "type": "fact", "gold_doc_ids": ["d1"], "answer_key": "답"}])
    monkeypatch.setattr(run_eval, "open_db", lambda **_: (_ for _ in ()).throw(FileNotFoundError()))
    problems = run_eval.check_eval_set(path)
    assert any("unanswerable" in p for p in problems)
    assert "fact" in capsys.readouterr().out


def test_check_eval_set_flags_empty_file(tmp_path, monkeypatch):
    from eval import run_eval
    path = tmp_path / "eval_set.jsonl"
    path.write_text("", encoding="utf-8")
    monkeypatch.setattr(run_eval, "open_db", lambda **_: (_ for _ in ()).throw(FileNotFoundError()))
    assert any("비어" in p for p in run_eval.check_eval_set(path))


class _FakeRetriever:
    """질문별로 정해 둔 (doc_id, 점수) 목록을 돌려준다."""

    def __init__(self, results):
        self.results = results

    def retrieve(self, query, *, k):
        from rag.retriever import RetrievedChunk
        return [RetrievedChunk(chunk_id=f"{d}#000", doc_id=d, text="", score=s, metadata={"title": d})
                for d, s in self.results[query][:k]]


def _items():
    from eval.run_eval import EvalItem
    return [
        EvalItem("f001", "q1", "fact", ("d1",), "a"),            # 1위 적중
        EvalItem("f002", "q2", "fact", ("d9", "d3"), "a"),       # 3위 적중
        EvalItem("c001", "q3", "comparison", ("dx",), "a"),      # 미적중
        EvalItem("u001", "q4", "unanswerable", (), "거절"),
    ]


def _fake():
    return _FakeRetriever({
        "q1": [("d1", 0.8), ("d2", 0.5), ("d3", 0.4)],
        "q2": [("d1", 0.6), ("d2", 0.5), ("d3", 0.45), ("d4", 0.4)],
        "q3": [("d5", 0.3), ("d6", 0.2), ("d7", 0.1)],
        "q4": [("d8", 0.25), ("d2", 0.2), ("d1", 0.1)],
    })


def test_evaluate_and_summarize_retrieval():
    from eval.run_eval import evaluate_retrieval, format_summary, summarize
    records = evaluate_retrieval(_items(), _fake(), k=5)
    by_id = {r["id"]: r for r in records}
    assert by_id["f001"]["gold_rank"] == 1 and by_id["f002"]["gold_rank"] == 3 and by_id["c001"]["gold_rank"] is None
    assert "hit@1" not in by_id["u001"] and by_id["u001"]["top_score"] == 0.25

    summary = summarize(records, k=5)
    assert summary["overall"] == {"count": 3, "hit@1": round(1 / 3, 4), "hit@3": round(2 / 3, 4),
                                  "hit@5": round(2 / 3, 4), "mrr": round((1 + 1 / 3) / 3, 4)}
    assert summary["by_type"]["comparison"]["hit@3"] == 0
    assert summary["top_score"]["answerable_hit"]["median"] == 0.8
    assert summary["top_score"]["unanswerable"]["median"] == 0.25
    assert summary["unanswerable_count"] == 1
    assert "Hit@3 0.667" in format_summary(summary, k=5)


def test_save_results_writes_files(tmp_path):
    from eval.run_eval import evaluate_retrieval, save_results, summarize
    records = evaluate_retrieval(_items(), _fake(), k=5)
    out = save_results(records, summarize(records, 5), {"tag": "t"}, "c1000", tmp_path)
    assert (out / "retrieval.jsonl").read_text(encoding="utf-8").count("\n") == 4
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["config"] == {"tag": "t"} and summary["overall"]["count"] == 3
    with pytest.raises(ValueError):
        save_results(records, summarize(records, 5), {}, "bad tag!", tmp_path)


def test_latency_p95_uses_nearest_rank():
    from eval.run_eval import summarize
    records = [{"gold_doc_ids": [], "top_score": 0.1, "latency_ms": ms, "type": "unanswerable"} for ms in (4.0, 6.0)]
    assert summarize(records, k=5)["latency_ms"] == {"mean": 5.0, "p95": 6.0}


class _FakeChain:
    def __init__(self, results):
        self.results = results

    def answer(self, question):
        return self.results[question]


def _answer_fixture():
    from rag.chain import RagResult
    from rag.prompts import REFUSAL_MESSAGE
    from rag.retriever import RetrievedChunk
    chunk = lambda d: RetrievedChunk(f"{d}#000", d, "t", 0.7, {"title": d, "published_at": "2025-12-16"})  # noqa: E731
    return _FakeChain({
        "q1": RagResult("3,916억원 [1]", False, [chunk("d1"), chunk("d2")], cited=[1], latency_ms=1000),
        "q2": RagResult("다른 문서 인용 [2]", False, [chunk("d1"), chunk("d2")], cited=[2], latency_ms=1000),
        "q3": RagResult(REFUSAL_MESSAGE, True, [], retrieval_refused=True, latency_ms=10),     # 잘못된 거절
        "q4": RagResult(REFUSAL_MESSAGE, True, [], retrieval_refused=True, latency_ms=10),     # 올바른 거절
    })


def test_evaluate_and_summarize_answers(tmp_path):
    import csv
    from eval.run_eval import evaluate_answers, summarize_answers, write_answers_csv
    rows = evaluate_answers(_items(), _answer_fixture())
    by_id = {r["id"]: r for r in rows}
    assert by_id["f001"]["cited_gold_auto"] is True and by_id["f002"]["cited_gold_auto"] is False
    assert by_id["u001"]["cited_gold_auto"] is None and by_id["u001"]["refused"] is True

    summary = summarize_answers(rows)
    assert summary["refusal_accuracy"] == 1.0
    assert summary["false_refusal_rate"] == round(1 / 3, 4)
    assert summary["citation_rate"] == 1.0 and summary["cited_gold_rate_auto"] == 0.5
    assert summary["counts"] == {"answerable": 3, "unanswerable": 1, "answered": 2}

    path = tmp_path / "answers.csv"
    write_answers_csv(rows, path)
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")                      # 엑셀용 BOM
    with path.open(encoding="utf-8-sig", newline="") as f:
        written = list(csv.DictReader(f))
    assert written[0]["grade"] == "" and "[1] d1 (2025-12-16, d1, 유사도 0.700)" in written[0]["sources"]


def test_load_eval_set_accepts_bom_and_crlf(tmp_path):
    # Windows 메모장은 UTF-8 저장 시 BOM 을 붙인다
    row = {"id": "f001", "q": "질문", "type": "fact", "gold_doc_ids": ["news_1"], "answer_key": "답"}
    path = tmp_path / "eval_set.jsonl"
    path.write_bytes((json.dumps(row, ensure_ascii=False) + "\r\n").encode("utf-8-sig"))
    assert [item.id for item in load_eval_set(path)] == ["f001"]


def test_top_score_is_highest_similarity_regardless_of_rank():
    from eval.run_eval import EvalItem, evaluate_retrieval
    from rag.retriever import RetrievedChunk

    class _Unsorted:  # 하이브리드 검색은 1위가 최고 유사도가 아닐 수 있다
        def retrieve(self, query, *, k):
            return [RetrievedChunk("a#000", "a", "", 0.55, {"title": "a", "keyword_rank": 1, "vector_rank": None}),
                    RetrievedChunk("b#000", "b", "", 0.71, {"title": "b", "keyword_rank": None, "vector_rank": 1})]

    item = EvalItem(id="f001", q="질문", type="fact", gold_doc_ids=("b",), answer_key="답")
    record = evaluate_retrieval([item], _Unsorted(), k=3)[0]
    assert record["top_score"] == 0.71 and record["gold_rank"] == 2
    assert record["retrieved"][0]["keyword_rank"] == 1 and record["retrieved"][0]["vector_rank"] is None
