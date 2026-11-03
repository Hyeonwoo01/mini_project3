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
