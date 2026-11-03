import csv

import pytest

from eval.grade import carry_grades, read_answers, summarize_grades, validate, write_answers

FIELDS = ["id", "type", "q", "answer", "sources", "grade", "citation_ok", "note"]


def _row(qid, qtype, grade, citation, answer="답", sources="[1] 기사", note=""):
    return {"id": qid, "type": qtype, "q": "질문", "answer": answer, "sources": sources,
            "grade": grade, "citation_ok": citation, "note": note}


def test_summarize_grades_counts_by_type_and_citation():
    rows = [_row("f001", "fact", "정확", "Y"), _row("f002", "fact", "부분", "N", note="누락"),
            _row("c001", "comparison", "오답", "-", note="잘못된 거절"), _row("u001", "unanswerable", "정확", "-")]
    s = summarize_grades(rows)
    assert s["answerable"]["grades"] == {"정확": 1, "부분": 1, "오답": 1}
    assert s["answerable"]["accuracy"] == pytest.approx(1 / 3, abs=1e-4)
    assert s["answerable"]["accuracy_with_partial"] == pytest.approx(2 / 3, abs=1e-4)
    assert s["answerable"]["by_type"]["fact"]["count"] == 2
    assert s["unanswerable"] == {"count": 1, "correct": 1}
    assert s["citation"] == {"judged": 2, "ok": 1, "accuracy": 0.5}       # '-' 는 판정 대상이 아님
    assert [i["id"] for i in s["issues"]] == ["f002", "c001"]


def test_validate_reports_blank_or_unknown_values():
    assert validate([_row("f001", "fact", "정확", "Y")]) == []
    problems = validate([_row("f001", "fact", "", "Y"), _row("f002", "fact", "맞음", "y")])
    assert len(problems) == 3


def test_carry_grades_only_for_identical_answers():
    previous = [_row("f001", "fact", "정확", "Y", answer="3916억원[1]"),
                _row("f002", "fact", "부분", "Y", answer="옛 답변", note="누락"),
                _row("f003", "fact", "정확", "N", answer="같은 답[1]", sources="[1] 옛 출처")]
    current = [_row("f001", "fact", "", "", answer="3916억원[1]"),
               _row("f002", "fact", "", "", answer="새 답변"),
               _row("f003", "fact", "", "", answer="같은 답[1]", sources="[1] 새 출처")]
    carried, with_citation, todo = carry_grades(previous, current)
    assert (carried, with_citation) == (2, 1)
    assert current[0]["grade"] == "정확" and current[0]["citation_ok"] == "Y"
    assert current[2]["grade"] == "정확" and current[2]["citation_ok"] == ""   # 출처가 바뀌면 다시 판정
    assert todo == ["f002", "f003"]


def test_carry_ignores_chunk_count_note_and_refusals():
    previous = [_row("f001", "fact", "정확", "Y", answer="답[1]", sources="[1] 기사 (2026-01-01, d1, 유사도 0.600)"),
                _row("u001", "unanswerable", "정확", "-", answer="제공된 자료에서 확인할 수 없습니다.", sources="[1] 옛 기사")]
    current = [_row("f001", "fact", "", "", answer="답[1]", sources="[1] 기사 (2026-01-01, d1, 유사도 0.600, 청크 2개)"),
               _row("u001", "unanswerable", "", "", answer="제공된 자료에서 확인할 수 없습니다.", sources="[1] 새 기사")]
    assert carry_grades(previous, current) == (2, 2, [])
    assert current[0]["citation_ok"] == "Y" and current[1]["citation_ok"] == "-"


def test_read_write_roundtrip_with_bom_and_multiline(tmp_path):
    path = tmp_path / "answers.csv"
    rows = [_row("f001", "fact", "정확", "Y", answer="첫 줄\n둘째 줄")]
    write_answers(path, rows)
    assert path.read_bytes().startswith("﻿".encode("utf-8"))           # 엑셀용 BOM
    assert read_answers(path) == rows
    path.write_bytes("id,type\nf001,fact\n".encode("cp949") + "한글".encode("cp949"))
    with pytest.raises(SystemExit, match="CSV UTF-8"):
        read_answers(path)
