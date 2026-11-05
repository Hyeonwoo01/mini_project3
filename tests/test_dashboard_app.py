"""대시보드 화면이 보기마다 오류 없이 그려지는지 (저장소의 eval/results 사용, 원문 DB 는 가짜)."""

import pytest
from streamlit.testing.v1 import AppTest

from app import dashboard


def _script() -> None:
    from app import dashboard as page

    page.render()


@pytest.fixture
def fake_db(monkeypatch):
    def composition():
        return ([("news", "kept", 10), ("news", "duplicate", 2), ("press_release", "kept", 3)],
                [("news", "2025-12", 6), ("news", "2026-01", 4), ("press_release", "2026-01", 3)])
    monkeypatch.setattr(dashboard, "composition", composition)


def _html(at: AppTest) -> str:
    return "\n".join(str(e.value) for e in at.get("html"))


EXPECTED_TEXT = {
    "요약": "실험별 결정",
    "검색 실험": "유형별 Hit@3",
    "답변 품질": "문항별 등급",
    "거절 임계값": "구분별 요약",
    "지식베이스": "수집 → 인덱싱",
}


@pytest.mark.parametrize("view", dashboard.VIEWS)
def test_every_view_renders(view, fake_db):
    at = AppTest.from_function(_script, default_timeout=30)
    at.session_state["dashboard_view"] = view
    at.run()
    assert not at.exception, at.exception
    assert EXPECTED_TEXT[view] in _html(at)
    assert "평가 결과 파일을 찾을 수 없습니다" not in _html(at)


def test_overview_hero_shows_adopted_accuracy(fake_db):
    at = AppTest.from_function(_script, default_timeout=30).run()
    assert '<div class="cb-hero__value">65.8%</div>' in _html(at)   # 분해 v2 + 보고서 + 법령, 38문항 (README 실험 기록)
    assert "답변 가능 38문항 · 3회 평균" in _html(at)


@pytest.mark.parametrize(("comparison", "question_set", "expected"), [
    ("reports", "all", "두 설정이 함께 답한 답변 가능 32문항"),      # 평가셋이 다른 설정은 공통 문항으로 비교
    ("reports-q", "all", "두 설정이 함께 답한 답변 가능 2문항"),
    ("reports-q", "report", "두 설정이 함께 답한 답변 가능 2문항"),
    ("laws", "all", "두 설정이 함께 답한 답변 가능 34문항"),
    ("laws-q", "law", "두 설정이 함께 답한 답변 가능 4문항"),
    ("prompt", "report", "이 비교에는 해당 문항이 없습니다"),         # 보고서 문항이 없는 설정끼리
])
def test_report_comparisons_use_common_questions(comparison, question_set, expected, fake_db):
    at = AppTest.from_function(_script, default_timeout=30)
    at.session_state["dashboard_view"] = "답변 품질"
    at.session_state["answer_comparison"] = comparison
    at.session_state["answer_question_set"] = question_set
    at.run()
    assert not at.exception, at.exception
    assert expected in " ".join(str(c.value) for c in at.caption) + _html(at)


def test_answer_view_filters_render(fake_db):
    at = AppTest.from_function(_script, default_timeout=30)
    at.session_state["dashboard_view"] = "답변 품질"
    at.session_state["answer_question_set"] = "primary"
    at.session_state["answer_comparison"] = "prompt"
    at.session_state["answer_grid_filter"] = "all"
    at.run()
    assert not at.exception, at.exception


@pytest.mark.parametrize("comparison", ["primary", "prompt", "prompt-v3", "seed"])
def test_answer_comparisons_render(comparison, fake_db):
    at = AppTest.from_function(_script, default_timeout=30)
    at.session_state["dashboard_view"] = "답변 품질"
    at.session_state["answer_comparison"] = comparison
    at.run()
    assert not at.exception, at.exception
    assert "문항별 등급" in _html(at)


def test_knowledge_base_without_db_shows_guidance(monkeypatch):
    def missing():
        raise FileNotFoundError("store/docs.db")
    monkeypatch.setattr(dashboard, "composition", missing)
    at = AppTest.from_function(_script, default_timeout=30)
    at.session_state["dashboard_view"] = "지식베이스"
    at.run()
    assert not at.exception
    assert "원문 DB(store/docs.db)가 없습니다" in _html(at)


def test_missing_results_show_guidance(monkeypatch, tmp_path, fake_db):
    monkeypatch.setattr(dashboard, "_results_dir", lambda: tmp_path / "eval" / "results")
    at = AppTest.from_function(_script, default_timeout=30)
    at.session_state["dashboard_view"] = "검색 실험"
    at.run()
    assert not at.exception
    assert "평가 결과 파일을 찾을 수 없습니다" in _html(at)


@pytest.mark.parametrize("comparison", ["chunking", "hybrid", "primary", "primary8", "decompose-v1", "decompose-v2"])
def test_retrieval_comparisons_render(comparison, fake_db):
    at = AppTest.from_function(_script, default_timeout=30)
    at.session_state["dashboard_view"] = "검색 실험"
    at.session_state["retrieval_comparison"] = comparison
    at.run()
    assert not at.exception, at.exception
    captions = " ".join(str(c.value) for c in at.caption)
    if comparison == "decompose-v1":
        assert "양쪽 근거 포함 all@5 0.40 → 0.40" in captions
    if comparison == "decompose-v2":
        assert "양쪽 근거 포함 all@5 0.40 → 0.60" in captions
