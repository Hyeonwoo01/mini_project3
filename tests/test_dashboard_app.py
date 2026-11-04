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
    assert '<div class="cb-hero__value">57.3%</div>' in _html(at)


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


@pytest.mark.parametrize("comparison", ["chunking", "hybrid", "primary", "primary8", "decompose-v1"])
def test_retrieval_comparisons_render(comparison, fake_db):
    at = AppTest.from_function(_script, default_timeout=30)
    at.session_state["dashboard_view"] = "검색 실험"
    at.session_state["retrieval_comparison"] = comparison
    at.run()
    assert not at.exception, at.exception
    captions = " ".join(str(c.value) for c in at.caption)
    if comparison == "decompose-v1":
        assert "양쪽 근거 포함 all@5 0.40 → 0.40" in captions
