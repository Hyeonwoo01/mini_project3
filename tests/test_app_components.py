from app import components as ui
from app.sources import SourceView


def _source(**overrides) -> SourceView:
    fields = dict(number=1, doc_id="kr_1", title="우대수수료율 <확정>", publisher="금융위원회", published="2025-12-30",
                  doc_type="press_release", tier="1차 자료", url="https://example.com/1",
                  passage="영세·중소 가맹점 <b>310.2만개</b>", score=0.7123, vector_rank=1, keyword_rank=None,
                  cited=True)
    fields.update(overrides)
    return SourceView(**fields)


def test_source_card_escapes_collected_text():
    html = ui.source_card_html(_source(title='<script>alert("x")</script>'))
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "&lt;b&gt;310.2만개&lt;/b&gt;" in html


def test_source_card_shows_tier_badge_and_scores_only_in_debug():
    plain = ui.source_card_html(_source())
    assert "cb-badge--approve" in plain and "1차 자료" in plain
    assert "금융위원회 · 2025-12-30 · 보도자료" in plain
    assert "유사도" not in plain

    debug = ui.source_card_html(_source(), show_scores=True)
    assert "유사도 0.712" in debug and "의미 1위" in debug and "키워드" not in debug


def test_source_card_handles_missing_date_and_long_passage():
    html = ui.source_card_html(_source(published="", passage="가" * 1000, tier=""))
    assert "발행일 미상" in html
    assert "…" in html and "가" * 1000 not in html
    assert "cb-badge" not in html


def test_highlighted_document_marks_passages_and_escapes():
    text = "앞 문장 <주의>.\n\n연 매출 30억원 이하 가맹점 중 영세·중소 가맹점은 310.2만개다.\n\n뒤 문장."
    html, count = ui.highlighted_document_html(text, "연 매출 30억원 이하 가맹점 중 영세·중소 가맹점은 310.2만개다.")
    assert count == 1
    assert '<mark class="cb-mark">연 매출 30억원' in html
    assert "&lt;주의&gt;" in html


def test_highlighted_document_without_match_returns_plain_text():
    html, count = ui.highlighted_document_html("원문 본문", "원문에 없는 근거 문단이 여기에 있습니다. 길게 씁니다.")
    assert count == 0
    assert "<mark" not in html and "원문 본문" in html


def test_callout_escapes_lines():
    html = ui.callout_html("제목", ["<x>", "둘째"], tone="reject")
    assert "cb-callout--reject" in html and "&lt;x&gt;" in html and html.count("<li>") == 2
