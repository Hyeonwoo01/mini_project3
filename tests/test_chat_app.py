"""챗봇 화면 동작 (Streamlit AppTest + 가짜 체인). 실제 검색·LLM 호출 없이 화면 흐름만 확인한다."""

from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

from app import chat
from common.schema import DocType
from rag.prompts import REFUSAL_MESSAGE
from rag.retriever import RetrievedChunk


def _chunk(doc_id: str, title: str, body: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{doc_id}#0", doc_id=doc_id, text=f"{title}\n\n{body}", score=0.68,
        metadata={"title": title, "source": "금융위원회", "published_at": "2025-12-30", "doc_type": "press_release",
                  "source_tier": "1차 자료", "url": f"https://example.com/{doc_id}"},
    )


class FakeChain:
    config = SimpleNamespace(min_score=0.5, top_k=5)
    retriever = SimpleNamespace(mode="hybrid", collection="docs__test")

    def __init__(self, answer: str = "총 310.2만개입니다 [1].", chunks=None, fail_on_stream: bool = False):
        self.answer = answer
        self.chunks = [_chunk("kr_1", "우대수수료율 적용 대상", "영세·중소 가맹점은 310.2만개다."),
                       _chunk("kr_2", "다른 보도자료", "관련 없는 내용이다.")] if chunks is None else chunks
        self.fail_on_stream = fail_on_stream
        self.calls: list[dict] = []

    def retrieve_context(self, question, history=(), *, filters=None):
        self.calls.append({"question": question, "history": list(history), "filters": filters})
        return list(self.chunks)

    def stream_answer(self, question, chunks, history=()):
        if self.fail_on_stream:
            raise ConnectionError("API 연결 실패")
        if not chunks:
            yield REFUSAL_MESSAGE
            return
        yield from (self.answer[:5], self.answer[5:])


def _script() -> None:
    from app import chat as chat_page

    chat_page.render()


@pytest.fixture
def use_chain(monkeypatch):
    def install(chain):
        monkeypatch.setattr(chat, "get_chain", lambda: chain)
        return chain
    return install


def _app() -> AppTest:
    return AppTest.from_function(_script, default_timeout=10).run()


def _turns(at: AppTest) -> list[chat.Turn]:
    return at.session_state[chat.TURNS]


def test_empty_state_offers_example_questions(use_chain):
    use_chain(FakeChain())
    at = _app()
    assert not at.exception
    assert len(at.button(key="example-0").label) > 0
    assert _turns(at) == []


def test_example_question_is_answered_with_numbered_sources(use_chain):
    chain = use_chain(FakeChain())
    at = _app()
    at.button(key="example-0").click().run()

    assert not at.exception
    [turn] = _turns(at)
    assert turn.question == chat.EXAMPLE_QUESTIONS[0]
    assert turn.answer == "총 310.2만개입니다 [1]."
    assert not turn.refused and turn.error is None
    assert [s.cited for s in turn.sources] == [True, False]
    assert turn.status_label.startswith("근거 2건 확인 · 1차 자료 2")
    assert any(":blue-badge[1]" in md.value for md in at.markdown)
    assert chain.calls[0]["filters"] is None


def test_follow_up_question_passes_history_and_filters(use_chain):
    chain = use_chain(FakeChain())
    at = _app()
    at.chat_input[0].set_value("우대수수료 가맹점 수는?").run()
    at.session_state[chat.FILTER_TYPES] = [DocType.PRESS_RELEASE]
    at.run()
    at.chat_input[0].set_value("그럼 작년은?").run()

    assert not at.exception
    assert len(_turns(at)) == 2
    second = chain.calls[-1]
    assert second["history"] == [("user", "우대수수료 가맹점 수는?"), ("assistant", "총 310.2만개입니다 [1].")]
    assert second["filters"].doc_types == (DocType.PRESS_RELEASE,)
    assert _turns(at)[1].filters_label == "보도자료"


def test_no_evidence_shows_guidance_instead_of_error(use_chain):
    use_chain(FakeChain(chunks=[]))
    at = _app()
    at.chat_input[0].set_value("망원동 카페 월평균 카드 매출은?").run()

    assert not at.exception
    [turn] = _turns(at)
    assert turn.retrieval_refused and turn.refused
    assert turn.answer == REFUSAL_MESSAGE
    assert turn.status_label == "기준을 넘는 근거 문서 없음"


def test_generation_failure_is_kept_out_of_history(use_chain):
    chain = use_chain(FakeChain(fail_on_stream=True))
    at = _app()
    at.chat_input[0].set_value("첫 질문").run()

    assert not at.exception
    [turn] = _turns(at)
    assert turn.error == "ConnectionError: API 연결 실패"

    chain.fail_on_stream = False
    at.chat_input[0].set_value("다시 질문").run()
    assert chain.calls[-1]["history"] == []


def test_configuration_error_disables_input(monkeypatch):
    def broken():
        raise RuntimeError("환경변수 OPENAI_API_KEY 가 설정되지 않았습니다.")
    monkeypatch.setattr(chat, "get_chain", broken)
    at = _app()

    assert not at.exception
    assert at.chat_input[0].disabled


def test_reset_clears_conversation(use_chain):
    use_chain(FakeChain())
    at = _app()
    at.chat_input[0].set_value("질문").run()
    assert len(_turns(at)) == 1
    at.button(key="reset").click().run()
    assert _turns(at) == []


def test_chat_history_skips_failed_turns():
    turns = [chat.Turn(question="a", answer="A"), chat.Turn(question="b", error="X"), chat.Turn(question="c", answer="C")]
    assert chat.chat_history(turns) == [("user", "a"), ("assistant", "A"), ("user", "c"), ("assistant", "C")]
