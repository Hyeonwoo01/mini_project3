import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from rag.chain import (
    ChainConfig,
    RagChain,
    _history_messages,
    cited_numbers,
    format_context,
    is_refusal,
    retrieval_query,
)
from rag.prompts import PARTIAL_PHRASE, REFUSAL_MESSAGE
from rag.retriever import RetrievedChunk


def _chunk(doc_id, score, title="상생페이백 11월분 지급"):
    return RetrievedChunk(chunk_id=f"{doc_id}#000", doc_id=doc_id, text=f"{title}\n\n본문 {doc_id}", score=score,
                          metadata={"title": title, "source": "연합뉴스", "source_tier": "언론 보도",
                                    "published_at": "2025-12-16T09:00:00", "doc_id": doc_id})


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.queries = []

    def retrieve(self, query, *, k, filters=None):
        self.queries.append(query)
        return self.chunks[:k]


def _chain(chunks, responses, min_score=0.4):
    retriever = FakeRetriever(chunks)
    # responses 가 비어 있으면 LLM 이 호출될 때 오류가 난다 → 'LLM 을 부르지 않았다'를 검증할 수 있다
    return RagChain(retriever, FakeListChatModel(responses=responses), ChainConfig(top_k=5, min_score=min_score)), retriever


def test_refuses_without_calling_llm_when_no_chunk_passes_threshold():
    chain, _ = _chain([_chunk("a", 0.35), _chunk("b", 0.2)], responses=[])
    result = chain.answer("2030년 폐업률 전망은?")
    assert result.answer == REFUSAL_MESSAGE
    assert result.refused and result.retrieval_refused
    assert result.chunks == [] and result.cited == []


def test_answer_with_citations_only_uses_chunks_above_threshold():
    chain, _ = _chain([_chunk("a", 0.7), _chunk("b", 0.5), _chunk("c", 0.3)],
                      responses=["11월분은 3,916억원이다 [1][3][2][1]."])
    result = chain.answer("상생페이백 11월분 지급액은?")
    assert [c.doc_id for c in result.chunks] == ["a", "b"]           # 0.3 은 근거에서 제외
    assert result.cited == [1, 2]                                     # 범위 밖 [3]·중복 제거
    assert not result.refused and not result.retrieval_refused


def test_llm_refusal_and_partial_answer_are_distinguished():
    refused, _ = _chain([_chunk("a", 0.7)], responses=[REFUSAL_MESSAGE])
    assert refused.answer("질문").refused
    partial, _ = _chain([_chunk("a", 0.7)], responses=[f"지급액은 3,916억원이다 [1]. 지급 대상 연령은 {PARTIAL_PHRASE}."])
    result = partial.answer("질문")
    assert not result.refused and result.cited == [1]


def test_stream_answer_yields_tokens_and_refusal_for_empty_context():
    chain, _ = _chain([_chunk("a", 0.7)], responses=["답변 [1]"])
    assert "".join(chain.stream_answer("질문", [_chunk("a", 0.7)])) == "답변 [1]"
    assert list(chain.stream_answer("질문", [])) == [REFUSAL_MESSAGE]


def test_follow_up_question_keeps_previous_topic_in_retrieval():
    chain, retriever = _chain([_chunk("a", 0.7)], responses=["답변 [1]"])
    history = [("user", "상생페이백 11월분 지급액은?"), ("assistant", "3,916억원입니다 [1].")]
    chain.answer("그럼 누적 지급액은?", history)
    assert retriever.queries == ["상생페이백 11월분 지급액은? 그럼 누적 지급액은?"]
    assert retrieval_query("첫 질문", []) == "첫 질문"


def test_history_is_truncated_and_validated():
    history = [("user", f"q{i}") if i % 2 == 0 else ("assistant", f"a{i}") for i in range(10)]
    messages = _history_messages(history)
    assert len(messages) == 6 and messages[-1].content == "a9"
    with pytest.raises(ValueError):
        _history_messages([("system", "x")])


def test_prompt_variables_and_context_format():
    chain, _ = _chain([], responses=[])
    assert set(chain._generate.first.input_variables) == {"history", "context", "question", "today"}
    context = format_context([_chunk("a", 0.7), _chunk("b", 0.6, title="다른 기사")])
    assert context.startswith("[1] 상생페이백 11월분 지급 | 연합뉴스 | 언론 보도 | 2025-12-16")
    assert "\n\n[2] 다른 기사" in context


def test_helpers_and_config_validation():
    assert is_refusal(f"  {REFUSAL_MESSAGE}") and not is_refusal(f"일부는 {PARTIAL_PHRASE}")
    assert cited_numbers("근거 [2], [12] 그리고 [2][1]", n_chunks=3) == [2, 1]
    with pytest.raises(ValueError):
        ChainConfig(top_k=0)
    with pytest.raises(ValueError):
        ChainConfig(min_score=1.0)


class ExpandingRetriever(FakeRetriever):
    def __init__(self, chunks):
        super().__init__(chunks)
        self.expanded = []

    def expand_context(self, query, chunks, *, per_doc, filters=None):
        self.expanded.append(([c.doc_id for c in chunks], per_doc))
        return [RetrievedChunk(c.chunk_id, c.doc_id, c.text + "\n추가 청크", c.score, c.metadata) for c in chunks]


def test_context_chunks_expand_only_documents_above_threshold():
    retriever = ExpandingRetriever([_chunk("a", 0.7), _chunk("b", 0.3)])
    chain = RagChain(retriever, FakeListChatModel(responses=["답 [1]"]),
                     ChainConfig(top_k=5, min_score=0.5, context_chunks=2))
    result = chain.answer("질문")
    assert retriever.expanded == [(["a"], 2)]                    # 임계값 미만 b 는 확장하지 않음
    assert result.chunks[0].text.endswith("추가 청크")
    with pytest.raises(ValueError):
        ChainConfig(context_chunks=4)


def test_context_chunks_default_does_not_expand():
    retriever = ExpandingRetriever([_chunk("a", 0.7)])
    RagChain(retriever, FakeListChatModel(responses=["답 [1]"]), ChainConfig(top_k=5, min_score=0.5)).answer("질문")
    assert retriever.expanded == []


def test_v1_prompt_text_is_frozen():
    """기존 실험과 비교할 기준 프롬프트(v1)는 바뀌면 안 된다. 바꾸려면 새 버전을 추가한다."""
    import hashlib

    from rag.prompts import SYSTEM_PROMPTS
    digest = hashlib.sha256(SYSTEM_PROMPTS["v1"].encode()).hexdigest()
    assert digest == "bd658f928d62d633bd4c9a1e80279f570c66d03e996b5d66d7067822b38706bc"


def test_prompt_version_selects_system_prompt():
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from rag.prompts import SYSTEM_PROMPTS

    class RecordingModel(FakeListChatModel):
        seen: list = []

        def _stream(self, messages, stop=None, run_manager=None, **kwargs):  # answer() 는 스트리밍으로 생성한다
            type(self).seen.append(messages[0].content)
            yield from super()._stream(messages, stop=stop, run_manager=run_manager, **kwargs)

    for version in ("v1", "v2"):
        RecordingModel.seen = []
        chain = RagChain(FakeRetriever([_chunk("a", 0.7)]), RecordingModel(responses=["답 [1]"]),
                         ChainConfig(top_k=5, min_score=0.5, prompt_version=version))
        chain.answer("질문")
        assert RecordingModel.seen == [SYSTEM_PROMPTS[version]]
    assert SYSTEM_PROMPTS["v2"].startswith(SYSTEM_PROMPTS["v1"]) and "9. 날짜는 종류를 구분한다" in SYSTEM_PROMPTS["v2"]


def test_unknown_prompt_version_is_rejected():
    with pytest.raises(ValueError, match="프롬프트 버전"):
        ChainConfig(prompt_version="v9")
