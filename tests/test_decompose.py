from langchain_core.language_models.fake_chat_models import FakeListChatModel

from rag.chain import ChainConfig, RagChain
from rag.decompose import (
    DecomposingRetriever,
    QueryDecomposer,
    fill_missing_targets,
    looks_like_comparison,
    parse_queries,
    sub_queries,
)
from rag.retriever import RetrievedChunk


def _chunk(doc_id: str, score: float = 0.7, title: str | None = None, date: str = "2026-01-01", index: int = 0):
    return RetrievedChunk(chunk_id=f"{doc_id}#{index}", doc_id=doc_id, text=f"{doc_id} 본문", score=score,
                          metadata={"title": title or f"제목 {doc_id}", "published_at": date, "doc_id": doc_id,
                                    "chunk_index": index})


class FakeRetriever:
    mode = "hybrid"
    collection = "docs__test"

    def __init__(self, results: dict[str, list[RetrievedChunk]]):
        self.results = results
        self.queries: list[str] = []
        self.expand_calls: list[tuple[str, list[str]]] = []

    def retrieve(self, query, *, k, filters=None):
        self.queries.append(query)
        return list(self.results.get(query, []))[:k]

    def expand_context(self, query, chunks, *, per_doc, filters=None):
        self.expand_calls.append((query, [c.doc_id for c in chunks]))
        return list(chunks)


class CountingDecomposer:
    def __init__(self, plan: dict[str, list[str]]):
        self.plan, self.calls = plan, 0

    def decompose(self, question):
        self.calls += 1
        return self.plan.get(question, [question])


def test_parse_queries_accepts_json_with_surrounding_text():
    assert parse_queries('```json\n{"queries": ["A 지급 대상", "B 지급 대상"]}\n```', "Q") == ["A 지급 대상", "B 지급 대상"]


def test_parse_queries_falls_back_to_question():
    assert parse_queries("비교 질문이 아닙니다", "Q") == ["Q"]
    assert parse_queries('{"queries": []}', "Q") == ["Q"]
    assert parse_queries('{"queries": "A"}', "Q") == ["Q"]
    assert parse_queries('{"queries": [', "Q") == ["Q"]


def test_parse_queries_dedupes_and_caps():
    assert parse_queries('{"queries": ["A", " A ", "B", "C", "D", 3]}', "Q") == ["A", "B", "C"]


def test_decomposer_caches_and_survives_llm_errors():
    llm = FakeListChatModel(responses=['{"queries": ["서울시 금융지원", "중기부 금융지원"]}'])
    decomposer = QueryDecomposer(llm)
    assert decomposer.decompose("서울시와 중기부 비교") == ["서울시 금융지원", "중기부 금융지원"]
    assert decomposer.decompose("서울시와 중기부 비교") == ["서울시 금융지원", "중기부 금융지원"]   # 캐시 (응답 1개뿐)

    class Broken(FakeListChatModel):
        def _call(self, *args, **kwargs):
            raise ConnectionError("down")

    assert QueryDecomposer(Broken(responses=["x"])).decompose("질문") == ["질문"]


def test_comparison_cues():
    assert looks_like_comparison("A와 B 는 지원 방식이 어떻게 다른가?")
    assert looks_like_comparison("정부와 반대 측은 각각 어떤 논리를 폈나?")
    assert looks_like_comparison("완화에 대해 어떤 상반된 근거가 있나?")
    assert not looks_like_comparison("상생페이백 11월분 지급 금액과 지급 인원은?")
    assert not looks_like_comparison("2026년 상반기 우대수수료율이 적용되는 가맹점은 몇 개인가?")


def test_fill_missing_targets_keeps_base_and_adds_missing_target():
    base = [_chunk("a1"), _chunk("a2"), _chunk("x1"), _chunk("x2"), _chunk("x3")]
    per_query = [[_chunk("a1"), _chunk("a9")], [_chunk("b1"), _chunk("b2")]]
    merged = fill_missing_targets(base, per_query, 5)
    assert [c.doc_id for c in merged] == ["a1", "a2", "x1", "x2", "b1"]   # 가장 낮은 순위(x3)를 B 의 1위로


def test_fill_missing_targets_gives_every_target_a_slot_from_the_bottom():
    base = [_chunk("x1"), _chunk("x2"), _chunk("x3"), _chunk("x4"), _chunk("x5")]
    per_query = [[_chunk("a1")], [_chunk("b1")], [_chunk("c1")]]
    # 대상 3개가 모두 빠졌으면 아래에서부터 3자리를 내준다 (k=5 면 원래 결과 상위 2건은 남는다)
    assert [c.doc_id for c in fill_missing_targets(base, per_query, 5)] == ["x1", "x2", "a1", "b1", "c1"]
    # 보강 문서끼리는 서로 밀어내지 않는다
    assert [c.doc_id for c in fill_missing_targets(base[:2], per_query, 3)] == ["a1", "b1", "c1"]


def test_fill_missing_targets_treats_same_title_as_present():
    base = [_chunk("x", title="같은 제목")]
    assert [c.doc_id for c in fill_missing_targets(base, [[_chunk("y", title="같은 제목")]], 5)] == ["x"]


def test_non_comparison_question_is_not_decomposed():
    inner = FakeRetriever({"사실 질문": [_chunk("d1"), _chunk("d2")]})
    decomposer = CountingDecomposer({})
    retriever = DecomposingRetriever(inner, decomposer)
    result = retriever.retrieve("사실 질문", k=5)
    assert [c.doc_id for c in result] == ["d1", "d2"] and decomposer.calls == 0
    assert all("sub_query" not in c.metadata for c in result)
    assert retriever.mode == "hybrid" and retriever.collection == "docs__test"


def test_comparison_query_fills_missing_target_and_tags_it():
    inner = FakeRetriever({"A와 B 비교": [_chunk("a1"), _chunk("a2"), _chunk("a3"), _chunk("x1"), _chunk("x2")],
                           "A 지원": [_chunk("a1"), _chunk("a2")],
                           "B 지원": [_chunk("b1"), _chunk("b2")]})
    retriever = DecomposingRetriever(inner, CountingDecomposer({"A와 B 비교": ["A 지원", "B 지원"]}))
    result = retriever.retrieve("A와 B 비교", k=5)
    assert inner.queries == ["A와 B 비교", "A 지원", "B 지원"]
    assert [c.doc_id for c in result] == ["a1", "a2", "a3", "x1", "b1"]
    assert result[-1].metadata["sub_query"] == "B 지원" and "sub_query" not in result[0].metadata
    assert sub_queries(result) == ["B 지원"]

    retriever.expand_context("A와 B 비교", result, per_doc=2)
    assert sorted(inner.expand_calls) == [("A와 B 비교", ["a1", "a2", "a3", "x1"]), ("B 지원", ["b1"])]


def test_chain_with_decompose_adds_the_missing_side():
    inner = FakeRetriever({"A 지원": [_chunk("a1", 0.8)], "B 지원": [_chunk("b1", 0.7)],
                           "A와 B 비교": [_chunk("a1", 0.8), _chunk("a2", 0.75)]})
    llm = FakeListChatModel(responses=['{"queries": ["A 지원", "B 지원"]}', "A 는 … [1], B 는 … [3]"])
    chain = RagChain(inner, llm, ChainConfig(top_k=5, min_score=0.5, decompose=True))
    result = chain.answer("A와 B 비교")
    assert [c.doc_id for c in result.chunks] == ["a1", "a2", "b1"]
    assert result.cited == [1, 3]


def test_chain_without_decompose_keeps_plain_retriever():
    inner = FakeRetriever({})
    chain = RagChain(inner, FakeListChatModel(responses=["x"]), ChainConfig(top_k=5, min_score=0.5, decompose=False))
    assert chain.retriever is inner


def test_decompose_is_default_and_skips_llm_for_non_comparison_questions():
    """채택 설정은 분해를 켠다. 비교 표현이 없는 질문은 LLM 을 부르지 않고 원래 검색 결과를 그대로 쓴다."""
    inner = FakeRetriever({"C 사업 지원 금액은?": [_chunk("c1")]})
    llm = FakeListChatModel(responses=[])   # 호출되면 응답이 없어 실패한다
    chain = RagChain(inner, llm, ChainConfig(top_k=5, min_score=0.5))
    assert isinstance(chain.retriever, DecomposingRetriever)
    assert [c.doc_id for c in chain.retriever.retrieve("C 사업 지원 금액은?", k=5)] == ["c1"]


def test_decomposing_retriever_records_sub_queries():
    inner = FakeRetriever({"A와 B 비교": [_chunk("a1")], "A 지원": [_chunk("a1")], "B 지원": [_chunk("b1")]})
    retriever = DecomposingRetriever(inner, CountingDecomposer({"A와 B 비교": ["A 지원", "B 지원"]}))
    retriever.retrieve("A와 B 비교", k=5)
    assert retriever.last_sub_queries == ["A 지원", "B 지원"]
    retriever.retrieve("사실 질문", k=5)
    assert retriever.last_sub_queries == []
