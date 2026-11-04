from langchain_core.language_models.fake_chat_models import FakeListChatModel

from rag.chain import ChainConfig, RagChain
from rag.decompose import DecomposingRetriever, QueryDecomposer, interleave, parse_queries, sub_queries
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


def test_interleave_alternates_and_skips_duplicates():
    a = [_chunk("a1"), _chunk("shared"), _chunk("a3")]
    b = [_chunk("b1"), _chunk("shared"), _chunk("b2"), _chunk("b3")]
    assert [c.doc_id for c in interleave([a, b], 5)] == ["a1", "b1", "shared", "a3", "b2"]
    # 같은 날 같은 제목의 다른 기사도 한 번만
    same = [_chunk("x", title="같은 제목"), _chunk("y", title="같은 제목")]
    assert [c.doc_id for c in interleave([same[:1], same[1:]], 5)] == ["x"]


def test_single_query_passes_through_unchanged():
    inner = FakeRetriever({"사실 질문": [_chunk("d1"), _chunk("d2")]})
    retriever = DecomposingRetriever(inner, CountingDecomposer({}))
    result = retriever.retrieve("사실 질문", k=5)
    assert [c.doc_id for c in result] == ["d1", "d2"]
    assert all("sub_query" not in c.metadata for c in result)
    assert retriever.mode == "hybrid" and retriever.collection == "docs__test"


def test_comparison_query_retrieves_each_target_and_tags_chunks():
    inner = FakeRetriever({"A 지원": [_chunk("a1"), _chunk("a2"), _chunk("a3")],
                           "B 지원": [_chunk("b1"), _chunk("b2")]})
    retriever = DecomposingRetriever(inner, CountingDecomposer({"A와 B 비교": ["A 지원", "B 지원"]}))
    result = retriever.retrieve("A와 B 비교", k=4)
    assert inner.queries == ["A 지원", "B 지원"]
    assert [c.doc_id for c in result] == ["a1", "b1", "a2", "b2"]
    assert [c.metadata["sub_query"] for c in result] == ["A 지원", "B 지원", "A 지원", "B 지원"]
    assert sub_queries(result) == ["A 지원", "B 지원"]

    retriever.expand_context("A와 B 비교", result, per_doc=2)
    assert sorted(inner.expand_calls) == [("A 지원", ["a1", "a2"]), ("B 지원", ["b1", "b2"])]


def test_chain_with_decompose_uses_split_queries():
    inner = FakeRetriever({"A 지원": [_chunk("a1", 0.8)], "B 지원": [_chunk("b1", 0.7)],
                           "A와 B 비교": [_chunk("a1", 0.8), _chunk("a2", 0.75)]})
    llm = FakeListChatModel(responses=['{"queries": ["A 지원", "B 지원"]}', "A 는 … [1], B 는 … [2]"])
    chain = RagChain(inner, llm, ChainConfig(top_k=5, min_score=0.5, decompose=True))
    result = chain.answer("A와 B 비교")
    assert [c.doc_id for c in result.chunks] == ["a1", "b1"]
    assert result.cited == [1, 2]


def test_chain_without_decompose_keeps_plain_retriever():
    inner = FakeRetriever({})
    chain = RagChain(inner, FakeListChatModel(responses=["x"]), ChainConfig(top_k=5, min_score=0.5))
    assert chain.retriever is inner
