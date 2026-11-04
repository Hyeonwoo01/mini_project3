"""비교 질문 분해 (검색 단계).

비교 질문은 의미 검색·키워드 검색 모두 '질문 전체와 비슷한 문서'를 찾으므로, 한쪽 대상의 문서가 상위를 채우기 쉽다.
평가셋 v2.1 의 비교 5문항(대상별 정답 문서 `gold_groups`) 기준으로, 채택 설정에서 양쪽 근거를 상위 5건에 모두 찾은
문항은 2개다 (c002·c005). c001·c004 는 한쪽만, c003 은 둘 다 못 찾았다 (README 실험 기록).

LLM 으로 질문을 대상별 검색 질의로 나눠 따로 검색하고, 결과를 순위대로 번갈아 합쳐 대상마다 근거 자리를 준다.
- 비교가 아닌 질문은 LLM 이 원래 질문 하나만 돌려주고, 그때는 기존 검색과 똑같이 동작한다
- 분해에 실패하면(LLM 오류·형식 오류) 원래 질문으로 검색한다. 분해가 검색을 막지 않는다
- 근거 수 k 는 그대로다. 대상마다 자리를 나눌 뿐 컨텍스트를 늘리지 않는다 (한 번에 한 변수만 바꾼다)
- 같은 질문의 분해 결과는 캐시한다. 평가 하네스의 검색 평가와 답변 평가가 같은 하위 질의를 쓰게 하기 위해서다
"""

import json
import logging
import re
from collections import OrderedDict, defaultdict
from collections.abc import Sequence
from dataclasses import replace

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from rag.retriever import RetrievedChunk, SearchFilters, title_key

logger = logging.getLogger(__name__)

MAX_QUERIES = 3
CACHE_SIZE = 256
SUB_QUERY_FIELD = "sub_query"   # 청크 메타데이터: 그 문서를 찾은 하위 질의 (분해했을 때만)

# 규칙은 질문 형식에 관한 것만 둔다 (평가 문항의 대상 이름을 넣지 않는다)
DECOMPOSE_SYSTEM_PROMPT = """당신은 문서 검색 질의를 만드는 도우미다. 질문을 읽고 검색에 쓸 질의를 JSON 으로만 답한다.

규칙:
1. 질문이 둘 이상의 대상(정책·사업·기관·지역·시기)을 비교하거나 서로 다른 입장을 대조하면, 대상마다 따로 검색할 질의를 하나씩 만든다.
2. 각 질의에는 그 대상의 이름과, 질문에 있는 시점·조건(연도, '하반기' 등)과 비교 항목을 그대로 넣는다. 다른 대상의 이름은 넣지 않는다.
3. 비교·대조 질문이 아니면 질문을 그대로 하나만 돌려준다.
4. 질의는 최대 3개다.

출력 형식: {{"queries": ["질의1", "질의2"]}}"""

_JSON_OBJECT = re.compile(r"\{.*\}", re.S)


def parse_queries(text: str, question: str) -> list[str]:
    """LLM 출력에서 질의 목록을 꺼낸다. 형식이 틀리면 [question]."""
    match = _JSON_OBJECT.search(text)
    try:
        payload = json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        payload = None
    raw = payload.get("queries") if isinstance(payload, dict) else None
    if not isinstance(raw, list):
        logger.warning("질문 분해 결과 형식 오류, 원래 질문으로 검색: %r", text[:200])
        return [question]
    queries = list(dict.fromkeys(q.strip() for q in raw if isinstance(q, str) and q.strip()))
    return queries[:MAX_QUERIES] or [question]


class QueryDecomposer:
    def __init__(self, llm: BaseChatModel, cache_size: int = CACHE_SIZE) -> None:
        prompt = ChatPromptTemplate.from_messages([("system", DECOMPOSE_SYSTEM_PROMPT), ("human", "{question}")])
        self._chain = prompt | llm | StrOutputParser()
        self._cache: OrderedDict[str, list[str]] = OrderedDict()
        self._cache_size = cache_size

    def decompose(self, question: str) -> list[str]:
        if question in self._cache:
            self._cache.move_to_end(question)
            return list(self._cache[question])
        try:
            queries = parse_queries(self._chain.invoke({"question": question}), question)
        except Exception:  # 분해는 보조 단계: LLM 호출이 실패해도 원래 질문으로 검색을 이어 간다
            logger.exception("질문 분해 실패, 원래 질문으로 검색: %s", question)
            return [question]
        self._cache[question] = queries
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return list(queries)


def interleave(result_lists: Sequence[Sequence[RetrievedChunk]], k: int) -> list[RetrievedChunk]:
    """하위 질의별 검색 결과를 순위대로 번갈아 합친다 (1위끼리, 2위끼리 …). 같은 문서·같은 날 같은 제목은 한 번만."""
    merged: list[RetrievedChunk] = []
    seen_docs: set[str] = set()
    seen_titles: set[tuple[str, str]] = set()
    for rank in range(max((len(results) for results in result_lists), default=0)):
        for results in result_lists:
            if rank >= len(results):
                continue
            chunk = results[rank]
            key = title_key(chunk.metadata)
            if chunk.doc_id in seen_docs or (key is not None and key in seen_titles):
                continue
            seen_docs.add(chunk.doc_id)
            if key is not None:
                seen_titles.add(key)
            merged.append(chunk)
            if len(merged) == k:
                return merged
    return merged


class DecomposingRetriever:
    """검색기를 감싸 비교 질문을 대상별로 나눠 검색한다. rag.retriever.Retriever 와 같은 방식으로 쓴다."""

    def __init__(self, retriever, decomposer: QueryDecomposer) -> None:
        self.inner = retriever
        self.decomposer = decomposer

    # 화면·평가 기록이 읽는 검색기 설정은 감싼 검색기 것을 그대로 보인다
    def __getattr__(self, name: str):
        if name in ("inner", "decomposer"):   # 생성 전(복사·역직렬화 중)에 불리면 무한 재귀가 된다
            raise AttributeError(name)
        return getattr(self.inner, name)

    def retrieve(self, query: str, *, k: int, filters: SearchFilters | None = None) -> list[RetrievedChunk]:
        queries = self.decomposer.decompose(query)
        if len(queries) < 2:
            return self.inner.retrieve(query, k=k, filters=filters)
        per_query = [
            [replace(c, metadata={**c.metadata, SUB_QUERY_FIELD: q}) for c in self.inner.retrieve(q, k=k, filters=filters)]
            for q in queries
        ]
        return interleave(per_query, k)

    def expand_context(self, query: str, chunks: list[RetrievedChunk], *, per_doc: int,
                       filters: SearchFilters | None = None) -> list[RetrievedChunk]:
        """근거 확장도 그 문서를 찾은 하위 질의로 한다 (질문 전체로 고르면 다른 대상 쪽 청크가 끌려온다)."""
        if not any(SUB_QUERY_FIELD in c.metadata for c in chunks):
            return self.inner.expand_context(query, chunks, per_doc=per_doc, filters=filters)
        groups: dict[str, list[RetrievedChunk]] = defaultdict(list)
        for chunk in chunks:
            groups[chunk.metadata.get(SUB_QUERY_FIELD, query)].append(chunk)
        expanded: dict[str, RetrievedChunk] = {}
        for sub_query, group in groups.items():
            for chunk in self.inner.expand_context(sub_query, group, per_doc=per_doc, filters=filters):
                expanded[chunk.chunk_id] = chunk
        return [expanded[c.chunk_id] for c in chunks]


def sub_queries(chunks: Sequence[RetrievedChunk]) -> list[str]:
    """근거 청크에 기록된 하위 질의 (분해하지 않았으면 빈 목록)."""
    return list(dict.fromkeys(c.metadata[SUB_QUERY_FIELD] for c in chunks if SUB_QUERY_FIELD in c.metadata))
