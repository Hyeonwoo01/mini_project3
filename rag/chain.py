"""LangChain 기반 RAG 체인 — app 과 eval 의 유일한 호출 지점.

    app : chunks = retrieve_context(...)      # 검색 진행 상태 표시
          st.write_stream(stream_answer(...))  # 스트리밍 출력
    eval: result = answer(...)                 # 같은 두 함수를 순서대로 호출

멀티턴: history 는 생성 단계에 전달한다. 검색 쿼리에 대화 맥락을 반영하는 것은
심화 확장(질의 재작성)으로 다루며, 적용 시 retrieve_context 안에서만 처리한다.
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from rag.retriever import RetrievedChunk, SearchFilters

# (role, content) — role 은 "user" 또는 "assistant"
ChatHistory = Sequence[tuple[str, str]]


@dataclass(frozen=True)
class RagResult:
    answer: str
    refused: bool
    chunks: list[RetrievedChunk]  # 답변의 [n] 번호는 chunks[n-1] 에 대응한다


def retrieve_context(
    question: str,
    history: ChatHistory = (),
    *,
    filters: SearchFilters | None = None,
    collection: str | None = None,
) -> list[RetrievedChunk]:
    """검색 후 유사도 임계값을 적용한다. 빈 리스트는 근거 없음(거절)을 뜻한다."""
    raise NotImplementedError


def stream_answer(
    question: str,
    chunks: list[RetrievedChunk],
    history: ChatHistory = (),
) -> Iterator[str]:
    """chunks 가 비어 있으면 LLM 을 호출하지 않고 REFUSAL_MESSAGE 를 내보낸다."""
    raise NotImplementedError


def answer(
    question: str,
    history: ChatHistory = (),
    *,
    filters: SearchFilters | None = None,
    collection: str | None = None,
) -> RagResult:
    raise NotImplementedError
