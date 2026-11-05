"""LangChain 기반 RAG 체인 — app 과 eval 의 유일한 호출 지점.

    app : chain = get_chain()
          chunks = chain.retrieve_context(question, history, filters=...)   # 검색 진행 상태 표시
          st.write_stream(chain.stream_answer(question, chunks, history))   # 스트리밍 출력
    eval: result = get_chain(collection).answer(question)                   # 같은 두 단계를 순서대로 호출

거절은 두 단계다 (가이드 4-6, 4-9).
1. 검색 단계: 유사도가 RAG_MIN_SCORE 미만인 결과는 근거에서 뺀다. 남는 근거가 없으면 LLM 을 부르지 않고
   REFUSAL_MESSAGE 로 답한다.
2. 생성 단계: 근거가 있어도 질문에 답할 내용이 없으면 프롬프트 규칙에 따라 LLM 이 REFUSAL_MESSAGE 만 답한다.

멀티턴: 대화 이력은 최근 MAX_HISTORY_TURNS 턴만 생성 단계에 넣는다. 검색 질의에는 직전 사용자 질문을
덧붙여 "그럼 작년은?" 같은 후속 질문도 주제를 잃지 않게 한다 (기본 대응. LLM 으로 질의를 다시 쓰는 것은
심화 확장 '질의 재작성'으로 다룬다).

답변의 [n] 은 chunks[n-1] 에 대응한다. 답변에 실제로 인용된 번호는 RagResult.cited 로 확인한다.
"""

import re
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from common.config import get_settings, require_env
from common.dates import annotate_relative_dates
from rag.decompose import DecomposingRetriever, QueryDecomposer
from rag.prompts import DEFAULT_PROMPT_VERSION, REFUSAL_MESSAGE, SYSTEM_PROMPTS, USER_PROMPT_TEMPLATE
from rag.retriever import RetrievedChunk, Retriever, SearchFilters

# (role, content) — role 은 "user" 또는 "assistant"
ChatHistory = Sequence[tuple[str, str]]

MAX_HISTORY_TURNS = 3
# 비교 질문 분해 (rag/decompose.py). v2 채택: 비교 문항 양쪽 근거 포함 all@5 0.40 → 0.60, 정답률 57.3 → 61.5%
# (README 실험 기록). 비교 표현이 없는 질문은 분해하지 않아 LLM 호출·응답 시간이 그대로다
DEFAULT_DECOMPOSE = True
# 답변 근거 청크의 상대 날짜('내일부터')에 발행일 기준 날짜를 덧붙인다 (common/dates.py). 검색은 바꾸지 않는다.
# 실험 중 (README 실험 기록 '상대 날짜를 답변 근거에만')
DEFAULT_ANNOTATE_DATES = False
_CITATION = re.compile(r"\[(\d{1,2})\]")


@dataclass(frozen=True)
class ChainConfig:
    top_k: int = 5
    min_score: float = 0.5
    context_chunks: int = 1   # 임계값을 통과한 문서마다 근거로 넣는 청크 수 (서비스 기본값은 설정 RAG_CONTEXT_CHUNKS)
    prompt_version: str = DEFAULT_PROMPT_VERSION   # rag.prompts.SYSTEM_PROMPTS 의 키
    decompose: bool = DEFAULT_DECOMPOSE   # 비교 질문을 대상별 질의로 나눠 검색 (LLM 호출 1회 추가)
    annotate_dates: bool = DEFAULT_ANNOTATE_DATES   # 근거 청크의 상대 날짜에 날짜를 덧붙여 프롬프트에 넣는다

    def __post_init__(self) -> None:
        if self.top_k < 1 or not 0.0 <= self.min_score < 1.0 or not 1 <= self.context_chunks <= 3:
            raise ValueError(f"잘못된 체인 설정: {self}")
        if self.prompt_version not in SYSTEM_PROMPTS:
            raise ValueError(f"알 수 없는 프롬프트 버전: {self.prompt_version!r} (가능: {', '.join(SYSTEM_PROMPTS)})")


@dataclass(frozen=True)
class RagResult:
    answer: str
    refused: bool
    chunks: list[RetrievedChunk]          # 답변의 [n] 번호는 chunks[n-1] 에 대응한다
    cited: list[int] = field(default_factory=list)   # 답변에 실제로 나온 근거 번호
    retrieval_refused: bool = False       # 근거가 임계값을 넘지 못해 LLM 호출 없이 거절했는지
    latency_ms: float = 0.0


def is_refusal(answer: str) -> bool:
    """근거 없음 거절인지. 부분 답변('…은 제공된 자료에서 확인되지 않는다')은 거절이 아니다."""
    return answer.strip().startswith(REFUSAL_MESSAGE.rstrip("."))


def cited_numbers(answer: str, n_chunks: int) -> list[int]:
    """답변에 나온 [n] 중 실제 근거 범위 안의 번호를 처음 나온 순서대로."""
    seen: list[int] = []
    for match in _CITATION.finditer(answer):
        number = int(match.group(1))
        if 1 <= number <= n_chunks and number not in seen:
            seen.append(number)
    return seen


def format_context(chunks: Sequence[RetrievedChunk], annotate_dates: bool = False) -> str:
    """번호·제목·발행처·출처 종류·발행일을 붙인 근거 목록. 번호가 답변의 [n] 이 된다.
    annotate_dates 이면 본문의 상대 날짜에 발행일 기준 날짜를 덧붙인다 ('내일부터' → '내일(2025년 2월 21일)부터')."""
    blocks = []
    for number, chunk in enumerate(chunks, start=1):
        meta = chunk.metadata
        published = str(meta.get("published_at", ""))[:10] or "발행일 미상"
        header = f"[{number}] {meta.get('title', '')} | {meta.get('source', '')} | {meta.get('source_tier', '')} | {published}"
        text = annotate_relative_dates(chunk.text, meta.get("published_at"))[0] if annotate_dates else chunk.text
        blocks.append(f"{header}\n{text}")
    return "\n\n".join(blocks)


def _history_messages(history: ChatHistory) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for role, content in list(history)[-MAX_HISTORY_TURNS * 2:]:
        if role == "user":
            messages.append(HumanMessage(content))
        elif role == "assistant":
            messages.append(AIMessage(content))
        else:
            raise ValueError(f"알 수 없는 role: {role!r}")
    return messages


def retrieval_query(question: str, history: ChatHistory) -> str:
    """직전 사용자 질문을 덧붙인 검색 질의. 이력이 없으면 질문 그대로."""
    previous = next((content for role, content in reversed(list(history)) if role == "user"), None)
    return f"{previous} {question}" if previous else question


class RagChain:
    def __init__(self, retriever: Retriever, llm: BaseChatModel, config: ChainConfig) -> None:
        # 분해 검색기는 체인 안에서 감싼다: 평가 하네스의 검색 평가도 chain.retriever 로 같은 경로를 탄다
        self.retriever = DecomposingRetriever(retriever, QueryDecomposer(llm)) if config.decompose else retriever
        self.config = config
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPTS[config.prompt_version]),
            MessagesPlaceholder("history"),
            ("human", USER_PROMPT_TEMPLATE),
        ])
        self._generate = prompt | llm | StrOutputParser()

    def retrieve_context(
        self, question: str, history: ChatHistory = (), *, filters: SearchFilters | None = None
    ) -> list[RetrievedChunk]:
        """검색 후 유사도 임계값 미만을 뺀다. 빈 리스트는 근거 없음(거절)을 뜻한다.
        context_chunks > 1 이면 남은 문서마다 질문과 유사한 청크를 더 모아 근거를 넓힌다."""
        query = retrieval_query(question, history)
        chunks = self.retriever.retrieve(query, k=self.config.top_k, filters=filters)
        kept = [c for c in chunks if c.score >= self.config.min_score]
        if kept and self.config.context_chunks > 1:
            kept = self.retriever.expand_context(query, kept, per_doc=self.config.context_chunks, filters=filters)
        return kept

    def stream_answer(self, question: str, chunks: list[RetrievedChunk], history: ChatHistory = ()) -> Iterator[str]:
        """chunks 가 비어 있으면 LLM 을 호출하지 않고 REFUSAL_MESSAGE 를 내보낸다."""
        if not chunks:
            yield REFUSAL_MESSAGE
            return
        yield from self._generate.stream({
            "history": _history_messages(history),
            "context": format_context(chunks, self.config.annotate_dates),
            "question": question,
            "today": date.today().isoformat(),
        })

    def answer(
        self, question: str, history: ChatHistory = (), *, filters: SearchFilters | None = None
    ) -> RagResult:
        started = time.perf_counter()
        chunks = self.retrieve_context(question, history, filters=filters)
        text = "".join(self.stream_answer(question, chunks, history)).strip()
        return RagResult(
            answer=text,
            refused=not chunks or is_refusal(text),
            chunks=chunks,
            cited=cited_numbers(text, len(chunks)),
            retrieval_refused=not chunks,
            latency_ms=(time.perf_counter() - started) * 1000,
        )


def get_chat_model(seed: int | None = None) -> BaseChatModel:
    require_env("OPENAI_API_KEY")
    # 근거 문서에 충실하도록 temperature 0. 공용 키라 응답 지연·호출 한도에 대비해 제한 시간과 재시도를 둔다.
    # seed: OpenAI 의 재현성 옵션 (같은 입력이면 같은 출력을 '최대한' 낸다, 보장은 아님). 반복 실행 변동 확인용.
    return ChatOpenAI(model=get_settings().chat_model, temperature=0, timeout=60, max_retries=3, seed=seed)


@lru_cache(maxsize=4)
def get_chain(collection: str | None = None, mode: str | None = None, context_chunks: int | None = None,
              prompt_version: str | None = None, seed: int | None = None, decompose: bool | None = None,
              annotate_dates: bool | None = None) -> RagChain:
    """설정(CHROMA_COLLECTION, RAG_RETRIEVAL_MODE, RAG_TOP_K, RAG_MIN_SCORE, RAG_CONTEXT_CHUNKS)을 쓰는 체인.
    챗봇·평가 하네스가 공유한다. 인자를 주면 그 값이 설정보다 우선한다 (평가 실험용).
    프롬프트 버전은 설정이 아닌 코드(rag.prompts.DEFAULT_PROMPT_VERSION)로 관리한다."""
    settings = get_settings()
    config = ChainConfig(top_k=settings.rag_top_k, min_score=settings.rag_min_score,
                         context_chunks=context_chunks or settings.rag_context_chunks,
                         prompt_version=prompt_version or DEFAULT_PROMPT_VERSION,
                         decompose=DEFAULT_DECOMPOSE if decompose is None else decompose,
                         annotate_dates=DEFAULT_ANNOTATE_DATES if annotate_dates is None else annotate_dates)
    return RagChain(
        retriever=Retriever(collection or settings.chroma_collection, mode=mode or settings.rag_retrieval_mode),
        llm=get_chat_model(seed),
        config=config,
    )


def retrieve_context(
    question: str, history: ChatHistory = (), *, filters: SearchFilters | None = None, collection: str | None = None
) -> list[RetrievedChunk]:
    return get_chain(collection).retrieve_context(question, history, filters=filters)


def stream_answer(
    question: str, chunks: list[RetrievedChunk], history: ChatHistory = (), *, collection: str | None = None
) -> Iterator[str]:
    return get_chain(collection).stream_answer(question, chunks, history)


def answer(
    question: str, history: ChatHistory = (), *, filters: SearchFilters | None = None, collection: str | None = None
) -> RagResult:
    return get_chain(collection).answer(question, history, filters=filters)
