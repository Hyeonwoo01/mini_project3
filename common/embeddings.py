"""임베딩 모델 생성.

indexer(문서 임베딩)와 rag(질문 임베딩)가 반드시 같은 모델을 쓰도록 생성 지점을 하나로 둔다.
모델이 다르면 오류 없이 검색 품질만 무너지므로, 모델 이름은 컬렉션 메타데이터에도 기록하고
retriever 에서 일치 여부를 확인한다 (indexer/embed.py, rag/retriever.py 참조).
"""

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from common.config import get_settings, require_env


def get_embedding_model() -> Embeddings:
    require_env("OPENAI_API_KEY")
    # 공용 키라 호출 한도(429)에 걸리기 쉬워 재시도를 넉넉히 둔다.
    # 요청 제한 시간 기본값(600초)이면 응답 없는 요청 하나에 10분씩 멈춰 있게 되므로 60초로 줄인다.
    return OpenAIEmbeddings(model=get_settings().embedding_model, max_retries=6, timeout=60)
