"""임베딩 생성 및 Chroma 적재·검증 (가이드 4-5).

- 임베딩 모델은 common.embeddings.get_embedding_model() 로만 생성한다
- 컬렉션 메타데이터에 임베딩 모델 이름과 청킹 설정을 기록한다 (retriever 가 일치 여부 확인)
- 벡터 DB 에는 청크와 메타데이터(common.schema.CHUNK_METADATA_FIELDS)만 저장한다
- 적재 후 문서 수, 청크 수, 빈 청크 발생 여부를 검증하고 샘플 검색 결과를 확인한다
"""

from indexer.chunk import ChunkConfig


def build_collection(config: ChunkConfig, collection: str) -> None:
    raise NotImplementedError


def verify_collection(collection: str) -> None:
    raise NotImplementedError
