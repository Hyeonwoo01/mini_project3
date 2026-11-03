"""② 인덱싱 계층 (배치 실행).

docs.db 의 documents 를 읽어 파싱·정제·중복 처리 후 processed_documents 에 기록하고(prepare),
청킹·임베딩하여 Chroma 컬렉션에 적재한다(build). 챗봇은 이 패키지를 import 하지 않는다.
"""
