"""대화 UI.

답변 화면 (가이드 4-6)
- 답변 본문과 번호로 연결된 출처 목록 (제목, 발행처, 발행일, 링크, 출처 종류)
- 원문 일부를 확인할 수 있는 영역 (common.docstore.get_document, 읽기 전용)
- 검색된 청크의 유사도 점수 (디버깅용)

UI 요건 (가이드 4-7)
- 스트리밍 출력, 검색 진행 상태 표시
- st.session_state 로 멀티턴 대화 유지
- 기간·출처 종류 필터 (사이드바 → rag.retriever.SearchFilters)
- 검색 결과가 없으면 오류 대신 안내 메시지
"""

import streamlit as st


def render() -> None:
    st.title("💬 챗봇")
    st.info("RAG 체인(rag/chain.py) 구현 후 대화 UI 를 연결합니다.")
