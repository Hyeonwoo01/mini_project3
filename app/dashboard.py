"""평가 결과 시각화 (가이드 4-8, Plotly).

eval/results/ 의 실험별 결과 파일을 읽어서 표시한다.
- 실험 설정별 Hit@1·Hit@3·MRR 막대 비교 (기준 설정 대비 개선안)
- 질문 유형별 정확도 — 취약 유형 식별
- 정답·오답 사례의 유사도 점수 분포 — 임계값 설정 근거
- 문서 종류·시기 분포 — 지식베이스 구성과 편향
- 응답시간 분포 — 심화 확장 적용 시
"""

import streamlit as st


def render() -> None:
    st.title("📊 평가 결과")
    st.info("평가 하네스(eval/run_eval.py) 결과가 쌓이면 시각화를 연결합니다.")
