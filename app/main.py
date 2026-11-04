"""Streamlit 진입점: python -m streamlit run app/main.py (프로젝트 루트에서)

챗봇과 평가 결과 시각화를 별도 페이지(탭)로 구성한다 (가이드 4-8).
"""

import sys
from pathlib import Path

# `streamlit run` 은 이 파일의 디렉터리(app/)만 sys.path 에 추가하므로,
# 공유 패키지(rag, common) import 를 위해 프로젝트 루트를 추가한다.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402

from app import chat, dashboard  # noqa: E402
from app.theme import WORDMARK, inject_theme  # noqa: E402

st.set_page_config(page_title=WORDMARK, page_icon=":material/manage_search:", layout="wide")
inject_theme()

# 페이지 링크는 상단 검정 글로벌 내비게이션에 둔다 (app/theme.py)
page = st.navigation(
    [
        st.Page(chat.render, title="근거 질의응답", url_path="chat", default=True),
        st.Page(dashboard.render, title="평가 결과", url_path="evaluation"),
    ],
    position="top",
)
page.run()
