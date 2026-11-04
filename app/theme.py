"""디자인 시스템 (Apple 기반 base.md 를 이 서비스에 맞춤): 토큰과 전역 CSS.

색·폰트·모서리처럼 Streamlit 테마 설정으로 되는 것은 .streamlit/config.toml 에 두고,
여기서는 테마로 안 되는 부분만 CSS 로 보완한다.
- 상단 검정 글로벌 내비게이션(44px), 반투명 서브 내비게이션(고정)
- 흰 카드 + 1px 헤어라인 (그림자 없음. 그림자는 원문 문서 시트에만)
- 버튼: 누를 때 scale(0.95), 키보드 포커스 2px #0071e3
- 숫자는 tabular-nums, 한글은 단어 단위 줄바꿈(keep-all)

Streamlit 내부 DOM 대신 가능한 한 key 로 붙는 클래스(.st-key-*)와 data-testid 를 선택자로 쓴다.
Streamlit 버전을 올리면 화면을 한 번 확인한다 (requirements.txt 에 고정).
"""

import streamlit as st

COLORS = {
    "primary": "#0066cc",
    "primary_focus": "#0071e3",
    "ink": "#1d1d1f",
    "ink_muted_80": "#333333",
    "ink_muted_48": "#7a7a7a",
    "divider_soft": "#f0f0f0",
    "hairline": "#e0e0e0",
    "canvas": "#ffffff",
    "parchment": "#f5f5f7",
    "pearl": "#fafafc",
    "chip_gray": "#d2d2d7",
    "approve": "#1d8348",
    "approve_bg": "#e8f8f0",
    "elective": "#b76e00",
    "elective_bg": "#fef5e7",
    "reject": "#c9342b",
    "reject_bg": "#fdecea",
    "evidence_source": "#fff8c5",
    "evidence_target": "#e6f2ff",
}

NAV_HEIGHT = 44
WORDMARK = "상권 정책 근거"

_C = COLORS
GLOBAL_CSS = f"""
:root {{
  --cb-primary: {_C["primary"]};
  --cb-focus: {_C["primary_focus"]};
  --cb-ink: {_C["ink"]};
  --cb-muted-80: {_C["ink_muted_80"]};
  --cb-muted-48: {_C["ink_muted_48"]};
  --cb-hairline: {_C["hairline"]};
  --cb-divider: {_C["divider_soft"]};
  --cb-canvas: {_C["canvas"]};
  --cb-parchment: {_C["parchment"]};
  --cb-pearl: {_C["pearl"]};
}}

.stApp {{ word-break: keep-all; overflow-wrap: break-word; letter-spacing: -0.374px; }}
.stApp p, .stApp li {{ line-height: 1.47; }}

/* global-nav: 검정 44px, 왼쪽 워드마크, 페이지 링크 12px */
header[data-testid="stHeader"] {{
  background: #000; height: {NAV_HEIGHT}px; min-height: {NAV_HEIGHT}px;
}}
header[data-testid="stHeader"]::before {{
  content: "{WORDMARK}"; position: absolute; left: 24px; top: 0; line-height: {NAV_HEIGHT}px;
  color: #fff; font-size: 13px; font-weight: 600; letter-spacing: -0.12px; z-index: 1;
}}
[data-testid="stToolbar"] {{ height: {NAV_HEIGHT}px; min-height: {NAV_HEIGHT}px; padding-left: 140px; }}
[data-testid="stToolbar"] button, [data-testid="stToolbar"] svg {{ color: rgba(255, 255, 255, 0.8); }}
[data-testid="stTopNavLink"] {{ background: transparent !important; padding: 0 10px; }}
[data-testid="stTopNavLink"] p {{
  color: rgba(255, 255, 255, 0.72); font-size: 12px; letter-spacing: -0.12px; font-weight: 400;
}}
[data-testid="stTopNavLink"][aria-current="page"] p, [data-testid="stTopNavLink"]:hover p {{ color: #fff; }}

/* 본문 영역: 내비게이션 높이만큼 띄운다. 폭은 페이지마다 set_page_width 로 정한다 */
[data-testid="stMainBlockContainer"] {{ padding-top: {NAV_HEIGHT + 8}px; padding-bottom: 48px; }}

/* sub-nav-frosted: 페이지 제목 + 빠른 필터·주요 동작. 스크롤해도 내비게이션 아래에 붙어 있다.
   sticky 는 부모 상자 안에서만 움직이므로 st.container 를 감싼 래퍼에 건다 */
[data-testid="stLayoutWrapper"]:has(> .st-key-subnav) {{ position: sticky; top: {NAV_HEIGHT}px; z-index: 50; }}
.st-key-subnav {{
  padding: 6px 0;
  background: rgba(245, 245, 247, 0.8);
  backdrop-filter: saturate(180%) blur(20px); -webkit-backdrop-filter: saturate(180%) blur(20px);
  border-bottom: 1px solid var(--cb-hairline); margin-bottom: 8px;
}}
.cb-subnav__title {{ font-size: 21px; font-weight: 600; line-height: 1.19; letter-spacing: 0.231px; color: var(--cb-ink); }}
.cb-subnav__caption {{
  display: block; margin-top: 2px; font-size: 12px; font-weight: 400; letter-spacing: -0.12px;
  color: var(--cb-muted-48);
}}

/* 버튼: 그림자 없음, 누르면 축소, 포커스 링 */
.stApp button {{ box-shadow: none !important; transition: transform 0.12s ease; }}
.stApp button:active {{ transform: scale(0.95); }}
.stApp button:focus-visible, .stApp a:focus-visible {{ outline: 2px solid var(--cb-focus); outline-offset: 2px; }}
/* button-secondary-pill: 흰 바탕 + Action Blue 글자·테두리 */
[data-testid="stBaseButton-secondary"] {{
  background: var(--cb-canvas); color: var(--cb-primary); border: 1px solid var(--cb-primary);
}}
[data-testid="stBaseButton-secondary"] p {{ font-size: 14px; letter-spacing: -0.224px; }}
[data-testid="stBaseButton-secondary"]:hover {{ background: var(--cb-canvas); color: var(--cb-primary); border-color: var(--cb-primary); }}
/* text-link: 원문 링크 */
[data-testid="stBaseLinkButton-secondary"] {{ border: none; background: transparent; color: var(--cb-primary); padding-left: 6px; padding-right: 6px; }}
[data-testid="stBaseLinkButton-secondary"] p {{ font-size: 14px; letter-spacing: -0.224px; }}
[data-testid="stBaseLinkButton-secondary"]:hover {{ background: transparent; color: var(--cb-primary); text-decoration: underline; }}

[data-testid="stPopoverButton"], [data-testid="stPopoverButton"]:hover, [data-testid="stPopoverButton"][aria-expanded="true"] {{
  background: var(--cb-canvas); color: var(--cb-primary); border: 1px solid var(--cb-primary);
}}
[data-testid="stPopoverButton"] p {{ font-size: 14px; letter-spacing: -0.224px; }}

/* decision-option-chip: 필터 칩(st.pills). 선택되면 배경은 그대로, 테두리만 2px Focus Blue */
button[data-variant="pills"] {{ background: var(--cb-canvas); color: var(--cb-ink); border: 1px solid var(--cb-hairline); }}
button[data-variant="pills"] p {{ font-size: 14px; letter-spacing: -0.224px; }}
/* Streamlit 의 선택 스타일이 선택자 우선순위가 더 높아 !important 로 덮는다 */
button[data-variant="pills"][aria-pressed="true"] {{
  background: var(--cb-canvas) !important; color: var(--cb-ink) !important; border-color: var(--cb-focus) !important;
  outline: 1px solid var(--cb-focus); outline-offset: -2px;
}}
button[data-variant="pills"][aria-pressed="true"] p {{ color: var(--cb-ink) !important; }}

/* decision-option-chip: 예시 질문 */
.st-key-examples button {{
  color: var(--cb-ink); border: 1px solid var(--cb-hairline); background: var(--cb-canvas);
  padding: 12px 16px; min-height: 44px; text-align: left;
}}
.st-key-examples button p {{ font-size: 14px; color: var(--cb-ink); }}
.st-key-examples button:hover {{ border-color: var(--cb-hairline); color: var(--cb-ink); }}
.st-key-examples button:focus-visible {{ outline: none; border: 2px solid var(--cb-focus); }}

/* search-input: 질문 입력창 (캡슐형 44px) */
[data-testid="stChatInput"] > div {{
  border-radius: 9999px; border: 1px solid rgba(0, 0, 0, 0.08); background: var(--cb-canvas); padding-left: 8px;
}}
[data-testid="stChatInput"] > div:focus-within {{ border-color: var(--cb-focus); outline: 1px solid var(--cb-focus); }}
[data-testid="stChatInputTextArea"] {{ font-size: 17px; letter-spacing: -0.374px; }}
[data-testid="stChatInputSubmitButton"]:not(:disabled) {{ background: var(--cb-primary); color: #fff; border-radius: 9999px; }}
[data-testid="stBottom"] > div {{ background: var(--cb-parchment); }}

/* 대화: 질문은 제목처럼, 답변은 evaluation-utility-card */
.cb-question {{
  font-size: 21px; font-weight: 600; line-height: 1.3; letter-spacing: 0.231px; color: var(--cb-ink);
  margin-top: 32px;
}}
[class*="st-key-answer-"] {{
  background: var(--cb-canvas); border: 1px solid var(--cb-hairline); border-radius: 18px; padding: 24px;
}}
[class*="st-key-answer-"] [data-testid="stMarkdownContainer"] p,
[class*="st-key-answer-"] [data-testid="stMarkdownContainer"] li {{ font-size: 17px; line-height: 1.47; }}
.cb-section-label {{
  font-size: 14px; font-weight: 600; letter-spacing: -0.224px; color: var(--cb-muted-80); margin-top: 8px;
}}

/* 출처 카드 */
[class*="st-key-src-"] {{
  background: var(--cb-canvas); border: 1px solid var(--cb-hairline); border-radius: 18px; padding: 17px 24px 12px;
  gap: 4px;
}}
.cb-source__head {{ display: flex; gap: 10px; align-items: baseline; }}
.cb-source__title {{ font-size: 17px; font-weight: 600; line-height: 1.24; letter-spacing: -0.374px; color: var(--cb-ink); }}
.cb-source__meta {{
  display: flex; flex-wrap: wrap; align-items: center; gap: 6px 8px; margin: 6px 0 0;
  font-size: 14px; color: var(--cb-muted-48); letter-spacing: -0.224px; font-variant-numeric: tabular-nums;
}}
.cb-num {{
  flex: none; min-width: 22px; padding: 0 6px; border-radius: 9999px; text-align: center;
  background: {_C["evidence_target"]}; color: var(--cb-primary);
  font-size: 13px; font-weight: 600; line-height: 22px; font-variant-numeric: tabular-nums;
}}
.cb-quote {{
  margin: 12px 0 4px; padding: 17px 24px; border-left: 3px solid var(--cb-primary); border-radius: 8px;
  background: var(--cb-parchment); color: var(--cb-ink); font-size: 15px; line-height: 1.55;
}}

/* confidence-badge-pill */
.cb-badge {{
  display: inline-block; padding: 2px 10px; border-radius: 9999px; font-size: 12px; font-weight: 600;
  line-height: 1.5; letter-spacing: -0.12px; font-variant-numeric: tabular-nums; white-space: nowrap;
}}
.cb-badge--neutral {{ background: var(--cb-pearl); color: var(--cb-muted-80); border: 1px solid var(--cb-divider); }}
.cb-badge--approve {{ background: {_C["approve_bg"]}; color: {_C["approve"]}; }}
.cb-badge--elective {{ background: {_C["elective_bg"]}; color: {_C["elective"]}; }}
.cb-badge--reject {{ background: {_C["reject_bg"]}; color: {_C["reject"]}; }}
.cb-badge--info {{ background: {_C["evidence_target"]}; color: var(--cb-primary); }}

/* 안내 콜아웃 (정보 부족·오류) */
.cb-callout {{ border-radius: 11px; padding: 14px 18px; font-size: 15px; line-height: 1.5; }}
.cb-callout strong {{ font-weight: 600; }}
.cb-callout ul {{ margin: 6px 0 0; padding-left: 18px; }}
.cb-callout li {{ font-size: 14px; }}
.cb-callout--elective {{ background: {_C["elective_bg"]}; color: {_C["elective"]}; }}
.cb-callout--reject {{ background: {_C["reject_bg"]}; color: {_C["reject"]}; }}
.cb-callout--info {{ background: {_C["evidence_target"]}; color: var(--cb-primary); }}

/* 원문 확인: 문서 시트 (시스템에서 유일하게 그림자를 쓰는 곳) */
.cb-doc-title {{ font-size: 21px; font-weight: 600; line-height: 1.3; letter-spacing: 0.231px; color: var(--cb-ink); }}
.cb-doc {{
  max-height: 60vh; overflow-y: auto; white-space: pre-wrap; background: var(--cb-canvas);
  padding: 32px 40px; border-radius: 8px; box-shadow: rgba(0, 0, 0, 0.22) 3px 5px 30px 0;
  font-size: 16px; line-height: 1.55; color: var(--cb-ink); margin: 8px 4px 16px;
}}
.cb-mark {{ background: {_C["evidence_source"]}; color: inherit; border-radius: 5px; padding: 2px 4px; }}

/* 빈 화면 */
.cb-empty {{ padding: 48px 0 16px; }}
.cb-empty__title {{ font-size: 34px; font-weight: 600; line-height: 1.25; letter-spacing: -0.374px; color: var(--cb-ink); }}
.cb-empty p {{ margin-top: 12px; font-size: 17px; color: var(--cb-muted-80); max-width: 640px; }}

/* footer */
.cb-footer {{ margin-top: 48px; padding-top: 16px; border-top: 1px solid var(--cb-hairline); }}
.cb-footer p {{ font-size: 12px; line-height: 1.5; letter-spacing: -0.12px; color: var(--cb-muted-48); margin: 0 0 4px; }}

/* ---- 대시보드 ---- */
/* evaluation-utility-card: 차트·표 카드 */
[class*="st-key-card-"] {{
  background: var(--cb-canvas); border: 1px solid var(--cb-hairline); border-radius: 18px; padding: 24px;
  margin-top: 8px;
}}
.cb-card-title {{ font-size: 17px; font-weight: 600; line-height: 1.24; letter-spacing: -0.374px; color: var(--cb-ink); }}
.cb-card-caption {{ margin-top: 4px; font-size: 14px; line-height: 1.43; letter-spacing: -0.224px; color: var(--cb-muted-48); }}
/* 대표 숫자 (화면당 하나) */
.cb-hero {{ padding: 32px 0 16px; }}
.cb-hero__label {{ font-size: 17px; font-weight: 600; color: var(--cb-muted-80); }}
.cb-hero__value {{ font-size: 56px; font-weight: 600; line-height: 1.07; letter-spacing: -0.28px; color: var(--cb-ink); margin-top: 8px; }}
.cb-hero__caption {{ margin-top: 12px; font-size: 14px; color: var(--cb-muted-48); max-width: 880px; }}
/* 지표 타일 */
.cb-tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin: 8px 0; }}
.cb-tile {{ background: var(--cb-canvas); border: 1px solid var(--cb-hairline); border-radius: 18px; padding: 20px 24px; }}
.cb-tile__label {{ font-size: 14px; font-weight: 600; letter-spacing: -0.224px; color: var(--cb-muted-80); }}
.cb-tile__value {{ margin-top: 8px; font-size: 34px; font-weight: 600; line-height: 1.1; letter-spacing: -0.374px; color: var(--cb-ink); }}
.cb-tile__caption {{ margin-top: 8px; font-size: 12px; line-height: 1.4; letter-spacing: -0.12px; color: var(--cb-muted-48); }}
/* 표 */
.cb-table-wrap {{ overflow-x: auto; margin-top: 12px; }}
.cb-table {{ width: 100%; border-collapse: collapse; font-size: 14px; letter-spacing: -0.224px; color: var(--cb-ink); }}
.cb-table th {{
  text-align: left; font-weight: 600; color: var(--cb-muted-80); padding: 8px 12px 8px 0;
  border-bottom: 1px solid var(--cb-hairline); white-space: nowrap;
}}
.cb-table td {{ padding: 10px 12px 10px 0; border-bottom: 1px solid var(--cb-divider); vertical-align: top; line-height: 1.43; }}
.cb-table .num {{ text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }}
.cb-table th.num {{ text-align: right; }}
.cb-table--grades td.cb-grade {{ text-align: center; }}
.cb-table--grades th:nth-child(n+4):not(.num) {{ text-align: center; }}
.cb-table--grades td.num {{ text-align: left; }}
.cb-table .cb-q {{ min-width: 240px; }}
/* segmented control: 보기·비교 전환 (선택 = 흰 바탕 + 2px Focus Blue 테두리) */
button[data-variant="segmented_control"] {{ background: var(--cb-canvas); color: var(--cb-ink); border-color: var(--cb-hairline); }}
button[data-variant="segmented_control"] p {{ font-size: 14px; letter-spacing: -0.224px; }}
button[data-variant="segmented_control"][aria-checked="true"] {{
  background: var(--cb-canvas) !important; color: var(--cb-ink) !important; border-color: var(--cb-focus) !important;
  outline: 1px solid var(--cb-focus); outline-offset: -2px; z-index: 1;
}}
button[data-variant="segmented_control"][aria-checked="true"] p {{ color: var(--cb-ink) !important; font-weight: 600; }}

/* 휴대폰 폭: 서브 내비게이션이 화면을 많이 가리지 않게 고정을 풀고 설명을 숨긴다 */
@media (max-width: 640px) {{
  [data-testid="stLayoutWrapper"]:has(> .st-key-subnav) {{ position: static; }}
  .cb-subnav__caption {{ display: none; }}
  .cb-question {{ font-size: 19px; }}
  .cb-empty__title {{ font-size: 28px; }}
  [class*="st-key-answer-"] {{ padding: 17px; }}
  [class*="st-key-src-"] {{ padding: 17px 17px 12px; }}
  .cb-quote {{ padding: 12px 17px; }}
  .cb-doc {{ padding: 20px; }}
  [class*="st-key-card-"] {{ padding: 17px; }}
  .cb-hero__value {{ font-size: 40px; }}
  .cb-tile__value {{ font-size: 28px; }}
}}

/* st.status · st.expander: 헤어라인 카드 */
[data-testid="stExpander"] details {{ border: 1px solid var(--cb-hairline); border-radius: 11px; background: var(--cb-canvas); }}
[data-testid="stExpander"] summary p {{ font-size: 14px; letter-spacing: -0.224px; }}
"""


def inject_theme() -> None:
    st.html(f"<style>{GLOBAL_CSS}</style>")


def set_page_width(max_px: int) -> None:
    """본문·입력창 최대 폭. 챗봇은 집중형 980px, 대시보드는 1440px (base.md Grid & Container)."""
    st.html(
        "<style>"
        f'[data-testid="stMainBlockContainer"], [data-testid="stBottomBlockContainer"] {{ max-width: {max_px}px; }}'
        "</style>"
    )
