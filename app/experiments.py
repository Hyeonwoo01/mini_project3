"""대시보드에 싣는 실험 목록 (eval/results 하위 폴더 이름).

결과 폴더는 정답 보정 전후(평가셋 v1 ~ v2.0)와 중간 실험이 섞여 있어, 같은 기준으로 비교할 수 있는 짝만
고른다. README '실험 기록'의 표와 같은 실험이다. 새 실험을 대시보드에 올리려면 여기에 추가한다
(tests/test_eval_data.py 가 폴더가 실제로 있는지 확인한다).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RunRef:
    run_id: str   # eval/results 하위 폴더 이름
    label: str


@dataclass(frozen=True)
class RetrievalComparison:
    """같은 평가셋 버전에서 한 가지만 바꾼 검색 실험 한 쌍."""

    key: str
    title: str
    eval_set: str
    baseline: RunRef
    candidate: RunRef
    decision: str


@dataclass(frozen=True)
class AnswerSetting:
    """반복 실행한 답변 설정. 같은 순서의 실행끼리 짝지어 문항 전체(뉴스 문항 + 1차 자료 문항) 정답률을 낸다."""

    key: str
    label: str
    news_question_runs: tuple[str, ...]      # 평가셋 v1.3 30문항 (답변 가능 24 + 답변 불가 6)
    primary_question_runs: tuple[str, ...]   # 평가셋 v2.0 에 추가한 1차 자료 문항 8개


RETRIEVAL_COMPARISONS: tuple[RetrievalComparison, ...] = (
    RetrievalComparison(
        key="chunking", title="청킹 크기", eval_set="v1",
        baseline=RunRef("20261103-1615_baseline-c1000", "c1000 의미 검색"),
        candidate=RunRef("20261103-1615_baseline-c500", "c500 의미 검색"),
        decision="모든 지표에서 앞선 c500 채택",
    ),
    RetrievalComparison(
        key="hybrid", title="검색 방식", eval_set="v1.2",
        baseline=RunRef("20261103-1702_baseline-c500-v1.2", "의미 검색"),
        candidate=RunRef("20261103-1702_hybrid-c500-v1.2", "하이브리드 (의미 + 키워드)"),
        decision="연도·기관명을 놓치던 문항이 살아나 하이브리드 채택",
    ),
    RetrievalComparison(
        key="primary", title="1차 자료 추가 (뉴스 문항)", eval_set="v1.3",
        baseline=RunRef("20261104-1019_hybrid-c500-v1.3", "뉴스만"),
        candidate=RunRef("20261104-1020_hybrid-c500-kb2-v1.3", "+ 1차 자료"),
        decision="Hit@1 소폭 상승, 나머지는 같음",
    ),
    RetrievalComparison(
        key="primary8", title="1차 자료 추가 (1차 자료 문항)", eval_set="v2.0",
        baseline=RunRef("20261104-1132_primary8-news-v2.0", "뉴스만"),
        candidate=RunRef("20261104-1132_primary8-kb2-v2.0", "+ 1차 자료"),
        decision="최신 제도의 세부 기준은 1차 자료로만 검색됨 → 1차 자료 유지",
    ),
)

# 답변 근거 청크 수 비교 (뉴스 지식베이스, 각 1회)
CONTEXT_CHUNK_COMPARISON = (
    RunRef("20261103-1649_hybrid-c500-answers", "문서당 청크 1개"),
    RunRef("20261103-1711_hybrid-c500-ctx2-answers", "문서당 청크 2개"),
)

ANSWER_SETTINGS: tuple[AnswerSetting, ...] = (
    AnswerSetting(
        key="kb2", label="+ 1차 자료 (채택)",
        news_question_runs=(
            "20261104-1040_hybrid-c500-ctx2-kb2-answers",
            "20261104-1050_hybrid-c500-ctx2-kb2-answers-r2",
            "20261104-1055_hybrid-c500-ctx2-kb2-answers-r3",
        ),
        primary_question_runs=(
            "20261104-1127_primary8-kb2-ans-r1",
            "20261104-1128_primary8-kb2-ans-r2",
            "20261104-1128_primary8-kb2-ans-r3",
        ),
    ),
    AnswerSetting(
        key="news", label="뉴스만",
        news_question_runs=(
            "20261104-1106_hybrid-c500-ctx2-news-r1",
            "20261104-1108_hybrid-c500-ctx2-news-r2",
            "20261104-1110_hybrid-c500-ctx2-news-r3",
        ),
        primary_question_runs=(
            "20261104-1129_primary8-news-ans-r1",
            "20261104-1129_primary8-news-ans-r2",
            "20261104-1130_primary8-news-ans-r3",
        ),
    ),
)
ADOPTED_ANSWER_SETTING = "kb2"

# 거절 임계값 근거: 답변 불가 문항까지 포함한 검색 실험 (30문항)
THRESHOLD_RUNS: tuple[RunRef, ...] = (
    RunRef("20261104-1035_hybrid-c500-kb2-pdffix", "하이브리드 + 1차 자료 (현재)"),
    RunRef("20261104-1019_hybrid-c500-v1.3", "하이브리드, 뉴스만"),
    RunRef("20261103-1702_baseline-c500-v1.2", "의미 검색 기준선"),
)
