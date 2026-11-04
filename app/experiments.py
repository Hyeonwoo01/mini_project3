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
    """반복 실행한 답변 설정. repetitions 의 원소 하나가 실행 1회분이고, 그 안의 결과 폴더들이 문항 전체를 함께 덮는다
    (뉴스 문항·1차 자료 문항을 따로 돌린 실험은 폴더 2개, 평가셋 v2.0 전체를 한 번에 돌린 실험은 폴더 1개)."""

    key: str
    label: str
    short: str                                  # 표 열 이름용 짧은 이름
    repetitions: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class AnswerComparison:
    """한 가지만 바꾼 답변 설정 한 쌍 (설정 key)."""

    key: str
    title: str
    baseline: str
    candidate: str
    decision: str


# 평가셋 v2.0 에 추가한 1차 자료 문항 (나머지는 뉴스로 만든 문항). 1차 자료 문항 실험의 --ids 와 같다
PRIMARY_QUESTION_IDS: frozenset[str] = frozenset({"f021", "f023", "f024", "f026", "f027", "t013", "t014", "c005"})


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

RETRIEVAL_COMPARISONS += (
    RetrievalComparison(
        key="decompose-v1", title="비교 질문 분해 v1", eval_set="v2.1",
        baseline=RunRef("20261104-0636_hybrid-c500-kb2-v2.1", "분해 없음"),
        candidate=RunRef("20261104-1541_hybrid-c500-ctx2-kb2-dec-r1", "질문 분해 v1 (번갈아 병합)"),
        decision="비교가 아닌 질문까지 쪼개 Hit@3 0.938 → 0.781, 양쪽 근거 포함(all@5) 0.40 그대로 → 채택 안 함, v2 로 수정",
    ),
    RetrievalComparison(
        key="decompose-v2", title="비교 질문 분해 v2", eval_set="v2.1",
        baseline=RunRef("20261104-0636_hybrid-c500-kb2-v2.1", "분해 없음"),
        candidate=RunRef("20261104-1558_hybrid-c500-ctx2-kb2-dec2-r1", "질문 분해 v2 (빠진 대상만 보강)"),
        decision="Hit@3·MRR 유지, 양쪽 근거 포함(all@5) 0.40 → 0.60 (3회 같음) → 채택",
    ),
)

# 답변 근거 청크 수 비교 (뉴스 지식베이스, 각 1회)
CONTEXT_CHUNK_COMPARISON = (
    RunRef("20261103-1649_hybrid-c500-answers", "문서당 청크 1개"),
    RunRef("20261103-1711_hybrid-c500-ctx2-answers", "문서당 청크 2개"),
)

ANSWER_SETTINGS: tuple[AnswerSetting, ...] = (
    AnswerSetting(
        key="news", label="뉴스만", short="뉴스만",
        repetitions=(
            ("20261104-1106_hybrid-c500-ctx2-news-r1", "20261104-1129_primary8-news-ans-r1"),
            ("20261104-1108_hybrid-c500-ctx2-news-r2", "20261104-1129_primary8-news-ans-r2"),
            ("20261104-1110_hybrid-c500-ctx2-news-r3", "20261104-1130_primary8-news-ans-r3"),
        ),
    ),
    AnswerSetting(
        key="kb2", label="1차 자료 포함 (프롬프트 v1)", short="v1",
        repetitions=(
            ("20261104-1040_hybrid-c500-ctx2-kb2-answers", "20261104-1127_primary8-kb2-ans-r1"),
            ("20261104-1050_hybrid-c500-ctx2-kb2-answers-r2", "20261104-1128_primary8-kb2-ans-r2"),
            ("20261104-1055_hybrid-c500-ctx2-kb2-answers-r3", "20261104-1128_primary8-kb2-ans-r3"),
        ),
    ),
    AnswerSetting(
        key="kb2-p2", label="1차 자료 포함, 프롬프트 v2", short="v2",
        repetitions=(
            ("20261104-1436_hybrid-c500-ctx2-kb2-p2-r1",),
            ("20261104-1438_hybrid-c500-ctx2-kb2-p2-r2",),
            ("20261104-1440_hybrid-c500-ctx2-kb2-p2-r3",),
        ),
    ),
    AnswerSetting(
        key="kb2-p3", label="1차 자료 포함, 프롬프트 v3", short="v3",
        repetitions=(
            ("20261104-1503_hybrid-c500-ctx2-kb2-p3-r1",),
            ("20261104-1504_hybrid-c500-ctx2-kb2-p3-r2",),
            ("20261104-1506_hybrid-c500-ctx2-kb2-p3-r3",),
        ),
    ),
    AnswerSetting(
        key="kb2-p2-seed", label="프롬프트 v2 + seed 42", short="v2+seed",
        repetitions=(
            ("20261104-1443_hybrid-c500-ctx2-kb2-p2-seed-r1",),
            ("20261104-1445_hybrid-c500-ctx2-kb2-p2-seed-r2",),
            ("20261104-1447_hybrid-c500-ctx2-kb2-p2-seed-r3",),
        ),
    ),
    AnswerSetting(
        key="kb2-dec2", label="1차 자료 포함, 비교 질문 분해 v2", short="분해",
        repetitions=(
            ("20261104-1558_hybrid-c500-ctx2-kb2-dec2-r1",),
            ("20261104-1600_hybrid-c500-ctx2-kb2-dec2-r2",),
            ("20261104-1602_hybrid-c500-ctx2-kb2-dec2-r3",),
        ),
    ),
)
ANSWER_COMPARISONS: tuple[AnswerComparison, ...] = (
    AnswerComparison(key="primary", title="1차 자료 추가", baseline="news", candidate="kb2",
                     decision="뉴스 문항은 변동 범위 안(t007 제외), 1차 자료 문항에서 크게 앞서 유지"),
    AnswerComparison(key="prompt", title="프롬프트 v2 (날짜 종류 구분)", baseline="kb2", candidate="kb2-p2",
                     decision="정답률 같음: t012 개선·t007 일부 개선, f006 잘못된 거절 → 채택하지 않음"),
    AnswerComparison(key="prompt-v3", title="프롬프트 v3 (상대 날짜)", baseline="kb2", candidate="kb2-p3",
                     decision="회귀 없이 정답률 +3.1%p(1문항)지만 목표 t007 은 3회 모두 부분 → 기준 미달로 v1 유지, 후보로 기록"),
    AnswerComparison(key="seed", title="seed 고정", baseline="kb2-p2", candidate="kb2-p2-seed",
                     decision="실행 간 같은 답변 비율이 그대로 → 변동을 줄이지 못함, 쓰지 않음"),
    AnswerComparison(key="decompose", title="비교 질문 분해 v2", baseline="kb2", candidate="kb2-dec2",
                     decision="비교 정답률 20 → 40%(c002 정확, c001·c004 오답 → 부분), 등급 회귀 없음 → 채택. "
                              "t011·c003 은 잘못된 거절이 틀린 답으로 바뀜 (한계로 기록)"),
)
ADOPTED_ANSWER_SETTING = "kb2-dec2"

# 거절 임계값 근거: 답변 불가 문항까지 포함한 검색 실험 (30문항)
THRESHOLD_RUNS: tuple[RunRef, ...] = (
    RunRef("20261104-1035_hybrid-c500-kb2-pdffix", "하이브리드 + 1차 자료 (현재)"),
    RunRef("20261104-1019_hybrid-c500-v1.3", "하이브리드, 뉴스만"),
    RunRef("20261103-1702_baseline-c500-v1.2", "의미 검색 기준선"),
)
