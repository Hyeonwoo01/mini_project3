# mini_project3
AI.M 데이터분석 트랙 미니 프로젝트 3 — 공공 문서·웹검색 결합 RAG 기반 질의응답 시스템

> 이 과제의 기준은 "챗봇이 동작하는가"가 아니라
> **"이 챗봇의 답변을 신뢰할 수 있는가, 그 판단의 근거는 무엇인가"** 에 측정으로 답하는 것이다.

## 시스템 정의

| 항목 | 내용 |
|---|---|
| 주제 | A. 지역 상권·소상공인 정책 |
| 대상 사용자 | 카드사 상권·가맹점 데이터 분석가 |
| 업무 상황 | 상권·가맹점 지표에서 변화나 이상치를 발견했을 때, 배경이 되는 정책·이슈를 근거 문서로 확인하고 분석 리포트에 인용한다 |
| 지역 범위 | 정책은 전국 단위, 지자체 사업은 서울시·자치구 |
| 업종 범위 | 전 업종. 카페·외식업은 미니프로젝트 2와 이어지는 대표 시나리오로 다룬다 |

미니프로젝트 1(서울 자치구 인구 구조 × 상권 업종)과 미니프로젝트 2(생활인구 기반 2030 카페 입지 대시보드)의
수치 분석 결과에 정책·시장 맥락을 근거와 함께 결합한다. 지식베이스는 공개 문서로만 구성하며
카드 결제 데이터 등 비공개 자료는 사용하지 않는다.

### 지식베이스 구성 (계획)

| 구분 | 자료 | 문서 종류 |
|---|---|---|
| 제공 | 소상공인 묶음 뉴스 42,453건 (소상공인 지원정책·상권 활성화·전통시장 활성화·골목상권·소상공인 폐업) | `news` |
| 제공 | 지역화폐 뉴스 36,888건 — 가이드의 주제 A 키워드 '지역사랑상품권'을 대신함 | `news` |
| 직접 수집 | 중소벤처기업부·소상공인시장진흥공단 보도자료 | `press_release` |
| 직접 수집 | 금융위원회 카드수수료(영세·중소가맹점 우대) 보도자료 | `press_release` |
| 직접 수집 | 서울시·자치구 상권 지원사업 공고 | `announcement` |
| 직접 수집 | 국책·지자체 연구기관 보고서 (NKIS, 서울연구원 등) | `policy_report` |
| 직접 수집 | 핵심 법령 현행 조문 3~5개 (국가법령정보센터) | `law` |
| 직접 수집 (선택) | 블로그 — 소상공인 체감 반응 | `blog` |

법령 후보 (Day 1 확정): 지역상권 상생 및 활성화에 관한 법률, 전통시장 및 상점가 육성을 위한 특별법,
여신전문금융업법(우대수수료 근거), 상가건물 임대차보호법, 지역사랑상품권 이용 활성화에 관한 법률.
국회 법안·검토보고서가 아니라 현행 조문만 넣는다. 분석가에게 필요한 것은 현재 적용 기준이기 때문이다.

제공 뉴스는 기간 2021-10-16 ~ 2026-10-16이며, 가이드의 주제 A 키워드 중 핵심 5개만 키워드로 수집되었다.
정책자금·상권활성화구역·공실·임대료·젠트리피케이션·생활인구 등은 별도 수집되지 않았으므로
EDA(`notebooks/eda.ipynb` 8번)로 본문 등장 건수를 확인하고, 부족한 영역을 직접 수집으로 보강한다.
서울 범위·품질 필터와 중복 처리는 적재 후 인덱서에서 적용한다 (처리 전후 건수 기록).

소스별 접근 가능 여부와 이용약관은 Day 1에 확인하고 결과를 리포트에 기록한다.

### 대표 질문 (초안 — EDA 후 확정)

| 유형 | 질문 |
|---|---|
| 사실 | 상권활성화구역 지정 요건은? / 영세·중소가맹점의 매출 구간 기준과 우대수수료율은? |
| 시점 | 올해 달라진 소상공인 정책자금 지원 내용은? |
| 비교 | 서울 자치구별 골목상권 지원사업의 차이는? |
| 종합·판단 | 지역사랑상품권이 소상공인 매출에 주는 효과에 대한 상반된 평가는? |
| 답변 불가 | 특정 동 카페의 월평균 카드 매출은? (비공개 데이터 — 거절이 정답) |

## 아키텍처

```
 ┌──────────┐      ┌──────────┐      ┌──────────────┐      ┌──────────┐
 │ ① 수집기  │ ───▶ │ ② 인덱서  │ ───▶ │ ③ 저장 계층    │ ◀─── │ ④ 챗봇    │
 │ collector│      │ indexer  │      │ store/       │ 조회  │ app      │
 └──────────┘      └──────────┘      │  docs.db     │      └────┬─────┘
    배치               배치           │  chroma/     │           │ rag.chain
                                     └──────────────┘      ┌────┴─────┐
                                                           │ 평가 하네스 │ eval
                                                           └──────────┘
```

- 데이터 흐름은 단방향이다. 챗봇은 인덱스에 쓰지 않고, 임베딩은 **질문 벡터화에만** 사용한다.
- `rag/` 를 챗봇과 평가 하네스가 공유한다. 평가한 체인과 서비스하는 체인이 같아야 평가가 유효하다.
- 계층 간 공유가 필요한 요소(설정, 스키마, 임베딩 모델, 원문 DB 접근)는 `common/` 에만 둔다.
  `rag` 가 `indexer` 를 import 하면 "인덱서 없이 챗봇 구동" 요건이 깨지기 때문이다.
- 계층 분리 규칙은 `tests/test_layer_boundaries.py` 가 정적으로 검사한다.

### 원문 DB(`store/docs.db`) 테이블 소유권

| 테이블 | 기록 계층 | 내용 |
|---|---|---|
| `documents` | collector | 수집 원문, 출처·제목·URL·발행일·문서 종류·수집 일시 |
| `processed_documents` | indexer | 정제 본문, 중복 판정 결과(`kept`/`duplicate`/`excluded`) |

`rag`/`app`/`eval` 은 `open_db(read_only=True)` 로만 연다 (SQLite 수준에서 쓰기 차단).

## 디렉터리 구조

```
mini_project3/
├── common/                 # 계층 공용
│   ├── config.py           #   경로·모델·크롤링 설정 (.env 로드)
│   ├── schema.py           #   문서 종류, 출처 신뢰도, 청크 메타데이터 필드
│   ├── embeddings.py       #   임베딩 모델 생성 (indexer·rag 공용)
│   └── docstore.py         #   원문 DB 접근
├── collector/              # ① 수집 계층 (배치)
│   ├── __main__.py
│   ├── fetch.py            #   robots.txt·딜레이를 강제하는 공용 요청 모듈
│   ├── load_provided.py    #   제공 데이터 적재 및 선별
│   ├── crawl_korea_kr.py   #   정책브리핑 보도자료 수집
│   ├── crawl_blog.py       #   블로그 검색 API 수집
│   ├── collect_law.py      #   법령 현행 조문 수집
│   └── storage.py          #   원본 파일 저장, doc_id 생성
├── indexer/                # ② 인덱싱 계층 (배치)
│   ├── __main__.py
│   ├── prepare.py          #   정제 → 품질 필터 → 중복 처리 (processed_documents)
│   ├── parse.py            #   PDF·HTML 텍스트 추출, 보일러플레이트 제거
│   ├── quality.py          #   품질 필터 (범위 밖·불량 제외, 신뢰도 플래그)
│   ├── dedup.py            #   완전 중복·재배포본 처리 (본문 유사도)
│   ├── chunk.py            #   문서 종류별 청킹 설정
│   └── embed.py            #   임베딩·Chroma 적재·검증
├── store/                  # ③ 저장 계층 (git 미포함, 인덱서 실행으로 생성)
│   ├── docs.db
│   └── chroma/
├── rag/                    # 체인 (챗봇과 평가가 공유)
│   ├── retriever.py
│   ├── chain.py
│   └── prompts.py
├── eval/                   # 평가 하네스
│   ├── README.md           #   평가셋 작성 가이드
│   ├── eval_set.jsonl
│   ├── find_docs.py        #   정답 문서 검색
│   ├── run_eval.py
│   └── results/            #   실험별 결과 (git 포함)
├── app/                    # ④ 챗봇 UI (조회 전용)
│   ├── main.py
│   ├── chat.py
│   └── dashboard.py
├── data/                   # 제공 데이터·크롤링 원본 (git 미포함)
│   ├── provided/
│   └── raw/
├── notebooks/eda.ipynb     # 문서 탐색
├── report/report.md        # 리포트
└── tests/
```

## 설치

Python 3.11 이상이 필요하다.

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

```bash
pip install -r requirements-dev.txt
```

```bash
cp .env.example .env        # 이후 .env 에 OPENAI_API_KEY 등을 채운다
```

## 실행 순서

모든 명령은 프로젝트 루트에서 실행한다. 각 계층은 독립적으로 실행된다.

**① 수집** — 결과: `store/docs.db` 의 `documents`, `data/raw/`

제공 CSV(소상공인 묶음, 지역화폐)를 **원래 파일명 그대로** `data/provided/` 에 넣은 뒤 적재한다.
파일명의 추출 기간을 수집 일시로 쓰기 때문이다.

```bash
python -m collector provided
```

```bash
python -m collector korea-kr --query "검색어"
```

```bash
python -m collector blog --query "검색어"
```

```bash
python -m collector law --name "법령명"
```

**② 인덱싱** — 결과: `processed_documents`, `store/chroma/` 의 `docs__<설정명>` 컬렉션

```bash
python -m indexer prepare
```

품질 필터 없이 실행하면 개선 전 기준 실험이 된다 (`--quality none`). 결과 요약의 사유별 건수가 처리 전후 건수다.

```bash
python -m indexer prepare --quality none
```

```bash
python -m indexer build --chunk-config baseline
```

```bash
python -m indexer verify --collection docs__baseline
```

**평가** — 결과: `eval/results/<일시>_<태그>/`

```bash
python -m eval.run_eval --tag baseline --collection docs__baseline
```

**④ 챗봇** — 인덱싱을 다시 실행하지 않고 단독으로 구동된다

```bash
streamlit run app/main.py
```

**테스트**

```bash
pytest
```

## 평가셋 형식

작성 절차와 문항 재료는 [`eval/README.md`](eval/README.md) 참조. 정답 문서는 `python -m eval.find_docs <검색어>` 로 찾고 `python -m eval.run_eval --check` 로 검사한다.

`eval/eval_set.jsonl` — 한 줄에 한 문항. `id` 는 실험 간 같은 문항을 대조하는 키이므로 바꾸지 않는다.

```jsonl
{"id": "f001", "q": "...", "type": "fact", "gold_doc_ids": ["kr_20260115_1a2b3c4d"], "answer_key": "..."}
{"id": "u001", "q": "...", "type": "unanswerable", "gold_doc_ids": [], "answer_key": "지식베이스에 근거 없음. 답변 거절이 정답"}
```

| type | 검증 대상 |
|---|---|
| `fact` | 기본 검색 정확도 |
| `comparison` | 복수 문서 종합 |
| `temporal` | 발행일 정보 활용 |
| `synthesis` | 상충 근거 처리 |
| `unanswerable` | 답변 거절 (검색 지표에서는 제외, 거절 정확도로 측정) |

## 진행 현황

- [x] 프로젝트 구조, 공용 설정·스키마·원문 DB, 평가 지표(Hit@k·MRR)
- [x] 주제·사용자·범위 정의
- [ ] 수집 소스 접근 가능 여부 확인
- [x] 제공 데이터 적재 모듈, EDA 노트북
- [x] 제공 데이터 EDA 실행 → 품질 필터 v1 확정
- [x] 정제·품질 필터·중복 처리 (`indexer prepare`)
- [ ] 중복 처리 임계값 확정 (EDA 12번), 대표 질문 확정
- [ ] 직접 수집 (1차 자료)
- [ ] 원본 파일 파싱, 청킹 (설정 2개 이상)
- [ ] 평가셋 작성
- [ ] 임베딩·Chroma 적재 및 검증
- [ ] 검색기·평가 하네스 1차 측정
- [ ] RAG 체인·거절 로직·출처 표기
- [ ] Streamlit 챗 UI
- [ ] 직접 채점, 개선 및 재측정
- [ ] 평가 결과 시각화
- [ ] 리포트
