# mini_project3
AI.M 데이터분석 트랙 미니 프로젝트 3 — 공공 문서·웹검색 결합 RAG 기반 질의응답 시스템

> 이 과제의 기준은 "챗봇이 동작하는가"가 아니라
> **"이 챗봇의 답변을 신뢰할 수 있는가, 그 판단의 근거는 무엇인가"** 에 측정으로 답하는 것이다.

- 주제: _(선택 후 기재)_
- 대상 사용자 / 업무 상황: _(Day 1 정의 후 기재)_

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
│   └── storage.py          #   원본 파일 저장, doc_id 생성
├── indexer/                # ② 인덱싱 계층 (배치)
│   ├── __main__.py
│   ├── parse.py            #   PDF·HTML 텍스트 추출, 보일러플레이트 제거
│   ├── dedup.py            #   중복 내용 처리
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
│   ├── eval_set.jsonl
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

```bash
python -m collector provided
```

```bash
python -m collector korea-kr --query "검색어"
```

```bash
python -m collector blog --query "검색어"
```

**② 인덱싱** — 결과: `processed_documents`, `store/chroma/` 의 `docs__<설정명>` 컬렉션

```bash
python -m indexer prepare
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
- [ ] 주제·사용자·대표 질문 정의
- [ ] 제공 데이터 EDA
- [ ] 직접 수집 (1차 자료)
- [ ] 파싱·중복 처리·청킹 (설정 2개 이상)
- [ ] 평가셋 작성
- [ ] 임베딩·Chroma 적재 및 검증
- [ ] 검색기·평가 하네스 1차 측정
- [ ] RAG 체인·거절 로직·출처 표기
- [ ] Streamlit 챗 UI
- [ ] 직접 채점, 개선 및 재측정
- [ ] 평가 결과 시각화
- [ ] 리포트
