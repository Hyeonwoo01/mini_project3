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

### 직접 수집 소스 접근 확인 (`python -m collector probe`, 2026-11-03, 수집 PC 에서 실행)

| 소스 | robots.txt | 결과 | 판단 |
|---|---|---|---|
| 정책브리핑 보도자료 (`korea.kr/briefing/pressReleaseList.do`) | 허용 | 목록 정적 HTML, 검색어·기간 파라미터 동작. 상세 본문은 문서뷰어 iframe 이라 HTML 에 없고, 원문 PDF·HWPX 첨부(내려받기 허용) | 주 수집원. 부처 보도자료(중기부·금융위 포함)를 검색어·기간으로 찾고 **PDF 원문에서 본문 추출**. 공공누리 제1유형(텍스트) |
| 금융위원회 보도자료 (`fsc.go.kr/no010101`) | 허용 | 목록·상세 정적 HTML, 첨부 `/comm/getFile` | 카드수수료 보도자료 원문 첨부 보강 |
| 중소벤처기업부 보도자료 (`mss.go.kr/.../List.do?cbIdx=86`) | 허용 | 상세 링크가 일반 주소가 아님(`#view`) | 같은 보도자료가 정책브리핑에 있어 별도 수집하지 않음 |
| 서울시 경제분야 누리집 (`news.seoul.go.kr/economy`) | 허용 | 목록·상세 정적 HTML (본문 `#post_content`) | 서울 소상공인 지원사업 안내. 공공누리 제4유형(출처표시·비상업·변경금지) |
| 서울시 보도자료 (`www.seoul.go.kr/news`) | **금지** | 요청하지 않음 | 제외 |
| 국가법령정보센터 Open API (`law.go.kr/DRF`) | 허용 | 응답 확인, 인증값(OC) 필요 | 활용 신청 후 법령 수집 |
| 공공데이터포털 Open API (`apis.data.go.kr`) | 규칙 없음 | 서버 응답 확인 | 필요 시 사용 (서비스키) |

이용약관(공공누리 유형)은 수집 대상 페이지에서 확인해 리포트에 기록한다.

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
│   ├── probe.py            #   수집 후보 사이트 접근 확인 (robots.txt·응답·표본 HTML)
│   ├── load_provided.py    #   제공 데이터 적재 및 선별
│   ├── crawl_korea_kr.py   #   정책브리핑 보도자료 수집 (PDF 원문)
│   ├── crawl_seoul_economy.py # 서울시 경제분야 누리집 소상공인 지원 소식
│   ├── crawl_blog.py       #   블로그 검색 API 수집
│   ├── collect_law.py      #   법령 현행 조문 수집
│   └── storage.py          #   원본 파일 저장, doc_id 생성
├── indexer/                # ② 인덱싱 계층 (배치)
│   ├── __main__.py
│   ├── prepare.py          #   정제 → 품질 필터 → 중복 처리 (processed_documents) + 키워드 색인
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
│   ├── pool.py             #   정답 문서 보정 (검색 결과 판정 → 평가셋 반영)
│   ├── grade.py            #   답변 직접 채점 집계 · 이전 채점 옮기기
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

직접 수집 크롤러를 만들기 전에, 수집할 PC 에서 후보 사이트의 robots.txt 와 응답을 확인한다.
결과는 `data/raw/_probe/report.txt`, 받은 HTML 은 같은 폴더에 저장된다 (파서 작성용 표본).

```bash
python -m collector probe
```

정책브리핑 보도자료는 검색어·기간(기본: 제공 뉴스와 같은 2021-10-16 ~ 2026-10-16)으로 찾고 부처명과
제목 키워드로 거른다. 검색이 본문·첨부까지 찾아 지나가듯 언급한 보도자료도 나오기 때문이다
('카드수수료' 148건 중 다수가 가상자산·중동 상황 점검 등). `--dry-run` 은 받을 목록만 보여 준다.
PDF 원문(없으면 HWPX)을 `data/raw/press_release/` 에 저장하고, 본문 추출은 인덱서(`indexer prepare`)가 한다.
옛 HWP 만 첨부된 보도자료는 받지 않고 '원문 없음'으로 보고한다.
이미 받은 문서는 건너뛰므로 중단되면 같은 명령을 다시 실행한다. 먼저 `--count-only` 로 결과 수를 확인한다.

```bash
python -m collector korea-kr --query 소상공인 --ministry 중소벤처기업부 --count-only
```

```bash
python -m collector korea-kr --query 소상공인 --ministry 중소벤처기업부
```

서울시 경제분야 누리집의 소기업·소상공인 종합지원 새소식은 본문을 수집 시점에 추출해 저장한다
(본문 영역 태그가 닫히지 않아 태그 목록 앞에서 잘라야 하는 등 사이트 구조에 맞춘 처리라 수집기가 맡는다).

```bash
python -m collector seoul-economy
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

먼저 `--dry-run` 으로 청크 수·길이 분포를 확인한 뒤 임베딩한다. 중간에 API 오류로 멈추면 같은 명령을 다시 실행하면 이어서 적재한다.

```bash
python -m indexer build --chunk-config c500 --dry-run
```

```bash
python -m indexer build --chunk-config c500
```

```bash
python -m indexer build --chunk-config c1000
```

```bash
python -m indexer verify --collection docs__c500
```

**평가** — 결과: `eval/results/<일시>_<태그>/`

청킹 설정별로 같은 평가셋을 돌려 Hit@1·Hit@3·Hit@5·MRR 을 비교한다. 검색은 챗봇과 같은 `rag.retriever` 로 한다.

```bash
python -m eval.run_eval --tag baseline-c1000 --collection docs__c1000
```

```bash
python -m eval.run_eval --tag baseline-c500 --collection docs__c500
```

하이브리드 검색(의미 검색 + 키워드 BM25, RRF 결합)은 `--mode hybrid` 로 비교한다. 키워드 색인은 `indexer prepare` 가 만든다.

```bash
python -m eval.run_eval --tag hybrid-c500 --collection docs__c500 --mode hybrid
```

정답 목록만 고쳤을 때는 검색을 다시 하지 않고 저장된 결과를 다시 채점한다 (원문 DB·벡터 DB·API 불필요).

```bash
python -m eval.run_eval --rescore eval/results/<실험> --tag <새 태그>
```

답변까지 생성해 거절 정확도를 집계하고, 직접 채점용 `answers.csv` 를 만든다 (챗봇과 같은 `rag.chain` 사용).

```bash
python -m eval.run_eval --tag hybrid-c500-answers --with-answers
```

`answers.csv` 의 grade(정확/부분/오답)·citation_ok(Y/N/-)·note 를 직접 채점한 뒤 집계한다. 다음 실험에서는 답변이 바뀌지 않은 문항의 채점을 옮겨 온다.

```bash
python -m eval.grade summary eval/results/<실험>/answers.csv
```

```bash
python -m eval.grade carry eval/results/<이전 실험>/answers.csv eval/results/<새 실험>/answers.csv
```

답변 근거로 넣는 문서당 청크 수는 `--context-chunks` 로 바꿔 비교한다 (기본 `RAG_CONTEXT_CHUNKS=1`). 검색 순위는 그대로이고 같은 기사의 세부 청크만 더 넣는다.

```bash
python -m eval.run_eval --tag hybrid-c500-ctx2-answers --with-answers --context-chunks 2
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

작성 절차와 문항 재료는 [`eval/README.md`](eval/README.md) 참조. 정답 문서는 `python -m eval.find_docs <검색어>` 로 찾고 `python -m eval.run_eval --check` 로 검사한다. 실험 결과에서 정답 목록 밖 문서를 판정해 보정하는 절차는 같은 문서의 '정답 문서 보정' 참조.

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

## 실험 기록

검색 평가 (답변 가능 24문항, k=5). 결과 파일은 `eval/results/`.

| 실험 | 평가셋 | Hit@1 | Hit@3 | Hit@5 | MRR | 평균 응답 |
|---|---|---|---|---|---|---|
| c1000 의미 검색 | v1 | 0.083 | 0.333 | 0.458 | 0.222 | 180ms |
| c500 의미 검색 (기준선) | v1 | 0.208 | 0.458 | 0.583 | 0.356 | 257ms |
| c500 하이브리드 | v1 | 0.542 | 0.667 | 0.708 | 0.599 | 350ms |
| c500 의미 검색 (기준선) | v1.2 | 0.458 | 0.667 | 0.750 | 0.576 | 235ms |
| c500 하이브리드 | v1.2 | 0.792 | 0.917 | 0.917 | 0.854 | 376ms |
| c500 하이브리드 (뉴스만) | v1.3 | 0.792 | 0.917 | 0.917 | 0.854 | 376ms |
| **c500 하이브리드 + 1차 자료 (채택)** | v1.3 | **0.833** | **0.917** | **0.917** | **0.875** | 420ms |

1. **청킹 설정**: c500 이 모든 지표에서 앞서 채택 (v1, 같은 조건 비교)
2. **실패 분석** (c500 기준선, Hit@3 실패 13문항): 질문의 연도·기관명 무시 5, 비교 문항에서 한쪽만 검색 2,
   관련 기사를 찾았지만 정답 목록에 없음 5, 기타 2
3. **평가셋 보정 v1 → v1.1** (`eval/pool.py`): 기준선·하이브리드 상위 3건 중 정답 목록 밖 83건을 판정해
   27건(13문항)을 정답에 추가, 56건은 정답 아님으로 기록 (`eval/pool_judged.json`). 두 실험 결과를 함께 판정해 공정하게 비교.
   **v1.1 → v1.2**: 답변 평가에서 발견한 판정 오류 2건(f020) 수정. 검색은 다시 하지 않고 저장된 결과를
   다시 채점했다 (`run_eval --rescore`)
4. **개선: 하이브리드 검색** — 키워드 색인(한글 2-gram·숫자)과 의미 검색을 RRF 로 결합. v1.2 기준 Hit@3 +0.250,
   MRR +0.278. 응답 시간은 평균 +140ms. 유형별 Hit@3: fact 0.6 → 1.0, temporal 0.667 → 0.833,
   comparison 0.5 → 0.75, synthesis 1.0 → 1.0
5. **1차 자료 추가** (정책브리핑 보도자료 487건·서울시 공고 32건, 청크 50,594 → 55,332): 보정 전(v1.2)에는
   Hit@3 0.917 → 0.875 로 떨어진 것처럼 보였다. 새로 검색된 보도자료가 정답 목록에 없었기 때문이다. 상위 3건을
   판정해 4건을 정답에 추가한 v1.3 에서 같은 기준으로 비교하면 Hit@1 0.792 → 0.833 (t010 정답이 2위 → 1위,
   점포철거비 600만원 상향 보도자료), Hit@3·Hit@5 는 같다. 응답 시간 +44ms.
   1차 자료가 상위 5건에 드는 문항은 30문항 중 5개(1위 2개)뿐이다. 평가셋을 뉴스로 만들었기 때문으로,
   1차 자료로 답하는 문항을 추가해 측정할 필요가 있다 (`TODO.md`)
6. **한계**: synthesis 는 한쪽 근거만 있어도 정답으로 판정해 검색 지표가 포화(1.0)되므로 답변 채점으로 평가한다.
   판정 깊이가 3이라 Hit@5·MRR 은 보수적이다. 답변 가능 문항 24개라 1문항이 약 0.04 에 해당한다

답변 평가 (하이브리드 c500, gpt-4o-mini, RAG_MIN_SCORE 0.5, 직접 채점). 개선 실험: 답변 근거의 문서당 청크 수 1 → 2

| 유형 | 청크 1개 정확/부분/오답 | 청크 2개 정확/부분/오답 | 정답률 1 → 2 |
|---|---|---|---|
| fact (10) | 7 / 2 / 1 | 9 / 1 / 0 | 70% → 90% |
| temporal (6) | 3 / 2 / 1 | 3 / 1 / 2 | 50% → 50% |
| comparison (4) | 0 / 1 / 3 | 0 / 1 / 3 | 0% → 0% |
| synthesis (4) | 2 / 2 / 0 | 2 / 2 / 0 | 50% → 50% |
| **답변 가능 계 (24)** | **12 / 7 / 5** | **14 / 5 / 5** | **50% → 58%** (부분 포함 79% → 79%) |
| 답변 불가 (6) | 거절 6 | 거절 6 | 거절 정확도 100% |
| 출처 정확도 | 86% (18/21) | **95%** (20/21) | |
| 평균 응답 시간 | 2.4초 | 2.8초 | |

결과: `eval/results/20261103-1649_hybrid-c500-answers/`(청크 1개), `eval/results/20261103-1711_hybrid-c500-ctx2-answers/`(청크 2개)

- **청크 2개 채택**: 근거 청크 부족으로 틀린 f006(오답 → 정확)·f016(부분 → 정확)이 해결되고 c002 는 출처 표기가
  정확해졌다. 대신 t009 가 부분 → 오답으로 악화 — 2025년 문서가 검색되지 않은 상태에서 늘어난 2026년 근거를
  2025년 사업으로 잘못 기술했다 (시점 혼동). 잘못된 거절 3문항(t011·c003·c004)은 검색 단계 문제라 변화 없음
- 남은 부분·오답 원인: 검색 실패(t011·c003·t009·c001), 거절 판단 불일치(c004 — 한쪽 근거가 있어도 전체 거절),
  시점 혼동(s006·t009), 출처 품질(c001 — AI 생성 기사 인용, 근거 없는 일반론), synthesis 한쪽 편향(s004),
  근거 부족(t010 — 임계값을 통과한 근거 2건)

## 진행 현황

남은 작업과 근거는 [`TODO.md`](TODO.md).

- [x] 프로젝트 구조, 공용 설정·스키마·원문 DB, 평가 지표(Hit@k·MRR)
- [x] 주제·사용자·범위 정의
- [x] 수집 소스 접근 가능 여부 확인 (`collector probe`)
- [x] 제공 데이터 적재 모듈, EDA 노트북
- [x] 제공 데이터 EDA 실행 → 품질 필터 v1 확정
- [x] 정제·품질 필터·중복 처리 (`indexer prepare`)
- [x] 중복 처리 기준 확정 (EDA 12·13번, dedup v2) → 뉴스 지식베이스 14,467건
- [ ] 대표 질문 확정
- [x] 직접 수집 (1차 자료): 정책브리핑 보도자료 487건 (중기부 451·금융위 18·행안부 18), 서울시 경제분야 누리집 공고 36건
- [x] 청킹 설정 2개 (c500·c1000) + 제목 제외 비교 설정, 임베딩·적재·검증 (`indexer build/verify`)
- [x] 원본 파일(1차 자료) 파싱: PDF (PyMuPDF), HWPX (구역 XML)
- [x] 평가셋 작성 (30문항: fact 10 · temporal 6 · comparison 4 · synthesis 4 · unanswerable 6)
- [x] 실제 임베딩 적재 및 검증 (docs__c500 50,594청크 · docs__c1000 26,816청크)
- [x] 검색기(`rag/retriever.py`)·검색 평가 하네스(`eval/run_eval.py`)
- [x] 1차 측정: c500 Hit@3 0.458 · MRR 0.356 > c1000 Hit@3 0.333 · MRR 0.222 → c500 선택
- [x] 하이브리드 검색(키워드 BM25 + 의미 검색, RRF)·정답 문서 보정 도구(`eval/pool.py`)
- [x] 정답 문서 보정(평가셋 v1.1 → 판정 오류 수정 v1.2) 후 재측정 → 하이브리드 Hit@3 0.917 · MRR 0.854, 기본 검색 방식으로 채택
- [x] RAG 체인·거절 로직(유사도 임계값 + 프롬프트)·출처 번호 표기, 답변 평가(`--with-answers`)
- [x] 거절 임계값(`RAG_MIN_SCORE`) 0.5 로 조정 (1위 유사도 분포 기준)
- [x] 답변 생성 평가·직접 채점: 정답률 50% (부분 포함 79%), 거절 정확도 100%, 잘못된 거절 12.5%, 출처 정확도 86%
- [x] 답변 품질 개선 실험: 문서당 청크 2개 컨텍스트 → 정답률 50% → 58%, 출처 정확도 86% → 95%, 기본값으로 채택
- [ ] 1차 자료 포함 지식베이스 재구축·재측정 (prepare → build → 평가셋 점검 → 재측정)
- [ ] Streamlit 챗 UI
- [ ] 직접 채점, 개선 및 재측정
- [ ] 평가 결과 시각화
- [ ] 리포트
