"""제공 데이터(뉴스) 적재 및 선별.

data/provided/ 의 원본을 읽어 DocumentRecord(doc_type=news, collect_method=provided)로
변환한 뒤 docs.db 에 적재한다. 제공 데이터의 형식은 Day 1 배포 후 EDA 로 확인하고 구현한다.

- 원본의 모든 행을 적재한다. 중복 제거는 indexer/dedup.py 에서 수행해야 처리 전후 건수가 남는다.
- 주제와 무관한 기사 선별 기준을 적용하는 경우 기준과 제외 건수를 기록한다.
"""


def run() -> None:
    raise NotImplementedError
