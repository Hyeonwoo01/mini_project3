"""핵심 법령 현행 조문 수집 (국가법령정보센터).

정책·제도 질문의 최종 근거가 되는 법령 3~5개만 수집한다. 후보는 README 지식베이스 구성 참조.
- 법령명·시행일·조문 번호를 메타데이터로 보존한다 (조문 단위 청킹에 사용)
- Open API 이용 조건과 신청 절차는 Day 1 소스 확인에서 확정하고, 크롤링보다 API를 우선한다
- 시행일을 published_at 으로 기록하여 시점 질문에서 개정 전후를 구분할 수 있게 한다
"""


def run(law_names: list[str]) -> None:
    raise NotImplementedError
