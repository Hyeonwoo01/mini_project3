"""블로그 수집 (네이버·카카오 검색 API).

검색 API 는 제목·요약문·원문 링크만 제공한다. 본문 수집 여부는 플랫폼 이용약관과
robots.txt 를 확인한 뒤 결정하고, 제한되는 경우 요약문 수준으로 한정하며 그 사유를
리포트에 기술한다 (가이드 3-3). 광고·홍보성 문서 판별에 쓸 수 있도록 출처 정보를 보존한다.
"""


def run(queries: list[str], max_results: int) -> None:
    raise NotImplementedError
