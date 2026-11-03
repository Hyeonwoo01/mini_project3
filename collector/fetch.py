"""크롤링 준수 사항을 강제하는 공용 HTTP 요청 모듈 (가이드 3-3).

- robots.txt 확인 후 허용된 경로만 요청한다
- 요청 간 최소 딜레이(CRAWL_DELAY_SEC)를 지키며 병렬 요청을 하지 않는다
- 로그인 이후 페이지·개인정보는 수집하지 않는다
- 응답과 함께 수집 일시를 반환하여 모든 문서에 기록되도록 한다

모든 크롤러(crawl_*.py)는 requests 를 직접 호출하지 않고 이 모듈을 거친다.
"""


def fetch(url: str) -> tuple[bytes, str]:
    """URL 본문과 수집 일시(ISO 8601)를 반환한다."""
    raise NotImplementedError
