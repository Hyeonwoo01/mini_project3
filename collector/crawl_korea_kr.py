"""대한민국 정책브리핑(https://www.korea.kr/) 보도자료 수집.

부처 보도자료 원문은 지원사업 요건·규모 등 1차 근거에 해당한다 (가이드 3-3).
수집 대상: 중소벤처기업부·소상공인시장진흥공단 소상공인 정책, 금융위원회 카드수수료
(영세·중소가맹점 우대) 보도자료. 정책브리핑에 없는 첨부 원문은 부처 홈페이지를 확인한다.
수집 전 robots.txt 와 이용약관을 확인하고, 요청은 collector/fetch.py 를 통해서만 보낸다.
원본 HTML/첨부 PDF 는 storage.save_raw() 로 저장하고 메타데이터는 docs.db 에 등록한다.
"""


def run(queries: list[str], max_pages: int) -> None:
    raise NotImplementedError
