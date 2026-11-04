"""수집 계층 실행 진입점.

    python -m collector probe                 # 수집 후보 사이트 접근 확인 (크롤러 작성 전)
    python -m collector provided              # 제공 뉴스 데이터 적재
    python -m collector korea-kr --query ... [--ministry ...]  # 정책브리핑 보도자료 수집
    python -m collector seoul-economy         # 서울시 경제분야 누리집 소상공인 지원 소식
    python -m collector blog --query ...      # 블로그 검색 API 수집
    python -m collector law --name ...        # 핵심 법령 현행 조문 수집
"""

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m collector", description="① 수집 계층")
    sub = parser.add_subparsers(dest="command", required=True)

    probe_cmd = sub.add_parser("probe", help="수집 후보 사이트의 robots.txt·응답 확인, 표본 HTML 저장")
    probe_cmd.add_argument("--only", nargs="+", metavar="KEY",
                           help="확인할 후보 key (예: fsc mss). 없는 key 를 주면 가능한 목록을 보여 준다")

    provided = sub.add_parser("provided", help="제공 데이터(뉴스 CSV) 적재")
    provided.add_argument(
        "--file", action="append", help="적재할 CSV 경로 (생략 시 data/provided/ 에서 지정 파일 자동 탐색)"
    )

    korea_kr = sub.add_parser("korea-kr", help="대한민국 정책브리핑 보도자료 수집 (PDF 원문)")
    korea_kr.add_argument("--query", action="append", required=True, help="검색어 (반복 지정 가능)")
    korea_kr.add_argument("--ministry", action="append",
                          help="이 부처의 보도자료만 받는다 (목록에 표시되는 부처명, 반복 지정 가능)")
    korea_kr.add_argument("--title-keyword", action="append",
                          help="제목에 이 단어가 하나라도 있는 보도자료만 받는다 (반복 지정 가능)")
    korea_kr.add_argument("--start", default="2021-10-16", help="검색 시작일 (YYYY-MM-DD)")
    korea_kr.add_argument("--end", default="2026-10-16", help="검색 종료일 (YYYY-MM-DD)")
    korea_kr.add_argument("--max-pages", type=int, help="검색어별 최대 목록 쪽 수 (생략 시 끝까지)")
    korea_kr.add_argument("--count-only", action="store_true", help="검색 결과 수만 확인하고 받지 않는다")
    korea_kr.add_argument("--dry-run", action="store_true",
                          help="목록만 넘기며 조건에 맞는 대상을 보여 주고 받지 않는다")

    seoul = sub.add_parser("seoul-economy", help="서울시 경제분야 누리집 소상공인 지원 소식 수집")
    seoul.add_argument("--list-url", action="append", help="수집할 카테고리 목록 주소 (생략 시 소기업·소상공인 종합지원 새소식)")
    seoul.add_argument("--start", default="2021-10-16", help="이 날짜 이후 등록된 글만 받는다 (YYYY-MM-DD)")
    seoul.add_argument("--max-pages", type=int, help="목록별 최대 쪽 수 (생략 시 끝까지)")
    seoul.add_argument("--dry-run", action="store_true", help="받을 목록만 보여 주고 받지 않는다")

    blog = sub.add_parser("blog", help="블로그 검색 API 수집")
    blog.add_argument("--query", action="append", required=True, help="검색어 (반복 지정 가능)")
    blog.add_argument("--max-results", type=int, default=100)

    law = sub.add_parser("law", help="국가법령정보센터 법령 현행 조문 수집")
    law.add_argument("--name", action="append", required=True, help="법령명 (반복 지정 가능)")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    # 명령에 필요한 모듈만 불러온다 (예: provided 적재에는 HTML 파서가 필요 없다)
    if args.command == "probe":
        from collector import probe
        probe.run(only=args.only)
    elif args.command == "provided":
        from collector import load_provided
        load_provided.run(files=args.file)
    elif args.command == "korea-kr":
        from collector import crawl_korea_kr
        crawl_korea_kr.run(queries=args.query, max_pages=args.max_pages, ministries=args.ministry,
                           title_keywords=args.title_keyword, start=args.start, end=args.end,
                           count_only=args.count_only, dry_run=args.dry_run)
    elif args.command == "seoul-economy":
        from collector import crawl_seoul_economy
        crawl_seoul_economy.run(list_urls=args.list_url, start=args.start, max_pages=args.max_pages,
                                dry_run=args.dry_run)
    elif args.command == "blog":
        from collector import crawl_blog
        crawl_blog.run(queries=args.query, max_results=args.max_results)
    elif args.command == "law":
        from collector import collect_law
        collect_law.run(law_names=args.name)


if __name__ == "__main__":
    main()
