"""수집 계층 실행 진입점.

    python -m collector probe                 # 수집 후보 사이트 접근 확인 (크롤러 작성 전)
    python -m collector provided              # 제공 뉴스 데이터 적재
    python -m collector korea-kr --query ...  # 정책브리핑 보도자료 수집
    python -m collector blog --query ...      # 블로그 검색 API 수집
    python -m collector law --name ...        # 핵심 법령 현행 조문 수집
"""

import argparse

from collector import collect_law, crawl_blog, crawl_korea_kr, load_provided, probe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m collector", description="① 수집 계층")
    sub = parser.add_subparsers(dest="command", required=True)

    probe_cmd = sub.add_parser("probe", help="수집 후보 사이트의 robots.txt·응답 확인, 표본 HTML 저장")
    probe_cmd.add_argument("--only", nargs="+", metavar="KEY",
                           help=f"확인할 후보 ({', '.join(t.key for t in probe.PROBE_TARGETS)})")

    provided = sub.add_parser("provided", help="제공 데이터(뉴스 CSV) 적재")
    provided.add_argument(
        "--file", action="append", help="적재할 CSV 경로 (생략 시 data/provided/ 에서 지정 파일 자동 탐색)"
    )

    korea_kr = sub.add_parser("korea-kr", help="대한민국 정책브리핑 보도자료 수집")
    korea_kr.add_argument("--query", action="append", required=True, help="검색어 (반복 지정 가능)")
    korea_kr.add_argument("--max-pages", type=int, default=5)

    blog = sub.add_parser("blog", help="블로그 검색 API 수집")
    blog.add_argument("--query", action="append", required=True, help="검색어 (반복 지정 가능)")
    blog.add_argument("--max-results", type=int, default=100)

    law = sub.add_parser("law", help="국가법령정보센터 법령 현행 조문 수집")
    law.add_argument("--name", action="append", required=True, help="법령명 (반복 지정 가능)")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "probe":
        probe.run(only=args.only)
    elif args.command == "provided":
        load_provided.run(files=args.file)
    elif args.command == "korea-kr":
        crawl_korea_kr.run(queries=args.query, max_pages=args.max_pages)
    elif args.command == "blog":
        crawl_blog.run(queries=args.query, max_results=args.max_results)
    elif args.command == "law":
        collect_law.run(law_names=args.name)


if __name__ == "__main__":
    main()
