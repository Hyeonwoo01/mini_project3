"""인덱싱 계층 실행 진입점.

    python -m indexer prepare                         # 정제·품질 필터·중복 처리 → processed_documents
    python -m indexer prepare --quality none          # 품질 필터 없이 (개선 전 기준 실험)
    python -m indexer build --chunk-config c500 --dry-run   # 청크 수·길이 분포만 (API 호출 없음)
    python -m indexer build --chunk-config c500             # 청킹·임베딩 → docs__c500
    python -m indexer verify --collection docs__c500        # 문서 수·청크 수·빈 청크·샘플 검색 확인

prepare 와 build 를 분리하여, 중복 처리 결과 하나로 여러 청킹 설정을 비교할 수 있게 한다.
"""

import argparse

from indexer import chunk, embed, prepare


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m indexer", description="② 인덱싱 계층")
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare", help="정제·품질 필터·중복 처리")
    prep.add_argument("--quality", default="v1", help="indexer/quality.py 의 설정 이름, 또는 none (필터 없음)")
    prep.add_argument("--dedup", default="v2", help="indexer/dedup.py 의 설정 이름 (v1: 본문 유사도만)")

    build = sub.add_parser("build", help="청킹·임베딩 후 Chroma 적재")
    build.add_argument("--chunk-config", required=True, help="indexer/chunk.py 의 CHUNK_CONFIGS 키")
    build.add_argument("--collection", help="적재할 컬렉션 이름 (기본: docs__<chunk-config>)")
    build.add_argument("--dry-run", action="store_true", help="임베딩 없이 청크 수·길이 분포만 출력")
    build.add_argument("--rebuild", action="store_true", help="기존 컬렉션을 지우고 처음부터 적재")

    verify = sub.add_parser("verify", help="적재 결과 검증")
    verify.add_argument("--collection", required=True)
    verify.add_argument("--no-search", action="store_true", help="샘플 검색(질문 임베딩 API 호출) 생략")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        prepare.run(quality_name=args.quality, dedup_name=args.dedup)
    elif args.command == "build":
        config = chunk.get_chunk_config(args.chunk_config)
        try:
            stats = embed.build_collection(config, args.collection or f"docs__{args.chunk_config}",
                                           rebuild=args.rebuild, dry_run=args.dry_run)
        except RuntimeError as exc:  # API 키 미설정 등 설정 문제는 안내 문구만 보여 준다
            raise SystemExit(str(exc)) from exc
        print(stats.summary())
    elif args.command == "verify":
        try:
            report = embed.verify_collection(args.collection, queries=() if args.no_search else embed.SAMPLE_QUERIES)
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        print(report.summary())


if __name__ == "__main__":
    main()
