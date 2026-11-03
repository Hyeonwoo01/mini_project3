"""인덱싱 계층 실행 진입점.

    python -m indexer prepare                         # 파싱·정제·중복 처리 → processed_documents
    python -m indexer build --chunk-config baseline   # 청킹·임베딩 → Chroma 컬렉션
    python -m indexer verify --collection docs        # 문서 수·청크 수·빈 청크·샘플 검색 확인

prepare 와 build 를 분리하여, 중복 처리 결과 하나로 여러 청킹 설정을 비교할 수 있게 한다.
"""

import argparse

from indexer import chunk, dedup, embed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m indexer", description="② 인덱싱 계층")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("prepare", help="파싱·정제·중복 처리")

    build = sub.add_parser("build", help="청킹·임베딩 후 Chroma 적재")
    build.add_argument("--chunk-config", required=True, help="indexer/chunk.py 의 CHUNK_CONFIGS 키")
    build.add_argument("--collection", help="적재할 컬렉션 이름 (기본: docs__<chunk-config>)")

    verify = sub.add_parser("verify", help="적재 결과 검증")
    verify.add_argument("--collection", required=True)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        dedup.run()
    elif args.command == "build":
        config = chunk.get_chunk_config(args.chunk_config)
        embed.build_collection(config, collection=args.collection or f"docs__{args.chunk_config}")
    elif args.command == "verify":
        embed.verify_collection(args.collection)


if __name__ == "__main__":
    main()
