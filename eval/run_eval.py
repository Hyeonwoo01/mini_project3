"""검색·답변 품질 측정.

    python -m eval.run_eval --tag baseline --collection docs__baseline
    python -m eval.run_eval --tag baseline --collection docs__baseline --with-answers

평가셋 형식 (eval/eval_set.jsonl, 한 줄에 한 문항):
    {"id": "f001", "q": "...", "type": "fact", "gold_doc_ids": ["kr_20260115_1a2b3c4d"], "answer_key": "..."}
    {"id": "u001", "q": "...", "type": "unanswerable", "gold_doc_ids": [], "answer_key": "답변 거절이 정답"}

id 는 실험 간 같은 문항을 대조하는 키이므로 문항을 수정·재정렬해도 바꾸지 않는다.

결과는 eval/results/<YYYYMMDD-HHMM>_<tag>/ 에 저장한다 (가이드 4-9).
    retrieval.jsonl  문항별 검색 결과·순위·점수
    summary.json     설정(컬렉션·모델·k)과 Hit@1·Hit@3·MRR, 유형별 집계
    answers.csv      --with-answers 시 생성. 직접 채점용 (grade: correct|partial|wrong, citation_ok)
"""

import argparse
import json
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path

QUESTION_TYPES = ("fact", "comparison", "temporal", "synthesis", "unanswerable")


@dataclass(frozen=True)
class EvalItem:
    id: str
    q: str
    type: str
    gold_doc_ids: tuple[str, ...]
    answer_key: str


def load_eval_set(path: Path) -> list[EvalItem]:
    """평가셋을 읽고 형식을 검증한다. 오류가 있으면 줄 번호와 함께 실패한다."""
    items: list[EvalItem] = []
    seen_ids: set[str] = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            gold = raw["gold_doc_ids"]
            if not isinstance(gold, list) or not all(isinstance(g, str) for g in gold):
                raise TypeError("gold_doc_ids 는 문자열 리스트여야 합니다")
            item = EvalItem(
                id=str(raw["id"]),
                q=str(raw["q"]),
                type=str(raw["type"]),
                gold_doc_ids=tuple(raw["gold_doc_ids"]),
                answer_key=str(raw["answer_key"]),
            )
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"{path}:{line_no}: 형식 오류 ({exc})") from exc
        if item.type not in QUESTION_TYPES:
            raise ValueError(f"{path}:{line_no}: 알 수 없는 type {item.type!r}")
        if (item.type == "unanswerable") != (not item.gold_doc_ids):
            raise ValueError(f"{path}:{line_no}: unanswerable 문항만 gold_doc_ids 가 비어 있어야 합니다")
        if item.id in seen_ids:
            raise ValueError(f"{path}:{line_no}: 중복 id {item.id!r}")
        seen_ids.add(item.id)
        items.append(item)
    return items


def _require_gold(gold_doc_ids: Collection[str]) -> None:
    # 답변 불가 문항을 검색 지표에 섞으면 오답으로 집계되어 수치가 왜곡된다.
    if not gold_doc_ids:
        raise ValueError("gold_doc_ids 가 빈 문항(답변 불가)은 검색 지표에서 제외해야 합니다")


def hit_at_k(ranked_doc_ids: Sequence[str], gold_doc_ids: Collection[str], k: int) -> bool:
    """상위 k개 결과에 정답 문서가 하나라도 있으면 True.

    ranked_doc_ids 는 검색 순위대로 나열한 각 결과의 doc_id 다. 청크 단위 결과를 그대로 넘기면
    같은 문서가 여러 번 나올 수 있으며, 이 경우 k 는 청크 개수 기준이 된다.
    """
    if k <= 0:
        raise ValueError(f"k 는 1 이상이어야 합니다: {k}")
    _require_gold(gold_doc_ids)
    return any(doc_id in gold_doc_ids for doc_id in ranked_doc_ids[:k])


def reciprocal_rank(ranked_doc_ids: Sequence[str], gold_doc_ids: Collection[str]) -> float:
    """첫 정답 문서 순위의 역수. 정답이 없으면 0.0. MRR 은 이 값의 문항 평균이다."""
    _require_gold(gold_doc_ids)
    for rank, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id in gold_doc_ids:
            return 1.0 / rank
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m eval.run_eval", description="평가 하네스")
    parser.add_argument("--tag", required=True, help="실험 태그 (결과 디렉터리 이름에 사용)")
    parser.add_argument("--collection", help="평가할 Chroma 컬렉션 (기본: CHROMA_COLLECTION)")
    parser.add_argument("--k", type=int, default=5, help="검색 결과 수")
    parser.add_argument("--with-answers", action="store_true", help="답변 생성 및 채점용 CSV 출력")
    parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
