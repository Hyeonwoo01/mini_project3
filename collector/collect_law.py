"""핵심 법령 현행 조문 수집 (국가법령정보센터 Open API).

    python -m collector law --dry-run          # 수집 대상 법령만 보여 준다 (요청 없음)
    python -m collector law                    # 수집 (.env 의 LAW_OC 필요)
    python -m collector law --only 여신전문금융업법

정책·제도 질문의 최종 근거가 되는 법령 5개만 수집한다 (README '법령 후보'). 국회 법안·검토보고서가 아니라
현행 조문만 넣는다. 분석가에게 필요한 것은 현재 적용 기준이기 때문이다.

- 인증키 OC 는 open.law.go.kr 에서 OPEN API 를 신청할 때 직접 정한다 (자동 승인). .env 의 LAW_OC 에 넣고, 주소에 들어가므로
  화면 출력·오류 메시지·저장 원본에서는 가린다
- 조문 하나를 문서 하나로 저장한다 (doc_type law). 청킹 설정(c500)은 그대로 쓰고, 조문이 길면 그 안에서 나뉜다.
  제목은 '법령명 제n조(조문 제목)' — 청크 제목에 들어가 어느 법 몇 조인지 알 수 있다
- 발행일(published_at)에는 법령 시행일을 넣는다. 개정되면 법령일련번호(MST)가 바뀌므로, 다시 수집하면 그 법령의
  예전 조문을 지우고 현행 조문으로 바꾼다. 목록에서 뺀 법령의 조문도 지운다
- 여신전문금융업법처럼 대부분이 카드사 인허가 조문인 법은 관련 조문만 고른다 (LawTarget.articles)
- 요청은 collector/fetch.py 를 통해서만 보낸다 (robots.txt·딜레이·User-Agent)

응답 구조 (DRF XML)
- 검색 lawSearch.do?target=law&query=법령명: <law> 마다 법령일련번호(MST)·법령명한글·법령구분명·현행연혁코드·시행일자
- 본문 lawService.do?target=law&MST=: 기본정보(법령명_한글·시행일자·소관부처) + 조문/조문단위 마다 조문번호·조문가지번호
  ('제18조의3' 의 3)·조문여부('조문' 또는 장·절 제목인 '전문')·조문제목·조문내용, 그 아래 항 → 호 → 목
"""

import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote
from xml.etree import ElementTree

import requests

from collector.fetch import Fetcher, RobotsDisallowed, describe_error
from collector.storage import make_doc_id, save_raw
from common.config import get_settings, require_env
from common.docstore import open_db, upsert_documents
from common.schema import CollectMethod, DocType, DocumentRecord

API_BASE = "https://www.law.go.kr/DRF"
OC_ENV = "LAW_OC"
SOURCE = "국가법령정보센터"
DOC_ID_PREFIX = "law"


@dataclass(frozen=True)
class LawTarget:
    name: str                              # 법령명 (검색 결과의 법령명한글과 정확히 같아야 한다)
    reason: str
    articles: tuple[str, ...] | None = None   # 고를 조문 ('18의3' = 제18조의3). None 이면 전체


CURATED_LAWS: tuple[LawTarget, ...] = (
    LawTarget("지역상권 상생 및 활성화에 관한 법률", "상권활성화구역·자율상권구역 지정 기준과 지원 근거"),
    LawTarget("전통시장 및 상점가 육성을 위한 특별법", "전통시장·상점가 정의, 온누리상품권 발행·가맹점 근거"),
    LawTarget("지역사랑상품권 이용 활성화에 관한 법률", "지역사랑상품권 발행·가맹점 등록·국가 재정 지원 근거"),
    LawTarget("상가건물 임대차보호법", "상가 임대료 증액 상한·계약갱신요구권 (임대료·젠트리피케이션)"),
    LawTarget("여신전문금융업법", "영세·중소가맹점 우대수수료율의 법적 근거", articles=("18의3",)),
)


class LawApiError(Exception):
    """API 가 법령 XML 대신 오류 안내를 돌려준 경우 (인증값 미등록·호출 IP 미등록 등)."""


@dataclass(frozen=True)
class LawInfo:
    mst: str
    name: str
    kind: str
    enforced_at: str | None      # YYYY-MM-DD
    ministry: str


@dataclass(frozen=True)
class Article:
    key: str           # '18' 또는 '18의3'
    title: str
    text: str


def _text(element: ElementTree.Element | None, tag: str) -> str:
    found = element.find(tag) if element is not None else None
    return " ".join((found.text or "").split()) if found is not None else ""


def _date(raw: str) -> str | None:
    digits = re.sub(r"\D", "", raw)
    return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}" if len(digits) == 8 else None


def _parse_xml(content: bytes, what: str) -> ElementTree.Element:
    try:
        return ElementTree.fromstring(content)
    except ElementTree.ParseError as exc:
        snippet = " ".join(content[:300].decode("utf-8", "replace").split())
        raise LawApiError(f"{what} 응답이 XML 이 아닙니다 (인증값·등록 IP 확인): {snippet}") from exc


def parse_search(content: bytes, name: str) -> LawInfo | None:
    """검색 결과에서 이름이 정확히 같은 법률을 고른다 (시행령·시행규칙과 이름이 비슷한 다른 법은 뺀다)."""
    root = _parse_xml(content, "검색")
    if root.tag != "LawSearch":     # 인증 실패 등은 다른 형식의 안내문으로 온다
        message = " ".join("".join(root.itertext()).split())[:200]
        raise LawApiError(f"검색 응답이 결과 목록이 아닙니다 (<{root.tag}>): {message}")
    matches = [law for law in root.iter("law") if _text(law, "법령명한글") == name and _text(law, "법령일련번호")]
    # 연혁(과거 법령)이 함께 오면 현행을 고른다. 현행연혁코드가 없는 응답이면 첫 결과
    current = [law for law in matches if _text(law, "현행연혁코드") in ("", "현행")]
    if not current:
        return None
    law = current[0]
    return LawInfo(mst=_text(law, "법령일련번호"), name=name, kind=_text(law, "법령구분명"),
                   enforced_at=_date(_text(law, "시행일자")), ministry=_text(law, "소관부처명"))


def _article_text(unit: ElementTree.Element) -> str:
    lines = [_text(unit, "조문내용")]
    for clause in unit.findall("항"):
        lines.append(_text(clause, "항내용"))
        for item in clause.findall("호"):
            lines.append(_text(item, "호내용"))
            lines.extend(_text(sub, "목내용") for sub in item.findall("목"))
    for item in unit.findall("호"):           # 항 없이 호가 바로 오는 조문
        lines.append(_text(item, "호내용"))
        lines.extend(_text(sub, "목내용") for sub in item.findall("목"))
    return "\n".join(line for line in lines if line)


_DELETED = re.compile(r"^제\d+조(의\d+)?\s*삭제")


def parse_law(content: bytes) -> tuple[dict[str, str], list[Article]]:
    """(기본정보, 조문 목록). 장·절 제목('전문')과 삭제된 조문은 뺀다."""
    root = _parse_xml(content, "본문")
    info_el = root.find("기본정보")
    if info_el is None:
        raise LawApiError(f"본문 응답에 기본정보가 없습니다: <{root.tag}>")
    info = {"name": _text(info_el, "법령명_한글"), "enforced_at": _date(_text(info_el, "시행일자")) or "",
            "ministry": _text(info_el, "소관부처")}
    articles = []
    for unit in root.iter("조문단위"):
        if _text(unit, "조문여부") != "조문":
            continue
        number, branch = _text(unit, "조문번호"), _text(unit, "조문가지번호")
        text = _article_text(unit)
        if not number or not text or _DELETED.match(text):
            continue
        articles.append(Article(key=f"{number}의{branch}" if branch else number, title=_text(unit, "조문제목"),
                                text=text))
    return info, articles


def article_label(key: str) -> str:
    number, _, branch = key.partition("의")
    return f"제{number}조의{branch}" if branch else f"제{number}조"


def article_url(law_name: str, key: str) -> str:
    return f"https://www.law.go.kr/법령/{quote(law_name)}/{quote(article_label(key))}"


def to_records(law: LawInfo, info: dict[str, str], articles: Sequence[Article], raw_path: str,
               collected_at: str, target: LawTarget) -> list[DocumentRecord]:
    wanted = set(target.articles) if target.articles else None
    enforced_at = info["enforced_at"] or law.enforced_at
    ministry = info["ministry"] or law.ministry
    records = []
    for article in articles:
        if wanted is not None and article.key not in wanted:
            continue
        label = article_label(article.key)
        url = article_url(law.name, article.key)
        records.append(DocumentRecord(
            doc_id=make_doc_id(DOC_ID_PREFIX, url, enforced_at), doc_type=DocType.LAW,
            source=f"{SOURCE} ({ministry})" if ministry else SOURCE,
            title=f"{law.name} {label}" + (f"({article.title})" if article.title else ""),
            url=url, published_at=enforced_at, collected_at=collected_at, collect_method=CollectMethod.API,
            raw_path=raw_path, raw_text=article.text,
            extra={"law_name": law.name, "mst": law.mst, "article": article.key, "law_kind": law.kind,
                   "ministry": ministry, "reason": target.reason},
        ))
    return records


def _redact(text: str, oc: str) -> str:
    return text.replace(oc, "***") if oc else text


def _get(fetcher: Fetcher, url: str, params: dict[str, str]) -> bytes:
    page = fetcher.get(url, params=params)
    if page.status != 200:
        raise requests.HTTPError(f"{url.rsplit('/', 1)[-1]} {page.status}")
    return page.content


def fetch_law(fetcher: Fetcher, oc: str, target: LawTarget) -> tuple[LawInfo, bytes, str]:
    search = _get(fetcher, f"{API_BASE}/lawSearch.do",
                  {"OC": oc, "target": "law", "type": "XML", "query": target.name, "display": "20"})
    law = parse_search(search, target.name)
    if law is None:
        raise LawApiError(f"검색 결과에 '{target.name}' 이 없습니다")
    page = fetcher.get(f"{API_BASE}/lawService.do", params={"OC": oc, "target": "law", "MST": law.mst, "type": "XML"})
    if page.status != 200:
        raise requests.HTTPError(f"lawService.do {page.status}")
    return law, page.content, page.collected_at


def replace_law_documents(conn, law_name: str, records: Sequence[DocumentRecord]) -> int:
    """같은 법령의 예전 조문(개정 전 MST·이번에 고르지 않은 조문)을 지우고 새 조문을 넣는다. 지운 수를 반환."""
    keep = {r.doc_id for r in records}
    stale = [doc_id for (doc_id,) in conn.execute(
        "SELECT doc_id FROM documents WHERE doc_type = ? AND json_extract(extra, '$.law_name') = ?",
        (DocType.LAW.value, law_name)) if doc_id not in keep]
    _delete(conn, stale)
    upsert_documents(conn, list(records))
    return len(stale)


def prune_unlisted(conn, targets: Sequence[LawTarget]) -> int:
    """목록에서 뺀 법령의 조문을 지운다 (처리 결과도 함께 — 다음 indexer prepare 가 다시 만든다)."""
    names = {t.name for t in targets}
    stale = [doc_id for doc_id, name in conn.execute(
        "SELECT doc_id, json_extract(extra, '$.law_name') FROM documents WHERE doc_type = ?", (DocType.LAW.value,))
        if name not in names]
    _delete(conn, stale)
    return len(stale)


def _delete(conn, doc_ids: Sequence[str]) -> None:
    if not doc_ids:
        return
    marks = ",".join("?" * len(doc_ids))
    conn.execute(f"DELETE FROM processed_documents WHERE doc_id IN ({marks}) OR canonical_doc_id IN ({marks})",
                 [*doc_ids, *doc_ids])
    conn.execute(f"DELETE FROM documents WHERE doc_id IN ({marks})", list(doc_ids))


@dataclass
class CollectStats:
    saved: dict[str, int] = field(default_factory=dict)      # 법령명 → 조문 수
    replaced: int = 0
    pruned: int = 0
    failed: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"법령 {len(self.saved)}개 · 조문 {sum(self.saved.values())}건 저장"
                 f" · 예전 조문 교체 {self.replaced}건 · 목록에서 빠져 삭제 {self.pruned}건 · 실패 {len(self.failed)}"]
        lines += [f"  {name}: {count}조" for name, count in self.saved.items()]
        lines += [f"  실패: {item}" for item in self.failed]
        return "\n".join(lines)


def write_review(records: Sequence[DocumentRecord], path: Path) -> None:
    """수집한 조문 확인용 (평가 문항 작성에 쓴다). data/raw 아래라 git 에 올리지 않는다."""
    parts = [f"## {r.title}\n- {r.source} · 시행 {r.published_at} · {r.doc_id}\n- {r.url}\n\n{r.raw_text}\n"
             for r in records]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def run(dry_run: bool = False, targets: Sequence[LawTarget] = CURATED_LAWS, fetcher: Fetcher | None = None,
        db_path: Path | None = None, review_path: Path | None = None, prune: bool = False,
        oc: str | None = None) -> CollectStats:
    """prune: 목록에 없는 법령의 조문을 지운다 (CLI 는 전체 목록일 때만 켠다)."""
    stats = CollectStats()
    if dry_run:
        for target in targets:
            scope = f"조문 {', '.join(map(article_label, target.articles))}" if target.articles else "전체 조문"
            print(f"  {target.name} ({scope})\n      {target.reason}")
        print(f"수집 대상 {len(targets)}개 법령 (--dry-run, 요청하지 않음)")
        return stats
    if oc is None:
        require_env(OC_ENV)
        oc = os.environ[OC_ENV].strip()
    fetcher = fetcher or Fetcher()
    saved: list[DocumentRecord] = []
    with open_db(db_path) as conn:
        for target in targets:
            try:
                law, content, collected_at = fetch_law(fetcher, oc, target)
                info, articles = parse_law(content)
                safe = content.replace(oc.encode(), b"***")
                raw_path = save_raw(DocType.LAW, f"{DOC_ID_PREFIX}_{law.mst}", safe, "xml")
                records = to_records(law, info, articles, raw_path, collected_at, target)
                if not records:
                    raise LawApiError(f"고른 조문이 없습니다 (본문 조문 {len(articles)}개)")
            except (requests.RequestException, RobotsDisallowed, LawApiError) as exc:
                message = _redact(describe_error(exc) if not isinstance(exc, LawApiError) else str(exc), oc)
                stats.failed.append(f"{target.name} — {message}")
                print(f"  실패 {target.name} — {message}", flush=True)
                continue
            stats.replaced += replace_law_documents(conn, law.name, records)
            stats.saved[law.name] = len(records)
            saved.extend(records)
            print(f"  저장 {law.name} (시행 {records[0].published_at}, MST {law.mst}) {len(records)}조", flush=True)
        if prune:
            stats.pruned = prune_unlisted(conn, targets)
    review_path = review_path or get_settings().raw_dir / "_probe" / "law" / "collected.md"
    write_review(saved, review_path)
    print(stats.summary())
    print(f"수집 조문 확인: {review_path}")
    return stats
