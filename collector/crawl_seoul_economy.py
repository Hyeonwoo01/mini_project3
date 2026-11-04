"""서울시 경제분야 누리집(https://news.seoul.go.kr/economy) 소상공인 지원 소식 수집.

서울시가 직접 안내하는 소상공인 지원사업·행사·모집 공고로, 서울 범위 질문의 1차 근거다.
서울시 보도자료 목록(www.seoul.go.kr/news)은 robots.txt 가 금지해 이 누리집을 쓴다.

    python -m collector seoul-economy --dry-run     # 받을 목록만 확인
    python -m collector seoul-economy               # 수집

페이지 구조 (2026-11-03, collector probe 로 받은 표본으로 확인)
- 목록 <카테고리>?paged=n: div.child_policyDL 마다 h3.tit a (제목, 상세 주소), span.time '등록일 : 2026-11-02'.
  마지막 쪽은 div.paging a.page-numbers 의 가장 큰 숫자
- 상세 archives/<글 번호>: 본문 div#post_content. 이 div 가 닫히지 않아 뒤따르는 태그 목록·공공누리 표시·
  만족도 조사·댓글 안내까지 그 안에 들어가므로, 태그 목록(div.tagDiv) 등 STOP 요소 앞까지의 글자만 쓴다.
  담당 부서 dd.dept, 수정일 dd.date
- 이용 조건: 공공누리 제4유형(출처표시·상업적 이용금지·변경금지). 비상업 교육 목적으로 출처를 표시해 쓴다

본문은 사이트 구조에 따라 잘라야 해서 수집기에서 추출해 raw_text 로 저장하고, 원본 HTML 도 함께 보관한다.
포스터 이미지만 있는 글은 본문이 짧아 인덱서 품질 필터에서 제외된다 (이미지 글자는 읽지 않는다).
"""

from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import requests
from bs4 import BeautifulSoup, Comment, NavigableString

from collector.fetch import Fetcher, RobotsDisallowed, describe_error
from collector.storage import make_doc_id, save_raw
from common.docstore import open_db, upsert_documents
from common.schema import CollectMethod, DocType, DocumentRecord

DEFAULT_LIST_URL = ("https://news.seoul.go.kr/economy/archives/category/nationaleconomy-news_c1/"
                    "street_biz_nationaleconomy-news-n1")
DOC_ID_PREFIX = "seoul"
DEFAULT_START = "2021-10-16"
SOURCE = "서울특별시"
LICENSE = "공공누리 제4유형(출처표시·상업적 이용금지·변경금지)"
BODY_STOP = "div.tagDiv, #sns_elem, #content_manager_info"


@dataclass(frozen=True)
class ListItem:
    url: str                   # 쿼리를 뗀 상세 주소
    title: str
    published_at: str | None   # YYYY-MM-DD (등록일)


@dataclass(frozen=True)
class Detail:
    text: str
    dept: str
    modified_at: str | None


@dataclass
class CrawlStats:
    pages: int = 0
    listed: int = 0
    out_of_period: int = 0
    already: int = 0
    saved: int = 0
    planned: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"목록 {self.pages}쪽 · 글 {self.listed}건",
                 f"  기간 밖 {self.out_of_period} · 이미 수집 {self.already} · 새로 저장 {self.saved}"
                 + (f" · 받을 대상 {len(self.planned)} (--dry-run)" if self.planned else ""),
                 f"  실패 {len(self.failed)}"]
        lines += [f"    실패: {item}" for item in self.failed[:10]]
        return "\n".join(lines)


def _text(tag) -> str:
    return " ".join(tag.get_text(" ", strip=True).split()) if tag else ""


def _canonical(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def parse_list(html: str) -> tuple[list[ListItem], int]:
    soup = BeautifulSoup(html, "lxml")
    items = []
    for block in soup.select("div.child_policyDL"):
        link = block.select_one("h3.tit a[href]")
        if link is None:
            continue
        time = _text(block.select_one("span.time"))
        date = time.split(":", 1)[1].split()[0] if "등록일" in time and ":" in time else ""
        items.append(ListItem(url=_canonical(link["href"]), title=link.get("title") or _text(link),
                              published_at=date if len(date) == 10 else None))
    numbers = [int(a.get_text(strip=True)) for a in soup.select("div.paging .page-numbers")
               if a.get_text(strip=True).isdigit()]
    return items, max(numbers, default=1)


def parse_detail(html: str) -> Detail:
    soup = BeautifulSoup(html, "lxml")
    body = soup.select_one("#post_content")
    if body is None:
        raise ValueError("본문 영역(#post_content)을 찾지 못함")
    stop = body.select_one(BODY_STOP)
    lines = []
    for node in body.descendants:
        if node is stop:
            break
        if isinstance(node, NavigableString) and not isinstance(node, Comment) \
                and node.parent.name not in ("script", "style", "noscript"):
            line = " ".join(node.split())
            if line:
                lines.append(line)
    dept = " ".join(span.get_text(strip=True) for span in soup.select("dd.dept span"))
    modified = _text(soup.select_one("dd.date")) or None
    return Detail(text="\n".join(lines), dept=dept, modified_at=modified)


def _already_saved(conn, doc_id: str) -> bool:
    # 본문이 비어 있는 글(포스터만 있는 공지)도 저장된 것으로 본다. 본문 여부로 판정하면 매번 다시 받는다
    return conn.execute("SELECT 1 FROM documents WHERE doc_id = ?", (doc_id,)).fetchone() is not None


def collect_item(fetcher: Fetcher, item: ListItem, list_url: str) -> DocumentRecord:
    if item.published_at is None:
        raise ValueError("목록에서 등록일을 읽지 못함")
    page = fetcher.get(item.url)
    if page.status != 200:
        raise requests.HTTPError(f"상세 페이지 {page.status}")
    detail = parse_detail(page.text)
    doc_id = make_doc_id(DOC_ID_PREFIX, item.url, item.published_at)
    raw_path = save_raw(DocType.ANNOUNCEMENT, doc_id, page.content, "html")
    return DocumentRecord(
        doc_id=doc_id, doc_type=DocType.ANNOUNCEMENT, source=SOURCE, title=item.title, url=item.url,
        published_at=item.published_at, collected_at=page.collected_at, collect_method=CollectMethod.CRAWL,
        raw_path=raw_path, raw_text=detail.text,
        extra={"portal": "서울시 경제분야 누리집", "dept": detail.dept, "modified_at": detail.modified_at,
               "list_url": list_url, "license": LICENSE},
    )


def run(list_urls: list[str] | None = None, start: str = DEFAULT_START, max_pages: int | None = None,
        dry_run: bool = False, fetcher: Fetcher | None = None, db_path: Path | None = None) -> CrawlStats:
    fetcher = fetcher or Fetcher()
    stats = CrawlStats()
    with open_db(db_path) as conn:
        for list_url in list_urls or [DEFAULT_LIST_URL]:
            page, last_page = 1, 1
            while page <= last_page and (max_pages is None or page <= max_pages):
                result = fetcher.get(list_url, params={"paged": page} if page > 1 else None)
                if result.status != 200:
                    print(f"{page}쪽 목록 응답 {result.status} — 중단")
                    break
                items, last_page = parse_list(result.text)
                stats.pages += 1
                for item in items:
                    stats.listed += 1
                    if item.published_at and item.published_at < start:
                        stats.out_of_period += 1
                        continue
                    label = f"{item.published_at} {item.title[:50]}"
                    if item.published_at and _already_saved(conn, make_doc_id(DOC_ID_PREFIX, item.url,
                                                                               item.published_at)):
                        stats.already += 1
                        continue
                    if dry_run:
                        stats.planned.append(label)
                        print(f"  대상 {len(stats.planned)}: {label}", flush=True)
                        continue
                    try:
                        record = collect_item(fetcher, item, list_url)
                    except (requests.RequestException, RobotsDisallowed, ValueError) as exc:
                        message = describe_error(exc) if isinstance(exc, requests.RequestException) else str(exc)
                        stats.failed.append(f"{label} — {message}")
                        continue
                    upsert_documents(conn, [record])
                    conn.commit()
                    stats.saved += 1
                    print(f"  저장 {stats.saved}: {label} ({len(record.raw_text or ''):,}자)", flush=True)
                print(f"{page}/{last_page}쪽 완료 · 누적 저장 {stats.saved}", flush=True)
                page += 1
    print(stats.summary())
    return stats
