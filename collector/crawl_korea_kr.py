"""대한민국 정책브리핑(https://www.korea.kr/) 보도자료 수집.

부처 보도자료 원문은 지원사업 요건·규모 등 1차 근거에 해당한다 (가이드 3-3). 정책브리핑은 모든 부처의
보도자료를 한곳에서 검색어·기간으로 찾을 수 있어 중소벤처기업부·금융위원회 등을 따로 크롤링하지 않는다.

    python -m collector korea-kr --query 소상공인 --ministry 중소벤처기업부 --count-only   # 검색 결과 수만
    python -m collector korea-kr --query 소상공인 --ministry 중소벤처기업부 --max-pages 1  # 시험 수집
    python -m collector korea-kr --query 소상공인 --ministry 중소벤처기업부                # 전체

페이지 구조 (2026-11-03, collector probe 로 받은 표본으로 확인)
- 목록 pressReleaseList.do?pageIndex&startDate&endDate&srchWord: div.list_type li > a 마다
  strong(제목), span.source > span (게시일, 부처명). 게시일은 일반 목록이 2026.11.03, 검색 결과가 2026-11-03
  형식이다. 마지막 쪽은 div.paging a.last 의 pageLink(n)
- 상세 pressReleaseView.do?newsId=: 본문은 HTML 에 없고 문서뷰어 iframe 으로 따로 불러온다.
  대신 div.filedown 에 원문 첨부가 있어 PDF 를, 없으면 HWPX 를 받아 본문으로 쓴다 (텍스트 추출은 indexer).
  옛 HWP(바이너리) 만 있는 보도자료는 받지 않고 '원문 없음'으로 보고한다 (첨부 확장자를 함께 출력)
- 이용 조건: 상세 화면 하단 '텍스트에 한하여 공공누리 제1유형(출처표시)'. 출처(부처·URL)를 메타데이터로 보존한다

부처 거르기는 목록의 부처명으로 한다. 목록 화면의 부처 선택 파라미터(repCode)는 동작을 확인하지 못했다.
검색은 본문·첨부까지 찾아서 지나가듯 언급한 보도자료도 나온다 ('카드수수료' 검색에 가상자산·중동 상황 점검
보도자료). 그래서 --title-keyword 로 제목에 주제어가 있는 것만 받는다. --dry-run 으로 받을 목록을 먼저 본다.
이미 원문을 받은 문서는 다시 받지 않으므로, 중단된 수집은 같은 명령으로 이어서 실행하면 된다.
요청은 collector/fetch.py 를 통해서만 보낸다 (robots.txt·딜레이·User-Agent).
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import parse_qs, urljoin, urlsplit

import requests
from bs4 import BeautifulSoup

from collector.fetch import Fetcher, RobotsDisallowed, describe_error
from collector.storage import make_doc_id, save_raw
from common.config import PROJECT_ROOT
from common.docstore import open_db, upsert_documents
from common.schema import CollectMethod, DocType, DocumentRecord

BASE_URL = "https://www.korea.kr"
LIST_URL = f"{BASE_URL}/briefing/pressReleaseList.do"
VIEW_URL = f"{BASE_URL}/briefing/pressReleaseView.do"
DOC_ID_PREFIX = "kr"
# 제공 뉴스와 같은 기간 (2021-10-16 ~ 2026-10-16)
DEFAULT_START, DEFAULT_END = "2021-10-16", "2026-10-16"
LICENSE = "공공누리 제1유형(출처표시), 텍스트에 한함"

_PAGE_LINK = re.compile(r"pageLink\((\d+)\)")
_TOTAL = re.compile(r"총\s*([\d,]+)\s*건")
_DATE = re.compile(r"(\d{4})[.-](\d{2})[.-](\d{2})")
_TITLE_TAG = re.compile(r"^\s*\[보도자료\]\s*")


@dataclass(frozen=True)
class ListItem:
    news_id: str
    title: str
    published_at: str | None   # YYYY-MM-DD
    ministry: str

    @property
    def url(self) -> str:
        """목록의 검색 조건 파라미터를 뺀 정규 주소 (상세 페이지 canonical 과 같다)."""
        return f"{VIEW_URL}?newsId={self.news_id}"


@dataclass(frozen=True)
class ListPage:
    items: list[ListItem]
    last_page: int
    total: int | None          # 검색어가 있을 때만 화면에 '검색결과 총 n건' 이 나온다


@dataclass(frozen=True)
class Attachment:
    name: str
    url: str

    @property
    def extension(self) -> str:
        return Path(self.name).suffix.lstrip(".").lower()


# 받을 원문 형식과 우선순위, 파일 시작 바이트. HWPX 는 zip 묶음이다
ORIGINAL_FORMATS: dict[str, bytes] = {"pdf": b"%PDF", "hwpx": b"PK"}


class NoOriginal(Exception):
    """본문을 추출할 수 있는 원문 첨부(PDF·HWPX)가 없다."""


def pick_original(attachments: list[Attachment]) -> Attachment:
    for extension in ORIGINAL_FORMATS:
        found = next((a for a in attachments if a.extension == extension), None)
        if found:
            return found
    kinds = ", ".join(a.extension or "?" for a in attachments) or "첨부 없음"
    raise NoOriginal(f"첨부: {kinds}")


@dataclass
class CrawlStats:
    pages: int = 0
    listed: int = 0
    skipped_ministry: int = 0
    skipped_title: int = 0
    planned: list[str] = field(default_factory=list)   # --dry-run 에서 받을 대상
    already: int = 0
    saved: int = 0
    no_original: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"목록 {self.pages}쪽 · 보도자료 {self.listed}건",
                 f"  부처 조건 밖 {self.skipped_ministry} · 제목 조건 밖 {self.skipped_title} · "
                 f"이미 수집 {self.already} · 새로 저장 {self.saved}"
                 + (f" · 받을 대상 {len(self.planned)} (--dry-run)" if self.planned else ""),
                 f"  원문 첨부(PDF·HWPX) 없음 {len(self.no_original)} · 실패 {len(self.failed)}"]
        lines += [f"    원문 없음: {item}" for item in self.no_original[:40]]
        lines += [f"    실패: {item}" for item in self.failed[:10]]
        return "\n".join(lines)


def _text(tag) -> str:
    return " ".join(tag.get_text(" ", strip=True).split()) if tag else ""


def _iso_date(value: str) -> str | None:
    match = _DATE.search(value)
    return "-".join(match.groups()) if match else None


def parse_list(html: str) -> ListPage:
    soup = BeautifulSoup(html, "lxml")
    items = []
    for a in soup.select("div.list_type li > a[href*='pressReleaseView.do']"):
        news_id = parse_qs(urlsplit(a["href"]).query).get("newsId", [""])[0]
        title = _TITLE_TAG.sub("", _text(a.select_one("strong")))
        source = [_text(span) for span in a.select("span.source > span")]
        if not news_id or not title:
            continue
        items.append(ListItem(news_id=news_id, title=title,
                              published_at=_iso_date(source[0]) if source else None,
                              ministry=source[1] if len(source) > 1 else ""))
    pages = [int(n) for a in soup.select("div.paging a[onclick]") for n in _PAGE_LINK.findall(a["onclick"])]
    total = _TOTAL.search(_text(soup.select_one("div.result")))
    return ListPage(items=items, last_page=max(pages, default=1),
                    total=int(total.group(1).replace(",", "")) if total else None)


def parse_attachments(html: str, base_url: str = BASE_URL) -> list[Attachment]:
    soup = BeautifulSoup(html, "lxml")
    attachments = []
    for a in soup.select("div.filedown dd p > span:first-child > a[href]"):
        attachments.append(Attachment(name=_text(a), url=urljoin(base_url, a["href"])))
    return attachments


def _list_params(query: str, page: int, start: str, end: str) -> dict:
    return {"pageIndex": page, "startDate": start, "endDate": end, "srchWord": query}


def _saved_raw_path(conn, doc_id: str) -> str | None:
    row = conn.execute("SELECT raw_path FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
    if row and row["raw_path"] and (PROJECT_ROOT / row["raw_path"]).is_file():
        return row["raw_path"]
    return None


def collect_item(fetcher: Fetcher, item: ListItem, query: str) -> DocumentRecord:
    """상세 페이지에서 원문 첨부(PDF, 없으면 HWPX)를 받아 저장하고 문서 레코드를 만든다. 없으면 NoOriginal."""
    if item.published_at is None:
        # doc_id 와 시점 질문이 게시일에 의존하므로 날짜 없이 저장하지 않는다 (목록 구조가 바뀐 경우)
        raise ValueError("목록에서 게시일을 읽지 못함")
    detail = fetcher.get(item.url)
    if detail.status != 200:
        raise requests.HTTPError(f"상세 페이지 {detail.status}")
    original = pick_original(parse_attachments(detail.text))
    file = fetcher.get(original.url)
    if file.status != 200 or not file.content.startswith(ORIGINAL_FORMATS[original.extension]):
        raise ValueError(f"{original.extension.upper()} 가 아닌 응답 ({file.status}, {file.content_type})")
    doc_id = make_doc_id(DOC_ID_PREFIX, item.url, item.published_at)
    raw_path = save_raw(DocType.PRESS_RELEASE, doc_id, file.content, original.extension)
    return DocumentRecord(
        doc_id=doc_id, doc_type=DocType.PRESS_RELEASE, source=item.ministry or "정책브리핑", title=item.title,
        url=item.url, published_at=item.published_at, collected_at=file.collected_at,
        collect_method=CollectMethod.CRAWL, keyword=query, raw_path=raw_path,
        extra={"portal": "정책브리핑", "news_id": item.news_id, "attachment_name": original.name,
               "attachment_url": original.url, "license": LICENSE},
    )


def run(queries: list[str], max_pages: int | None = None, ministries: list[str] | None = None,
        title_keywords: list[str] | None = None, start: str = DEFAULT_START, end: str = DEFAULT_END,
        count_only: bool = False, dry_run: bool = False,
        fetcher: Fetcher | None = None, db_path: Path | None = None) -> CrawlStats:
    fetcher = fetcher or Fetcher()
    wanted = set(ministries or [])
    keywords = [k for k in (title_keywords or []) if k.strip()]
    stats = CrawlStats()
    with open_db(db_path) as conn:
        for query in queries:
            page, last_page = 1, 1
            while page <= last_page and (max_pages is None or page <= max_pages):
                result = fetcher.get(LIST_URL, params=_list_params(query, page, start, end))
                if result.status != 200:
                    print(f"[{query}] {page}쪽 목록 응답 {result.status} — 중단")
                    break
                listing = parse_list(result.text)
                last_page = listing.last_page
                stats.pages += 1
                if page == 1:
                    print(f"[{query}] {start} ~ {end} 검색 결과 {listing.total if listing.total is not None else '?'}건"
                          f" · {last_page}쪽")
                    if count_only:
                        break
                for item in listing.items:
                    stats.listed += 1
                    if wanted and item.ministry not in wanted:
                        stats.skipped_ministry += 1
                        continue
                    if keywords and not any(k in item.title for k in keywords):
                        stats.skipped_title += 1
                        continue
                    if _saved_raw_path(conn, make_doc_id(DOC_ID_PREFIX, item.url, item.published_at)):
                        stats.already += 1
                        continue
                    label = f"{item.published_at} {item.ministry} {item.title[:40]}"
                    if dry_run:
                        stats.planned.append(label)
                        print(f"  대상 {len(stats.planned)}: {label}", flush=True)
                        continue
                    try:
                        record = collect_item(fetcher, item, query)
                    except NoOriginal as exc:
                        stats.no_original.append(f"{label} — {exc}")
                        continue
                    except (requests.RequestException, RobotsDisallowed, ValueError) as exc:
                        message = describe_error(exc) if isinstance(exc, requests.RequestException) else str(exc)
                        stats.failed.append(f"{label} — {message}")
                        continue
                    upsert_documents(conn, [record])
                    conn.commit()           # 문서마다 저장해 중단돼도 받은 만큼은 남는다
                    stats.saved += 1
                    print(f"  저장 {stats.saved}: {label}", flush=True)
                print(f"[{query}] {page}/{last_page}쪽 완료 · 누적 저장 {stats.saved}"
                      + (f" · 대상 {len(stats.planned)}" if dry_run else ""), flush=True)
                page += 1
    print(stats.summary())
    return stats

