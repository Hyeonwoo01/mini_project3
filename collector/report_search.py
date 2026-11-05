"""연구기관 보고서 후보 검색 — 선별 수집의 1단계 (README '연구기관 보고서 선별 기준').

    python -m collector reports-search                     # 전체 기관 × 주제어
    python -m collector reports-search --only kdi_focus     # 일부 기관만

기관 목록에서 주제어로 검색한 결과(제목·발행일·상세 주소)를 data/raw/_probe/reports/candidates.tsv 로 모은다.
이 목록에서 선별 기준(기간·주제·주제별 2~4건)으로 10~20건을 골라 수집 목록을 정한다. 보고서 본문은 받지 않는다.
기관마다 검색 결과 목록 1쪽과 첫 후보의 상세 페이지를 표본으로 함께 저장한다 (수집기 파서를 실제 HTML 에 맞추기 위해).

사이트 구조 (2026-11-04, collector probe 2차 표본으로 확인)
- KDI 연구보고서·FOCUS: 목록 GET ?pg=&pp=&nm=검색어. 항목 li > a[href=./reportView?pub_no=N] (제목 앞에 '정책연구시리즈'
  등 구분). 목록에는 발행일이 없다 (상세 strong.title span)
- 산업연구원 이슈·연구보고서: 목록 GET ?pp=&pg=&skey=A(제목)&sval=검색어. 항목 li.item a[href=./issueView?issue_no=N],
  같은 항목 안에 '2026.10.26' 발행일
- 중소벤처기업연구원 정책연구 보고서: 목록 주소 뒤에 &titleMain=검색어 (사이트 검색 스크립트가 만드는 주소와 같다).
  항목 h4.title a[href=...basicResearchDetail?dataSequence=코드]. 목록에는 발행일이 없다
- 한국조세재정연구원 연구발간자료: 목록 list.do ?pageIndex=&searchCondition=all&searchKeyword=검색어. 사이트 폼은 POST 지만
  GET 파라미터로 같은 검색이 되는지 이 결과로 확인한다. 항목 a[onclick=fn_search_detail('일련번호')], 글자에 '발간월 2026-10'
- 서울연구원 연구보고서·정책리포트: 게시판 /bbs/list.do?key=&sw=검색어 (사이트 폼은 POST, GET 동작은 결과로 확인).
  항목은 a[onclick=goView('글번호')], 상세 /bbs/view.do?key=&pstSn=글번호 (첫 화면 링크로 확인). 1차 검색에서는
  항목 링크 형식을 몰라 0건이었다
- 국가정책연구포털(NKIS)은 검색·상세가 모두 POST·자바스크립트라 쓰지 않는다. 소관 기관(KDI·산업연구원·조세연)은
  기관 누리집에서 직접 찾는다

사이트 검색이 검색어를 무시하면 최신 목록이 그대로 나온다. 그래서 제목에 주제어가 있는지(title_topics)를 따로 적고,
기관별로 제목이 주제어와 맞는 후보가 하나도 없으면 경고한다.
"""

import csv
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote, urlencode

import requests
from bs4 import BeautifulSoup

from collector.fetch import Fetcher, RobotsDisallowed, describe_error
from common.config import get_settings

# 선별 기준의 주제 → 검색어. 검색어는 주제 이름의 핵심 단어만 쓴다 (평가 문항의 표현을 넣지 않는다)
TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "소상공인·자영업": ("소상공인", "자영업"),
    "골목상권·전통시장": ("상권", "전통시장"),
    "지역화폐·소비 지원": ("지역화폐", "지역사랑상품권", "소비쿠폰"),
    "유통 규제": ("대형마트", "의무휴업", "유통"),
    "카드수수료·가맹점": ("카드", "가맹점", "수수료"),
}
PERIOD = ("2021-10-16", "2026-10-16")   # 제공 뉴스 기간 (선별 기준)
CANDIDATES_FILE = "candidates.tsv"
_DATE = re.compile(r"(20\d\d)[.\-/]\s?(\d{1,2})(?:[.\-/]\s?(\d{1,2}))?")
_TITLE_TAIL = re.compile(r"\s+(?:저자\s*:?|발간월|관련키워드|자세히보기).*$")

KOSI_MENU = "59996687477237983229272069890802427549"


@dataclass(frozen=True)
class ReportSource:
    key: str
    institute: str
    search_url: Callable[[str], str]        # 검색어 → 목록 주소 (searchable=False 면 검색어를 쓰지 않는다)
    item_link: re.Pattern                   # 항목 링크의 href·onclick 에서 상세를 가리키는 부분
    detail_url: Callable[[re.Match], str]
    searchable: bool = True


def _kdi(board: str, view: str) -> ReportSource:
    return ReportSource(
        key=f"kdi_{board}", institute="한국개발연구원",
        search_url=lambda kw: f"https://www.kdi.re.kr/research/{board}List?" + urlencode({"pg": 1, "pp": 30, "nm": kw}),
        item_link=re.compile(rf"{view}\?pub_no=(\d+)"),
        detail_url=lambda m: f"https://www.kdi.re.kr/research/{view}?pub_no={m[1]}",
    )


def _kiet(board: str, view: str, id_name: str) -> ReportSource:
    return ReportSource(
        key=f"kiet_{board}", institute="산업연구원",
        search_url=lambda kw: (f"https://www.kiet.re.kr/research/{board}List?"
                               + urlencode({"pp": 20, "pg": 1, "skey": "A", "sval": kw})),
        item_link=re.compile(rf"{view}\?{id_name}=(\d+)"),
        detail_url=lambda m: f"https://www.kiet.re.kr/research/{view}?{id_name}={m[1]}",
    )


def _si(name: str, board_key: str) -> ReportSource:
    return ReportSource(
        key=f"si_{name}", institute="서울연구원",
        search_url=lambda kw: "https://www.si.re.kr/bbs/list.do?" + urlencode({"key": board_key, "sw": kw, "pageIndex": 1}),
        item_link=re.compile(r"goView\('(\d+)'\)"),
        detail_url=lambda m: f"https://www.si.re.kr/bbs/view.do?key={board_key}&pstSn={m[1]}",
    )


REPORT_SOURCES: tuple[ReportSource, ...] = (
    _kdi("report", "reportView"),
    _kdi("focus", "focusView"),
    _kiet("issue", "issueView", "issue_no"),
    _kiet("report", "reportView", "report_no"),
    ReportSource(
        key="kosi_report", institute="중소벤처기업연구원",
        # 사이트 검색 스크립트(searchFormApply)가 만드는 주소. menuFrontURL 값 안의 '?' 를 그대로 둔다
        search_url=lambda kw: (f"https://www.kosi.re.kr/front/functionDisplay?menuFrontNo={KOSI_MENU}"
                               f"&menuFrontURL=front/basicResearchList?limit=12&titleMain={quote(kw)}"),
        item_link=re.compile(r"basicResearchDetail\?dataSequence=(\w+)"),
        detail_url=lambda m: (f"https://www.kosi.re.kr/front/functionDisplay?menuFrontNo={KOSI_MENU}"
                              f"&menuFrontURL=front/basicResearchDetail?dataSequence={m[1]}"),
    ),
    ReportSource(
        key="kipf_all", institute="한국조세재정연구원",
        search_url=lambda kw: ("https://www.kipf.re.kr/kor/Publication/All/kiPublish/ALL/list.do?"
                               + urlencode({"pageIndex": 1, "searchCondition": "all", "searchKeyword": kw})),
        item_link=re.compile(r"fn_search_detail\('(\d+)'\)"),
        detail_url=lambda m: f"https://www.kipf.re.kr/kor/Publication/All/kiPublish/ALL/view.do?serialNo={m[1]}",
    ),
    _si("report", "2024100039"),
    _si("policy", "2024100154"),
)

# 수집기 작성에 필요한 추가 표본: 중기연 PDF 내려받기 주소는 공용 스크립트의 kosbi.dbUrl 로 만든다
EXTRA_SAMPLES: tuple[tuple[str, str], ...] = (("kosi_script.js", "https://www.kosi.re.kr/js/script.js"),)


@dataclass
class Candidate:
    source: str
    institute: str
    title: str
    url: str
    published: str | None                 # YYYY-MM-DD 또는 YYYY-MM (목록에 없으면 None)
    keywords: list[str] = field(default_factory=list)

    @property
    def title_topics(self) -> list[str]:
        return [topic for topic, words in TOPIC_KEYWORDS.items() if any(w in self.title for w in words)]

    @property
    def in_period(self) -> bool | None:
        if not self.published:
            return None
        start, end = PERIOD
        size = len(self.published)          # 월까지만 있으면 월 단위로 비교한다
        return start[:size] <= self.published <= end[:size]


def _clean(text: str) -> str:
    return " ".join(text.split())


def parse_date(text: str) -> str | None:
    match = _DATE.search(text)
    if not match:
        return None
    year, month, day = match.groups()
    return f"{year}-{int(month):02d}" + (f"-{int(day):02d}" if day else "")


def _item_url(anchor, source: ReportSource) -> str | None:
    match = source.item_link.search(f"{anchor.get('href') or ''} {anchor.get('onclick') or ''}")
    return source.detail_url(match) if match else None


def parse_candidates(html: str, source: ReportSource, keyword: str | None) -> list[Candidate]:
    """목록 HTML 에서 상세 링크가 있는 항목을 뽑는다. 같은 상세 주소는 한 번만 (글자가 있는 첫 링크)."""
    soup = BeautifulSoup(html, "lxml")
    found: dict[str, Candidate] = {}
    for anchor in soup.find_all("a"):
        url = _item_url(anchor, source)
        text = _clean(anchor.get_text(" ", strip=True))
        if not url or not text or url in found:
            continue
        published = parse_date(text)
        if published is None:   # 같은 항목(li·tr) 안의 발행일. 항목 하나에 상세 주소가 하나일 때만 쓴다
            row = anchor.find_parent(["li", "tr"])
            if row is not None and {_item_url(a, source) for a in row.find_all("a")} - {None} == {url}:
                published = parse_date(row.get_text(" ", strip=True))
        found[url] = Candidate(source.key, source.institute, _TITLE_TAIL.sub("", text), url, published,
                               [keyword] if keyword else [])
    return list(found.values())


def merge(candidates: Iterable[Candidate]) -> list[Candidate]:
    """같은 상세 주소는 하나로 합치고 찾은 검색어를 모은다."""
    merged: dict[str, Candidate] = {}
    for cand in candidates:
        if cand.url in merged:
            kept = merged[cand.url]
            kept.keywords += [k for k in cand.keywords if k not in kept.keywords]
            kept.published = kept.published or cand.published
        else:
            merged[cand.url] = Candidate(cand.source, cand.institute, cand.title, cand.url, cand.published,
                                         list(cand.keywords))
    return list(merged.values())


def write_candidates(candidates: Sequence[Candidate], path: Path) -> None:
    # 기관별로, 제목에 주제어가 있는 후보를 먼저, 그 안에서 최신순
    rows = sorted(candidates, key=lambda c: c.published or "", reverse=True)
    rows.sort(key=lambda c: (c.source, not c.title_topics))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["source", "institute", "published", "in_period", "title_topics", "keywords", "title", "url"])
        for c in rows:
            period = "" if c.in_period is None else ("Y" if c.in_period else "N")
            writer.writerow([c.source, c.institute, c.published or "", period, " · ".join(c.title_topics),
                             " ".join(c.keywords), c.title, c.url])


@dataclass
class SourceStats:
    requests: int = 0
    failures: list[str] = field(default_factory=list)


def _fetch_html(fetcher: Fetcher, url: str, stats: SourceStats) -> str | None:
    stats.requests += 1
    try:
        result = fetcher.get(url)
    except (requests.RequestException, RobotsDisallowed) as exc:
        stats.failures.append(f"{url} — {describe_error(exc)}")
        return None
    if result.status != 200:
        stats.failures.append(f"{url} — HTTP {result.status}")
        return None
    return result.text


def search_source(fetcher: Fetcher, source: ReportSource, keywords: Sequence[str], out_dir: Path,
                  emit) -> tuple[list[Candidate], SourceStats]:
    stats = SourceStats()
    found: list[Candidate] = []
    queries: list[str | None] = list(keywords) if source.searchable else [None]
    for i, keyword in enumerate(queries):
        html = _fetch_html(fetcher, source.search_url(keyword or ""), stats)
        if html is None:
            continue
        if i == 0:
            (out_dir / f"{source.key}_list.html").write_text(html, encoding="utf-8")
        found += parse_candidates(html, source, keyword)
    merged = merge(found)
    if not source.searchable:          # 검색하지 않은 목록은 제목에 주제어가 있는 항목만 남긴다
        merged = [c for c in merged if c.title_topics]
    if found:                          # 첫 후보의 상세 페이지 (수집기 파서 표본)
        detail = _fetch_html(fetcher, found[0].url, stats)
        if detail is not None:
            (out_dir / f"{source.key}_detail.html").write_text(detail, encoding="utf-8")
    matched = sum(1 for c in merged if c.title_topics)
    emit(f"[{source.key}] {source.institute} · 요청 {stats.requests} · 후보 {len(merged)} (제목에 주제어 {matched})"
         + (f" · 실패 {len(stats.failures)}" if stats.failures else ""))
    if source.searchable and found and matched == 0:
        emit("  경고: 제목에 주제어가 있는 후보가 없다 — 사이트가 검색어를 무시했을 수 있다 (표본 목록 HTML 확인)")
    for failure in stats.failures[:3]:
        emit(f"  실패: {failure}")
    return merged, stats


def run(only: Sequence[str] | None = None, keywords: Sequence[str] | None = None, out_dir: Path | None = None,
        fetcher: Fetcher | None = None) -> Path:
    sources = [s for s in REPORT_SOURCES if not only or s.key in only]
    unknown = sorted(set(only or []) - {s.key for s in REPORT_SOURCES})
    if unknown:
        raise SystemExit(f"없는 기관 key: {', '.join(unknown)} (가능: {', '.join(s.key for s in REPORT_SOURCES)})")
    keywords = list(keywords or [w for words in TOPIC_KEYWORDS.values() for w in words])
    out_dir = out_dir or get_settings().raw_dir / "_probe" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    fetcher = fetcher or Fetcher()
    lines: list[str] = []

    def emit(line: str) -> None:
        print(line, flush=True)
        lines.append(line)

    emit(f"검색어 {len(keywords)}개: {' '.join(keywords)} · 요청 간격 {fetcher.delay:g}초 이상")
    candidates: list[Candidate] = []
    for source in sources:
        found, _ = search_source(fetcher, source, keywords, out_dir, emit)
        candidates += found
    if not only:
        for name, url in EXTRA_SAMPLES:
            stats = SourceStats()
            text = _fetch_html(fetcher, url, stats)
            if text is not None:
                (out_dir / name).write_text(text, encoding="utf-8")
            emit(f"[표본] {name} " + ("저장" if text is not None else f"실패 — {stats.failures[0]}"))

    path = out_dir / CANDIDATES_FILE
    write_candidates(candidates, path)
    in_period = sum(1 for c in candidates if c.title_topics and c.in_period is not False)
    emit(f"\n후보 {len(candidates)}건 (제목에 주제어가 있고 기간 밖이 아닌 것 {in_period}건) → {path}")
    (out_dir / "search_log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
