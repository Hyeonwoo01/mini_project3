"""수집 후보 사이트 접근 확인 — 크롤러를 만들기 전에 실제로 수집할 PC 에서 실행한다.

    python -m collector probe                  # 후보 전체
    python -m collector probe --only fsc mss   # 일부만

후보마다 다음을 출력하고 data/raw/_probe/report.txt 에도 저장한다.
- robots.txt 판단 (허용·금지·확인 불가)과 Crawl-delay. 허용하지 않으면 페이지를 요청하지 않는다
- 페이지 응답 코드·형식·크기·제목, 본문 글자 수, 링크 수 (그중 javascript: 링크 수).
  링크와 본문이 거의 없으면 화면을 자바스크립트로 그리는 사이트라 정적 수집이 어렵다
- 키워드가 들어간 링크 (게시판 주소를 찾는 데 쓴다)와 javascript 링크 예시 (상세 페이지 호출 방식 확인)
- 상세 페이지 주소 형식에 맞는 링크 수와 첫 상세 페이지 응답
받은 원본은 data/raw/_probe/<후보>_list.html, <후보>_detail.html 로 저장한다. 파서를 실제 HTML 에
맞춰 작성하기 위한 표본이며 data/raw/ 는 git 에 올리지 않는다.

후보 주소는 웹 검색으로 찾았다. verified=False 인 주소는 검색 결과로 확인하지 못한 추정이며,
이 명령의 결과로 확정하거나 바꾼다.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlencode, urljoin

import requests
from bs4 import BeautifulSoup

from collector.fetch import ALLOWED, Fetcher, FetchResult, RobotsDisallowed, describe_error
from common.config import get_settings

KEYWORD_LINK_LIMIT = 8
JS_SAMPLE_LIMIT = 3
ATTACHMENT_SAMPLE_LIMIT = 3
TITLE_LIKE_CHARS = 8      # 이보다 긴 링크 글자는 게시물 제목일 가능성이 높다 (메뉴 링크는 짧다)
PREVIEW_CHARS = 200
# 첨부파일(원문 HWP·PDF) 링크로 보이는 주소. 사이트마다 다운로드 경로 이름이 달라 흔한 형태를 모두 본다
ATTACHMENT_HINT = re.compile(r"download|getfile|filedown|atchfile|\.(?:hwpx?|pdf)(?:$|[?#&])", re.IGNORECASE)


@dataclass(frozen=True)
class ProbeTarget:
    key: str
    name: str
    url: str
    keywords: tuple[str, ...] = ()
    detail_pattern: str | None = None   # 상세 페이지 주소 정규식 (절대 URL 기준)
    verified: bool = False              # 주소를 웹 검색 결과로 확인했는가


PROBE_TARGETS: tuple[ProbeTarget, ...] = (
    ProbeTarget("korea_kr", "정책브리핑 보도자료 목록 (최신)", "https://www.korea.kr/briefing/pressReleaseList.do",
                keywords=("보도자료",), detail_pattern=r"pressReleaseView\.do", verified=True),
    # 목록 주소에 보이는 검색어·기간 파라미터가 실제로 결과를 거르는지 확인한다
    ProbeTarget("korea_kr_search", "정책브리핑 보도자료 검색 (검색어 소상공인, 기간 지정)",
                "https://www.korea.kr/briefing/pressReleaseList.do?" + urlencode(
                    {"pageIndex": 1, "startDate": "2021-10-16", "endDate": "2026-10-16", "srchWord": "소상공인"}),
                keywords=("소상공인",), detail_pattern=r"pressReleaseView\.do", verified=True),
    ProbeTarget("fsc", "금융위원회 보도자료 목록 (카드수수료)", "https://www.fsc.go.kr/no010101",
                keywords=("보도자료",), detail_pattern=r"fsc\.go\.kr/no010101/\d+", verified=True),
    # 게시판 주소는 첫 화면 probe 로 확인했다. 상세 주소 형식은 추정 (javascript 로 열면 JS링크에 나온다)
    ProbeTarget("mss_press", "중소벤처기업부 보도자료 게시판", "https://www.mss.go.kr/site/smba/ex/bbs/List.do?cbIdx=86",
                keywords=("소상공인",), detail_pattern=r"View\.do\?cbIdx=86", verified=True),
    ProbeTarget("seoul_economy", "서울시 경제분야 누리집 소상공인 소식",
                "https://news.seoul.go.kr/economy/archives/category/nationaleconomy-news_c1/"
                "street_biz_nationaleconomy-news-n1/",
                keywords=("보도자료", "소상공인"), detail_pattern=r"news\.seoul\.go\.kr/economy/archives/\d+",
                verified=True),
    # robots.txt 가 금지 (2026-11-03 확인) → 수집하지 않는다. 판단 근거를 남기려고 확인 대상에 둔다
    ProbeTarget("seoul", "서울시 보도자료 목록", "https://www.seoul.go.kr/news/news_report.do",
                keywords=("보도자료",), verified=True),
    ProbeTarget("law", "국가법령정보센터 Open API (인증값 OC 없이 응답만 확인)",
                "https://www.law.go.kr/DRF/lawSearch.do?target=law&type=XML&query=%EC%86%8C%EC%83%81%EA%B3%B5%EC%9D%B8",
                verified=True),
    ProbeTarget("data_go_kr", "공공데이터포털 Open API 서버", "https://apis.data.go.kr/", verified=True),
    # 연구기관 보고서 (선별 수집 10~20건). 보고서 목록 주소를 아직 모르므로 첫 화면에서 '연구보고서' 등
    # 키워드 링크를 찾는다. 주소는 기관 대표 주소 (웹 검색으로 목록 주소를 확인하지 못함)
    ProbeTarget("nkis", "국가정책연구포털 (국책연구기관 보고서 통합)", "https://www.nkis.re.kr/",
                keywords=("연구보고서", "통합검색", "보고서")),
    ProbeTarget("si", "서울연구원", "https://www.si.re.kr/", keywords=("연구보고서", "정책리포트", "보고서")),
    ProbeTarget("kosi", "중소벤처기업연구원", "https://www.kosi.re.kr/", keywords=("연구보고서", "소상공인", "보고서")),
    ProbeTarget("kdi", "한국개발연구원", "https://www.kdi.re.kr/", keywords=("연구보고서", "KDI FOCUS", "보고서")),
    ProbeTarget("kiet", "산업연구원", "https://www.kiet.re.kr/", keywords=("연구보고서", "이슈", "보고서")),
    ProbeTarget("kipf", "한국조세재정연구원", "https://www.kipf.re.kr/", keywords=("연구보고서", "보고서")),
)


@dataclass
class PageInfo:
    title: str = ""
    text_chars: int = 0
    link_count: int = 0
    js_links: int = 0
    js_samples: list[str] = field(default_factory=list)
    keyword_links: list[tuple[str, str]] = field(default_factory=list)   # (링크 글자, 절대 URL)
    detail_urls: list[str] = field(default_factory=list)
    attachments: list[tuple[str, str]] = field(default_factory=list)     # (링크 글자, 절대 URL)


def analyze_html(html: str, base_url: str, keywords: tuple[str, ...] = (),
                 detail_pattern: str | None = None) -> PageInfo:
    soup = BeautifulSoup(html, "lxml")
    info = PageInfo(title=soup.title.get_text(" ", strip=True) if soup.title else "")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    info.text_chars = len(soup.get_text(" ", strip=True))
    detail = re.compile(detail_pattern) if detail_pattern else None
    seen_keyword, seen_detail, seen_attachment = set(), set(), set()
    js_titles, js_others = [], []     # 게시물 제목처럼 긴 링크의 호출 방식을 먼저 보여 준다
    for a in soup.find_all("a"):
        href = (a.get("href") or "").strip()
        onclick = (a.get("onclick") or "").strip()
        if not href and not onclick:
            continue
        info.link_count += 1
        text = a.get_text(" ", strip=True)
        if href.lower().startswith("javascript:") or (onclick and href in ("", "#")):
            info.js_links += 1
            (js_titles if len(text) >= TITLE_LIKE_CHARS else js_others).append((onclick or href)[:120])
            continue
        url = urljoin(base_url, href)
        if keywords and any(k in text for k in keywords) and url not in seen_keyword:
            seen_keyword.add(url)
            if len(info.keyword_links) < KEYWORD_LINK_LIMIT:
                info.keyword_links.append((text[:40], url))
        if detail and detail.search(url) and url not in seen_detail:
            seen_detail.add(url)
            info.detail_urls.append(url)
        if ATTACHMENT_HINT.search(url) and url not in seen_attachment:
            seen_attachment.add(url)
            info.attachments.append((text[:40], url))
    info.js_samples = list(dict.fromkeys(js_titles + js_others))[:JS_SAMPLE_LIMIT]
    return info


def _describe(result: FetchResult) -> str:
    return (f"{result.status} · {result.content_type or '형식 미상'} · {len(result.content) / 1024:.1f}KB"
            f" · 인코딩 {result.encoding or '미상'}")


def _preview(result: FetchResult) -> str:
    return " ".join(result.text.split())[:PREVIEW_CHARS]


def probe_target(fetcher: Fetcher, target: ProbeTarget, out_dir: Path, emit) -> str:
    """후보 하나를 확인하고 요약 한 줄을 반환한다. 자세한 내용은 emit 으로 내보낸다."""
    emit(f"[{target.key}] {target.name}" + ("" if target.verified else "  (주소 추정)"))
    emit(f"  URL     {target.url}")
    robots = fetcher.check_robots(target.url)
    delay = f", Crawl-delay {robots.crawl_delay:g}초" if robots.crawl_delay else ""
    emit(f"  robots  {robots.status} ({robots.detail}{delay})")
    if robots.status != ALLOWED:
        emit("  → robots.txt 가 허용하지 않거나 확인할 수 없어 페이지를 요청하지 않음")
        return f"robots {robots.status}"

    try:
        result = fetcher.get(target.url)
    except requests.RequestException as exc:
        emit(f"  응답    실패 — {describe_error(exc)}")
        return f"요청 실패 {type(exc).__name__}"
    emit(f"  응답    {_describe(result)}")
    if result.url != target.url:
        emit(f"  최종URL {result.url}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{target.key}_list.html").write_bytes(result.content)

    if "html" not in result.content_type.lower():
        emit(f"  내용    {_preview(result)}")
        return f"{result.status} (HTML 아님)"

    info = analyze_html(result.text, result.url, target.keywords, target.detail_pattern)
    emit(f"  제목    {info.title or '(없음)'}")
    emit(f"  본문    {info.text_chars:,}자 · 링크 {info.link_count}개 (javascript {info.js_links}개)")
    for text, url in info.keyword_links:
        emit(f"  키워드  {text} → {url}")
    for sample in info.js_samples:
        emit(f"  JS링크  {sample}")
    summary = f"{result.status} · 링크 {info.link_count}"
    if target.detail_pattern is None:
        return summary

    emit(f"  상세    주소 형식에 맞는 링크 {len(info.detail_urls)}개")
    if not info.detail_urls:
        return summary + " · 상세 링크 0"
    first = info.detail_urls[0]
    emit(f"  첫상세  {first}")
    try:
        detail = fetcher.get(first)
    except (requests.RequestException, RobotsDisallowed) as exc:
        emit(f"          실패 — {describe_error(exc)}")
        return summary + f" · 상세 {len(info.detail_urls)}건, 요청 실패"
    (out_dir / f"{target.key}_detail.html").write_bytes(detail.content)
    detail_info = analyze_html(detail.text, detail.url) if "html" in detail.content_type.lower() else PageInfo()
    emit(f"          {_describe(detail)} · 제목 {detail_info.title or '(없음)'} · 본문 {detail_info.text_chars:,}자")
    summary += f" · 상세 {len(info.detail_urls)}건, 첫 상세 {detail.status}"
    if not detail_info.attachments:
        emit("  첨부    첨부파일로 보이는 링크 없음")
        return summary
    # 첨부는 내려받지 않고, 내려받아도 되는지(robots.txt)만 확인한다
    first_attachment = detail_info.attachments[0][1]
    emit(f"  첨부    {len(detail_info.attachments)}개 · 첫 첨부 robots {fetcher.check_robots(first_attachment).status}")
    for text, url in detail_info.attachments[:ATTACHMENT_SAMPLE_LIMIT]:
        emit(f"          {text or '(글자 없음)'} → {url}")
    return summary + f", 첨부 {len(detail_info.attachments)}"


def run(only: list[str] | None = None, out_dir: Path | None = None, fetcher: Fetcher | None = None) -> None:
    targets = [t for t in PROBE_TARGETS if not only or t.key in only]
    unknown = sorted(set(only or []) - {t.key for t in PROBE_TARGETS})
    if unknown:
        raise SystemExit(f"없는 후보: {', '.join(unknown)} (가능: {', '.join(t.key for t in PROBE_TARGETS)})")
    out_dir = out_dir or get_settings().raw_dir / "_probe"
    fetcher = fetcher or Fetcher()
    lines: list[str] = []

    def emit(line: str) -> None:
        print(line, flush=True)
        lines.append(line)

    emit(f"User-Agent: {fetcher.user_agent} · 요청 간격 {fetcher.delay:g}초 이상")
    emit("")
    summaries = []
    for target in targets:
        summaries.append((target.key, probe_target(fetcher, target, out_dir, emit)))
        emit("")
    emit("요약")
    width = max(len(key) for key, _ in summaries)
    for key, summary in summaries:
        emit(f"  {key:<{width}}  {summary}")

    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "report.txt"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n저장: {report} (받은 HTML 도 같은 폴더)")
