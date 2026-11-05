"""연구기관 보고서 선별 수집 — 2단계 (1단계 후보 검색은 collector.report_search).

    python -m collector reports --dry-run     # 수집 목록만 확인
    python -m collector reports               # 수집

후보 목록(candidates.tsv)에서 README '연구기관 보고서 선별 기준'으로 고른 보고서의 상세 화면을 받아, 화면이 제공하는
요약(짧은 보고서는 전문)을 본문으로 저장한다 (doc_type policy_report, 1차 자료).

PDF 를 받지 않는 이유 (2026-11-04, 2차 probe·후보 검색 표본으로 확인)
- 내려받기 방식이 기관마다 다르다: KDI 는 GET(/file/download?atch_no=), 중기연은 스크립트가 암호화한 주소를 POST,
  조세연은 다운로드 목적을 묻는 팝업, 산업연구원 연구보고서는 자바스크립트 함수. 기관별로 우회 구현을 하지 않는다
- 선별 기준의 색인 범위(요약·정책 제언·결론)는 상세 화면 요약으로 대부분 덮인다. KDI FOCUS 는 상세 화면에 전문이 있다
- 연구보고서의 결론 장은 들어가지 않는다 (README 에 한계로 기록)

상세 화면 구조
- KDI: og:title 이 '제목 - KDI 한국개발연구원 - …'. 발행일은 본문 영역(section#contents)의 첫 'YYYY.MM.DD'.
  FOCUS 는 div.editor-template 에 전문, 연구보고서는 div.bundlebox.desc 중 '국문요약' 으로 시작하는 블록
- 중기연: 표(th 제목·발행일 → td). 요약은 가장 긴 td ('01 연구의 배경 및 목적 …')
- 산업연구원: og:title 이 '제목 - KIET 산업연구원 - …'. 발행일은 div.page_view 의 첫 'YYYY.MM.DD'.
  요약은 div.p_d-cont ('요약 언론보도 …' 탭 글자 뒤). 연구보고서는 이 칸에 목차만 있어 요약이 없는 것으로 처리한다
- 이용 조건: 각 기관 누리집 저작권 정책(출처 표시, 비상업 이용)에 따라 출처·주소를 문서마다 남긴다
"""

import copy
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from collector.fetch import Fetcher, RobotsDisallowed, describe_error
from collector.report_search import PERIOD, parse_date
from collector.storage import make_doc_id, save_raw
from common.config import get_settings
from common.docstore import open_db, upsert_documents
from common.schema import CollectMethod, DocType, DocumentRecord

DOC_ID_PREFIX = "rpt"
MIN_TEXT_CHARS = 300          # 이보다 짧으면 요약이 없는 것으로 본다 (목차·안내문만 있는 화면)
_BLOCK_TAGS = ("p", "div", "li", "dt", "dd", "tr", "table", "ul", "ol", "dl", "section",
               "h1", "h2", "h3", "h4", "h5", "h6")
KOSI_DETAIL = ("https://www.kosi.re.kr/front/functionDisplay?menuFrontNo=59996687477237983229272069890802427549"
               "&menuFrontURL=front/basicResearchDetail?dataSequence=")
INSTITUTES = {"kdi": "한국개발연구원", "kosi": "중소벤처기업연구원", "kiet": "산업연구원"}


@dataclass(frozen=True)
class CuratedReport:
    site: str        # kdi | kosi | kiet (상세 화면 파서)
    url: str
    title: str       # 후보 목록의 제목 (상세 화면 제목이 있으면 그것을 쓴다)
    topic: str       # 선별 기준의 주제
    reason: str      # 고른 이유


# 선별 (2026-11-04, candidates.tsv 124건 중). 기준: 기간 2021-10~2026-10, 주제별 최대 4건, 최신 우선,
# 상세 화면에 요약이 있는 기관. 지역화폐·소비 지원과 카드수수료·가맹점은 기간 안 후보가 없어 0건이고, 그래서 아직
# 10건이 안 된다 (서울연구원 후보로 보충 예정). 제외 사유는 README '연구기관 보고서' 기록 참조
CURATED_REPORTS: tuple[CuratedReport, ...] = (
    # 유통 규제 — 평가 문항 s004(의무휴업 규제 완화의 영향)의 근거가 지금은 뉴스로만 들어 있다
    CuratedReport("kdi", "https://www.kdi.re.kr/research/focusView?pub_no=19187",
                  "의무휴업일 평일 전환이 시사하는 유통정책의 전환 방향", "유통 규제",
                  "의무휴업일 평일 전환 지역의 카드 매출 분석 원문 (s004 정답 기준의 KDI 분석)"),
    CuratedReport("kdi", "https://www.kdi.re.kr/research/focusView?pub_no=19221",
                  "온라인 유통의 성장과 유통시장 정책 개선 방향", "유통 규제",
                  "온라인 유통 성장에 따른 대형마트 규제 개선 방향"),
    CuratedReport("kdi", "https://www.kdi.re.kr/research/reportView?pub_no=19217",
                  "유통시장의 환경 변화와 경제적 영향에 관한 연구: 시장구조·정책·수요 변화를 중심으로", "유통 규제",
                  "위 두 FOCUS 의 바탕 연구보고서 (국문요약)"),
    CuratedReport("kiet", "https://www.kiet.re.kr/research/reportView?report_no=1126",
                  "유통산업 디지털 전환 전략 연구 - 대규모점포 규제 효과를 중심으로", "유통 규제",
                  "대규모점포 규제 효과 분석 (산업연구원 2024-10). 상세 화면에 요약이 없으면 수집하지 않는다"),
    # 소상공인·자영업 — 최신 순
    CuratedReport("kosi", KOSI_DETAIL + "J260416K02", "거시환경 변화에 따른 정책방향 연구 : 소상공인･상생분야",
                  "소상공인·자영업", "소상공인 정책 방향 (최신)"),
    CuratedReport("kosi", KOSI_DETAIL + "J260126K01", "2026년 중소기업･소상공인 10대 이슈", "소상공인·자영업",
                  "2026년 소상공인 주요 이슈 개관"),
    CuratedReport("kosi", KOSI_DETAIL + "B260522K02", "소상공인 특화 생존지수 (Survival Index) 개발 및 지역경제 활성화 방안",
                  "소상공인·자영업", "지역별 소상공인 생존지수 — 지역 상권 분석 업무와 직접 연결"),
    CuratedReport("kosi", KOSI_DETAIL + "H250319K15", "빅데이터를 활용한 소상공인 경영실태 분석", "소상공인·자영업",
                  "매출 데이터 기반 경영실태 분석 — 사용자(카드 데이터 분석가)의 데이터와 같은 종류"),
    # 골목상권·전통시장 — 기간 안 후보가 1건
    CuratedReport("kosi", KOSI_DETAIL + "H250319K01", "지방소멸 시대 지속가능한 상권 생태계 재구축 방안",
                  "골목상권·전통시장", "지역 상권 생태계 정책 방향"),
)


@dataclass(frozen=True)
class ReportPage:
    title: str | None
    published_at: str | None     # YYYY-MM-DD
    text: str
    text_kind: str               # 전문 | 요약


def _lines(tag) -> str:
    """문단(블록 요소·<br>) 단위로 줄을 나눈 본문. 강조용 <span> 같은 인라인 요소에서는 줄을 바꾸지 않는다
    (KDI FOCUS 는 문장 중간에 강조 span 이 있어 태그마다 줄을 나누면 문장이 끊긴다)."""
    tag = copy.copy(tag)
    for br in tag.find_all("br"):
        br.replace_with("\n")
    for block in tag.find_all(_BLOCK_TAGS):
        block.insert_before("\n")
        block.insert_after("\n")
    lines = (" ".join(line.split()) for line in tag.get_text().splitlines())
    return "\n".join(line for line in lines if line)


def _og_title(soup: BeautifulSoup) -> str:
    meta = soup.find("meta", property="og:title")
    return " ".join(((meta.get("content") if meta else "") or "").split())


def _full_date(text: str) -> str | None:
    value = parse_date(text)
    return value if value and len(value) == 10 else None


def parse_kdi(html: str) -> ReportPage:
    soup = BeautifulSoup(html, "lxml")
    og_title = _og_title(soup)
    title = og_title.split(" - KDI")[0].strip() or None
    contents = soup.select_one("section#contents") or soup
    published = _full_date(contents.get_text(" ", strip=True))
    if og_title.endswith("KDI FOCUS"):     # FOCUS: 화면에 전문. 끝에 붙는 목차는 뺀다
        bodies = sorted(contents.select("div.editor-template"), key=lambda t: len(t.get_text()), reverse=True)
        text = re.split(r"\nKDI FOCUS\s*목차", _lines(bodies[0]))[0] if bodies else ""
        return ReportPage(title, published, text, "전문")
    for block in contents.select("div.bundlebox.desc"):   # 연구보고서: 국문요약 (영문요약·목차 블록은 건너뛴다)
        text = _lines(block)
        if text.startswith("국문요약"):
            return ReportPage(title, published, re.sub(r"^국문요약\s*(?:열기)?\s*", "", text), "요약")
    return ReportPage(title, published, "", "요약")


def parse_kosi(html: str) -> ReportPage:
    soup = BeautifulSoup(html, "lxml")
    fields: dict[str, str] = {}
    for th in soup.find_all("th"):
        td = th.find_next_sibling("td")
        name = " ".join(th.get_text(" ", strip=True).split())
        if td is not None and name not in fields:
            fields[name] = " ".join(td.get_text(" ", strip=True).split())
    cells = sorted(soup.find_all("td"), key=lambda t: len(t.get_text(strip=True)), reverse=True)
    text = _lines(cells[0]) if cells else ""
    return ReportPage(fields.get("제목") or None, _full_date(fields.get("발행일", "")), text, "요약")


def parse_kiet(html: str) -> ReportPage:
    soup = BeautifulSoup(html, "lxml")
    view = soup.select_one("div.page_view") or soup
    published = _full_date(view.get_text(" ", strip=True))
    box = soup.select_one("div.p_d-cont")
    text = _lines(box) if box else ""
    text = re.sub(r"^요약\s*(?:언론보도)?\s*", "", text)
    text = re.split(r"\n첨부파일\b", text)[0]               # 탭 아래 첨부파일 목록은 본문이 아니다
    if re.match(r"목차|머리말", text) or "제1장" in text[:200]:   # 요약 탭이 비고 목차만 있는 화면
        text = ""
    return ReportPage(_og_title(soup).split(" - KIET")[0].strip() or None, published, text, "요약")


PARSERS: dict[str, Callable[[str], ReportPage]] = {"kdi": parse_kdi, "kosi": parse_kosi, "kiet": parse_kiet}


class Skipped(Exception):
    """수집 대상에서 뺀 보고서 (요약 없음·기간 밖·발행일 미상). 오류가 아니라 선별 결과로 기록한다."""


def collect_report(fetcher: Fetcher, report: CuratedReport) -> DocumentRecord:
    page = fetcher.get(report.url)
    if page.status != 200:
        raise requests.HTTPError(f"상세 페이지 {page.status}")
    parsed = PARSERS[report.site](page.text)
    if parsed.published_at is None:
        raise Skipped("발행일을 읽지 못함")
    if not PERIOD[0] <= parsed.published_at <= PERIOD[1]:
        raise Skipped(f"기간 밖 ({parsed.published_at})")
    if len(parsed.text) < MIN_TEXT_CHARS:
        raise Skipped(f"상세 화면에 요약 없음 ({len(parsed.text)}자)")
    doc_id = make_doc_id(DOC_ID_PREFIX, report.url, parsed.published_at)
    raw_path = save_raw(DocType.POLICY_REPORT, doc_id, page.content, "html")
    return DocumentRecord(
        doc_id=doc_id, doc_type=DocType.POLICY_REPORT, source=INSTITUTES[report.site],
        title=parsed.title or report.title, url=report.url, published_at=parsed.published_at,
        collected_at=page.collected_at, collect_method=CollectMethod.CRAWL, raw_path=raw_path, raw_text=parsed.text,
        extra={"topic": report.topic, "reason": report.reason, "text_kind": parsed.text_kind,
               "text_chars": len(parsed.text)},
    )


@dataclass
class CollectStats:
    saved: list[DocumentRecord] = field(default_factory=list)
    already: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"저장 {len(self.saved)} · 이미 수집 {len(self.already)} · 제외 {len(self.skipped)} · 실패 {len(self.failed)}"]
        lines += [f"  제외: {item}" for item in self.skipped]
        lines += [f"  실패: {item}" for item in self.failed]
        return "\n".join(lines)


def write_review(records: Sequence[DocumentRecord], path: Path) -> None:
    """수집한 본문 확인용 (평가 문항 작성·선별 검토에 쓴다). data/raw 아래라 git 에 올리지 않는다."""
    parts = []
    for r in records:
        parts.append(f"## {r.title}\n- {r.source} · {r.published_at} · {r.extra['topic']} · "
                     f"{r.extra['text_kind']} {r.extra['text_chars']:,}자 · {r.doc_id}\n- {r.url}\n\n{r.raw_text}\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def run(dry_run: bool = False, reports: Sequence[CuratedReport] = CURATED_REPORTS, fetcher: Fetcher | None = None,
        db_path: Path | None = None, review_path: Path | None = None) -> CollectStats:
    stats = CollectStats()
    if dry_run:
        for report in reports:
            print(f"  [{report.topic}] {INSTITUTES[report.site]} · {report.title}\n      {report.reason}")
        print(f"수집 대상 {len(reports)}건 (--dry-run, 요청하지 않음)")
        return stats
    fetcher = fetcher or Fetcher()
    with open_db(db_path) as conn:
        for report in reports:
            label = f"{INSTITUTES[report.site]} · {report.title[:40]}"
            if conn.execute("SELECT 1 FROM documents WHERE url = ?", (report.url,)).fetchone():
                stats.already.append(label)
                continue
            try:
                record = collect_report(fetcher, report)
            except Skipped as exc:
                stats.skipped.append(f"{label} — {exc}")
                print(f"  제외 {label} — {exc}", flush=True)
                continue
            except (requests.RequestException, RobotsDisallowed) as exc:
                stats.failed.append(f"{label} — {describe_error(exc)}")
                print(f"  실패 {label} — {describe_error(exc)}", flush=True)
                continue
            upsert_documents(conn, [record])
            stats.saved.append(record)
            print(f"  저장 {record.published_at} {label} ({record.extra['text_kind']} {record.extra['text_chars']:,}자)",
                  flush=True)
    review_path = review_path or get_settings().raw_dir / "_probe" / "reports" / "collected.md"
    write_review(stats.saved, review_path)
    print(stats.summary())
    print(f"수집 본문 확인: {review_path}")
    return stats
