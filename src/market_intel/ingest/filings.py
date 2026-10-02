"""SEC EDGAR filings (8-K, 10-Q, 10-K) for the watchlist.

EDGAR fair-access rules: declare a User-Agent with contact info and stay at or below
10 requests/second. We use ~7/second at most.
"""
from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta, timezone
from html.parser import HTMLParser

import duckdb
import pandas as pd
import requests

from market_intel.config import FILINGS_LOOKBACK_DAYS, SEC_FORMS, WATCHLIST, get_sec_user_agent
from market_intel.db import upsert_df
from market_intel.ingest.http import RateLimitedSession
from market_intel.quality import trim_to_first_item

logger = logging.getLogger(__name__)

MIN_INTERVAL_S = 0.15
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"
MAX_CHARS = {"8-K": 30_000, "10-Q": 40_000, "10-K": 40_000}
DOC_COLUMNS = ["doc_id", "ticker", "source", "doc_type", "title", "url", "published_at", "fetched_at", "text"]


class _TextExtractor(HTMLParser):
    """Collect visible text, skipping scripts, styles, and inline-XBRL header blocks."""

    _SKIP = {"script", "style", "ix:header"}

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data):
        if not self._skip_depth:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    return re.sub(r"\s+", " ", " ".join(parser.parts)).strip()


def extract_mdna(text: str, max_chars: int) -> str:
    """Heuristic: start at the last "Management's Discussion and Analysis" heading.

    The first occurrences are usually the table of contents, so the last one is more likely the real section.
    Falls back to the start of the document when no heading is found.
    """
    matches = list(re.finditer(r"management.s discussion and analysis", text, flags=re.IGNORECASE))
    start = matches[-1].start() if matches else 0
    return text[start : start + max_chars]


def parse_submissions(payload: dict, cik: int, ticker: str, since: date) -> list[dict]:
    """Pick recent 8-K/10-Q/10-K filings from an EDGAR submissions payload."""
    recent = payload["filings"]["recent"]
    filings = []
    for i, form in enumerate(recent["form"]):
        filed = date.fromisoformat(recent["filingDate"][i])
        if form not in SEC_FORMS or filed < since:
            continue
        accession = recent["accessionNumber"][i]
        items = recent.get("items", [""] * len(recent["form"]))[i]
        filings.append(
            {
                "doc_id": f"sec-{accession}",
                "ticker": ticker,
                "form": form,
                "filed": filed,
                "title": f"{ticker} {form} filed {filed.isoformat()}" + (f" (items {items})" if items else ""),
                "url": ARCHIVE_URL.format(
                    cik=cik, accession=accession.replace("-", ""), document=recent["primaryDocument"][i]
                ),
            }
        )
    return filings


def load_cik_map(session: RateLimitedSession) -> dict[str, int]:
    data = session.get(TICKERS_URL).json()
    return {row["ticker"]: int(row["cik_str"]) for row in data.values()}


def ingest_filings(
    con: duckdb.DuckDBPyConnection,
    watchlist: dict[str, str] | None = None,
    lookback_days: int = FILINGS_LOOKBACK_DAYS,
    session: RateLimitedSession | None = None,
) -> int:
    """Download new filings for each watchlist ticker. Already-stored filings are skipped (they never change)."""
    session = session or RateLimitedSession(MIN_INTERVAL_S, {"User-Agent": get_sec_user_agent()})
    since = date.today() - timedelta(days=lookback_days)
    fetched_at = datetime.now(timezone.utc).replace(tzinfo=None)
    existing = {r[0] for r in con.execute("SELECT doc_id FROM documents").fetchall()}
    try:
        cik_map = load_cik_map(session)
    except requests.RequestException as exc:
        logger.warning("Could not load the SEC ticker map: %s", exc)
        return 0

    total = 0
    for ticker in watchlist or WATCHLIST:
        cik = cik_map.get(ticker)
        if cik is None:
            logger.warning("No CIK found for %s", ticker)
            continue
        try:
            payload = session.get(SUBMISSIONS_URL.format(cik=cik)).json()
            rows = []
            for filing in parse_submissions(payload, cik, ticker, since):
                if filing["doc_id"] in existing:
                    continue
                text = html_to_text(session.get(filing["url"]).text)
                if filing["form"] != "8-K":
                    text = extract_mdna(text, MAX_CHARS[filing["form"]])
                else:
                    text = trim_to_first_item(text)[: MAX_CHARS["8-K"]]
                rows.append(
                    {
                        "doc_id": filing["doc_id"],
                        "ticker": ticker,
                        "source": "sec-edgar",
                        "doc_type": filing["form"],
                        "title": filing["title"],
                        "url": filing["url"],
                        "published_at": datetime.combine(filing["filed"], datetime.min.time()),
                        "fetched_at": fetched_at,
                        "text": text,
                    }
                )
        except requests.RequestException as exc:
            logger.warning("EDGAR request failed for %s: %s", ticker, exc)
            continue
        total += upsert_df(con, "documents", pd.DataFrame(rows, columns=DOC_COLUMNS))
    return total
