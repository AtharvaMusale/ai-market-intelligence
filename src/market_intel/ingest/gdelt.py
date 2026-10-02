"""News headlines from the GDELT DOC 2.0 API.

Data from the GDELT Project (https://www.gdeltproject.org/): cite it wherever news is shown.
We store only title, URL, publisher domain and date, never article text. GDELT asks for at most
one request every 5 seconds.
"""
from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime, timezone

import duckdb
import pandas as pd
import requests

from market_intel.config import NEWS_TIMESPAN, WATCHLIST
from market_intel.db import upsert_df
from market_intel.ingest.http import RateLimitedSession
from market_intel.quality import is_junk_title

logger = logging.getLogger(__name__)

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
MIN_INTERVAL_S = 10.0  # GDELT says one request per 5 seconds, but 6s still drew 429s in practice
RETRIES = 3
RETRY_WAIT_S = 30.0
DOC_COLUMNS = ["doc_id", "ticker", "source", "doc_type", "title", "url", "published_at", "fetched_at", "text"]


def parse_articles(payload: dict, ticker: str, keyword: str, fetched_at: datetime) -> pd.DataFrame:
    """Turn a GDELT artlist payload into document rows, keeping only titles that mention the keyword."""
    rows = []
    for article in payload.get("articles", []):
        title = (article.get("title") or "").strip()
        url = article.get("url")
        seen = article.get("seendate")  # e.g. 20260930T123000Z
        if not title or not url or not seen or keyword.lower() not in title.lower() or is_junk_title(title):
            continue
        rows.append(
            {
                "doc_id": "news-" + hashlib.sha1(f"{ticker}|{url}".encode()).hexdigest()[:16],
                "ticker": ticker,
                "source": f"gdelt:{article.get('domain', 'unknown')}",
                "doc_type": "news",
                "title": title,
                "url": url,
                "published_at": datetime.strptime(seen, "%Y%m%dT%H%M%SZ"),
                "fetched_at": fetched_at,
                "text": title,
            }
        )
    return pd.DataFrame(rows, columns=DOC_COLUMNS)


def _get_with_retry(session: RateLimitedSession, params: dict, sleep=time.sleep) -> requests.Response:
    """GET with a long pause and retry when GDELT answers 429 or drops the connection."""
    for attempt in range(1, RETRIES + 1):
        try:
            return session.get(GDELT_URL, params=params)
        except requests.RequestException as exc:
            if attempt == RETRIES:
                raise
            logger.warning("GDELT attempt %d/%d failed (%s); waiting %ss", attempt, RETRIES, exc, RETRY_WAIT_S)
            sleep(RETRY_WAIT_S)
    raise AssertionError("unreachable")


def fetch_payload(session: RateLimitedSession, keywords: list[str], sleep=time.sleep) -> dict:
    """One combined request for all keywords (GDELT throttles hard, so fewer requests is safer)."""
    quoted = " OR ".join(f'"{k}"' for k in keywords)
    params = {
        "query": f"({quoted}) sourcelang:english",
        "mode": "artlist",
        "format": "json",
        "maxrecords": 250,  # GDELT's maximum
        "timespan": NEWS_TIMESPAN,
        "sort": "datedesc",
    }
    response = _get_with_retry(session, params, sleep)
    try:
        return response.json()
    except ValueError:  # GDELT answers rate-limit and query errors with plain text
        logger.warning("GDELT returned non-JSON: %s", response.text[:120])
        return {}


def ingest_news(
    con: duckdb.DuckDBPyConnection,
    watchlist: dict[str, str] | None = None,
    session: RateLimitedSession | None = None,
) -> int:
    """Fetch headlines for the whole watchlist in one request; store only documents we do not already have."""
    watchlist = watchlist or WATCHLIST
    session = session or RateLimitedSession(MIN_INTERVAL_S, {"User-Agent": "ai-market-intelligence-portfolio"})
    fetched_at = datetime.now(timezone.utc).replace(tzinfo=None)
    try:
        payload = fetch_payload(session, list(watchlist.values()))
    except requests.RequestException as exc:
        logger.warning("GDELT request failed: %s. Try again later; filings are unaffected.", exc)
        return 0
    existing = {r[0] for r in con.execute("SELECT doc_id FROM documents").fetchall()}
    total = 0
    for ticker, keyword in watchlist.items():
        df = parse_articles(payload, ticker, keyword, fetched_at)
        total += upsert_df(con, "documents", df[~df["doc_id"].isin(existing)])
    return total
