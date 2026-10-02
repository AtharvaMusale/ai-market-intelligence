"""Most recent stored documents per ticker (deterministic, point-in-time, no Pinecone needed)."""
from __future__ import annotations

import duckdb

from market_intel.quality import is_junk_title, trim_to_first_item


def recent_documents(
    con: duckdb.DuckDBPyConnection,
    tickers: list[str],
    as_of: str | None = None,
    filings_per_ticker: int = 2,
    news_per_ticker: int = 2,
    excerpt_chars: int = 400,
) -> list[dict]:
    """Latest filings and non-junk news per ticker, published on or before `as_of` (no look-ahead)."""
    docs = []
    for ticker in tickers:
        for is_news, limit in ((False, filings_per_ticker), (True, news_per_ticker)):
            rows = con.execute(
                """
                SELECT doc_id, ticker, doc_type, title, url, CAST(published_at AS DATE), substr(text, 1, 3000)
                FROM documents
                WHERE ticker = ? AND (doc_type = 'news') = ?
                  AND (CAST(? AS DATE) IS NULL OR published_at < CAST(? AS DATE) + INTERVAL 1 DAY)
                ORDER BY published_at DESC
                LIMIT ?
                """,
                [ticker, is_news, as_of, as_of, limit * 5],  # over-fetch: junk is filtered below
            ).fetchall()
            kept = 0
            for doc_id, tick, doc_type, title, url, day, text in rows:
                if kept == limit:
                    break
                if is_news and is_junk_title(title):
                    continue
                if doc_type == "8-K":
                    text = trim_to_first_item(text or "")
                docs.append({"doc_id": doc_id, "ticker": tick, "doc_type": doc_type, "title": title,
                             "url": url, "date": day.isoformat(), "excerpt": (text or "")[:excerpt_chars]})
                kept += 1
    return docs
