"""Most recent stored documents per ticker (deterministic, point-in-time, no Pinecone needed)."""
from __future__ import annotations

import duckdb


def recent_documents(
    con: duckdb.DuckDBPyConnection,
    tickers: list[str],
    as_of: str | None = None,
    filings_per_ticker: int = 2,
    news_per_ticker: int = 2,
    excerpt_chars: int = 400,
) -> list[dict]:
    """Latest filings and news per ticker, published on or before `as_of` (no look-ahead)."""
    docs = []
    for ticker in tickers:
        for is_news, limit in ((False, filings_per_ticker), (True, news_per_ticker)):
            rows = con.execute(
                f"""
                SELECT doc_id, ticker, doc_type, title, url, CAST(published_at AS DATE), substr(text, 1, ?)
                FROM documents
                WHERE ticker = ? AND (doc_type = 'news') = ?
                  AND (CAST(? AS DATE) IS NULL OR published_at < CAST(? AS DATE) + INTERVAL 1 DAY)
                ORDER BY published_at DESC
                LIMIT ?
                """,
                [excerpt_chars, ticker, is_news, as_of, as_of, limit],
            ).fetchall()
            docs.extend(
                {"doc_id": r[0], "ticker": r[1], "doc_type": r[2], "title": r[3], "url": r[4],
                 "date": r[5].isoformat(), "excerpt": r[6] or ""}
                for r in rows
            )
    return docs
