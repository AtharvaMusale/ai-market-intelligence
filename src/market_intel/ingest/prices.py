"""Daily price ingestion from Yahoo Finance (unofficial data; see README limitations)."""
from __future__ import annotations

import logging
from datetime import date, timedelta

import duckdb
import pandas as pd
import yfinance as yf

from market_intel.config import ALL_PRICE_TICKERS, HISTORY_YEARS, OVERLAP_DAYS
from market_intel.db import upsert_df

logger = logging.getLogger(__name__)

_COLUMNS = ["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]


def normalize_download(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Turn yfinance's wide download into long rows (one per ticker/date), warning on empty tickers."""
    if not raw.empty and not isinstance(raw.columns, pd.MultiIndex) and len(tickers) == 1:
        raw = pd.concat({tickers[0]: raw}, axis=1)

    frames = []
    for ticker in tickers:
        if raw.empty or ticker not in raw.columns.get_level_values(0):
            logger.warning("No price data returned for %s", ticker)
            continue
        sub = raw[ticker].dropna(how="all")
        sub.columns = [str(c).lower().replace(" ", "_") for c in sub.columns]
        sub = sub.dropna(subset=["adj_close"])
        if sub.empty:
            logger.warning("No price data returned for %s", ticker)
            continue
        sub = sub.rename_axis("date").reset_index()
        sub["ticker"] = ticker
        frames.append(sub[_COLUMNS])
    if not frames:
        return pd.DataFrame(columns=_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def fetch_prices(tickers: list[str], start: date) -> pd.DataFrame:
    # auto_adjust=False keeps both close and adj_close; returns must use adj_close.
    raw = yf.download(
        tickers,
        start=start.isoformat(),
        auto_adjust=False,
        group_by="ticker",
        progress=False,
        threads=False,  # threaded downloads hit a yfinance cache lock ("database is locked")
    )
    return normalize_download(raw, tickers)


def ingest_prices(
    con: duckdb.DuckDBPyConnection,
    tickers: list[str] | None = None,
    today: date | None = None,
) -> int:
    """Download new prices incrementally: from (last stored date - OVERLAP_DAYS), or full history if new."""
    tickers = tickers or ALL_PRICE_TICKERS
    today = today or date.today()
    last_dates = dict(con.execute("SELECT ticker, max(date) FROM prices GROUP BY ticker").fetchall())

    full_start = today - timedelta(days=365 * HISTORY_YEARS)
    starts = [
        last_dates[t] - timedelta(days=OVERLAP_DAYS) if t in last_dates else full_start
        for t in tickers
    ]
    start = min(starts)
    logger.info("Fetching %d tickers from %s", len(tickers), start)
    return upsert_df(con, "prices", fetch_prices(tickers, start))
