"""Read DuckDB into a wide DataFrame (date index, one column per ticker).

The loader takes `as_of` so callers can replay history without look-ahead.
"""
from __future__ import annotations

from datetime import date

import duckdb
import pandas as pd

_PRICE_FIELDS = {"open", "high", "low", "close", "adj_close", "volume"}


def load_price_matrix(
    con: duckdb.DuckDBPyConnection,
    tickers: list[str],
    as_of: date | str | None = None,
    field: str = "adj_close",
) -> pd.DataFrame:
    if field not in _PRICE_FIELDS:  # field is interpolated into SQL, so allow-list it
        raise ValueError(f"Unknown price field: {field!r}")
    df = con.execute(
        f"""
        SELECT date, ticker, {field} AS value
        FROM prices
        WHERE list_contains(?, ticker)
          AND (CAST(? AS DATE) IS NULL OR date <= CAST(? AS DATE))
        """,
        [tickers, as_of, as_of],
    ).df()
    if df.empty:
        return pd.DataFrame()
    wide = df.pivot(index="date", columns="ticker", values="value").sort_index()
    wide.index = pd.to_datetime(wide.index)
    return wide
