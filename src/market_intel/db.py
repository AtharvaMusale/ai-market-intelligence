"""DuckDB connection, schema, and the idempotent upsert helper. Leaf module."""
from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import pandas as pd

from market_intel.config import get_duckdb_path

logger = logging.getLogger(__name__)

# Allow-list: table names cannot be bound as SQL parameters, so we only accept these.
TABLES: dict[str, dict[str, list[str]]] = {
    "prices": {
        "columns": ["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"],
        "pk": ["ticker", "date"],
    },
    "documents": {
        "columns": ["doc_id", "ticker", "source", "doc_type", "title", "url", "published_at", "fetched_at", "text"],
        "pk": ["doc_id"],
    },
    "chunks": {
        "columns": ["chunk_id", "doc_id", "chunk_idx", "text", "content_hash", "pushed_at"],
        "pk": ["chunk_id"],
    },
}

_DDL = [
    """
    CREATE TABLE IF NOT EXISTS prices (
        ticker VARCHAR NOT NULL,
        date DATE NOT NULL,
        open DOUBLE,
        high DOUBLE,
        low DOUBLE,
        close DOUBLE,
        adj_close DOUBLE,
        volume BIGINT,
        PRIMARY KEY (ticker, date)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS documents (
        doc_id VARCHAR PRIMARY KEY,
        ticker VARCHAR NOT NULL,
        source VARCHAR NOT NULL,
        doc_type VARCHAR NOT NULL,
        title VARCHAR,
        url VARCHAR NOT NULL,
        published_at TIMESTAMP NOT NULL,
        fetched_at TIMESTAMP NOT NULL,
        text VARCHAR
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id VARCHAR PRIMARY KEY,
        doc_id VARCHAR NOT NULL,
        chunk_idx INTEGER NOT NULL,
        text VARCHAR NOT NULL,
        content_hash VARCHAR NOT NULL,
        pushed_at TIMESTAMP
    )
    """,
]


def connect(path: str | Path | None = None, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open DuckDB (use ":memory:" for tests) and make sure the tables exist.

    read_only=True never writes, and lets several readers (scripts, the app) share the file.
    """
    target = get_duckdb_path() if path is None else path
    if read_only:
        return duckdb.connect(str(target), read_only=True)
    if str(target) != ":memory:":
        Path(target).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(target))
    init_schema(con)
    return con


def init_schema(con: duckdb.DuckDBPyConnection) -> None:
    for statement in _DDL:
        con.execute(statement)


def upsert_df(con: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame) -> int:
    """Insert rows, replacing any with the same primary key. Safe to run repeatedly."""
    if table not in TABLES:
        raise ValueError(f"Unknown table: {table!r}")
    columns = TABLES[table]["columns"]
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing columns for {table}: {sorted(missing)}")
    if df.empty:
        return 0

    data = df[columns].copy()
    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"])
    # Duplicate keys inside one INSERT can fail, so keep the last occurrence first.
    data = data.drop_duplicates(subset=TABLES[table]["pk"], keep="last")

    column_sql = ", ".join(f'"{c}"' for c in columns)
    con.register("incoming_rows", data)
    try:
        con.execute(
            f"INSERT OR REPLACE INTO {table} ({column_sql}) SELECT {column_sql} FROM incoming_rows"
        )
    finally:
        con.unregister("incoming_rows")
    logger.info("Upserted %d rows into %s", len(data), table)
    return len(data)
