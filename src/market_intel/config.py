"""Central configuration: tickers and paths. Leaf module (imports nothing from the project)."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

SECTOR_ETFS: dict[str, str] = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Health Care",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
}

BENCHMARKS: dict[str, str] = {
    "SPY": "S&P 500",
    "QQQ": "Nasdaq 100",
    "IWM": "Russell 2000",
}

CROSS_ASSET: dict[str, str] = {
    "^VIX": "CBOE Volatility Index",
    "TLT": "20+ Year Treasury Bonds",
    "GLD": "Gold",
    "UUP": "US Dollar Index Fund",
}

# Treasury yield indexes quoted by Yahoo. Their "price" is the yield in percent.
RATES: dict[str, str] = {
    "^IRX": "13-Week Treasury Yield",
    "^FVX": "5-Year Treasury Yield",
    "^TNX": "10-Year Treasury Yield",
    "^TYX": "30-Year Treasury Yield",
}

# Small watchlist for text ingestion (keeps Pinecone free-tier usage low). ticker -> news search keyword.
WATCHLIST: dict[str, str] = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "Nvidia",
    "JPM": "JPMorgan",
    "GS": "Goldman Sachs",
    "XOM": "Exxon",
}

SEC_FORMS = ("8-K", "10-Q", "10-K")
FILINGS_LOOKBACK_DAYS = 90
NEWS_TIMESPAN = "7d"  # GDELT's DOC API covers a rolling recent window
EMBED_TOKEN_BUDGET_PER_RUN = 500_000  # abort a run that would embed more (Pinecone free tier: 5M tokens/month)

ALL_PRICE_TICKERS: list[str] = [*SECTOR_ETFS, *BENCHMARKS, *CROSS_ASSET, *RATES, *WATCHLIST]

# LLM: Haiku 4.5 is the only model. Prices are USD per million tokens (Anthropic pricing, checked 2026-10-02).
LLM_MODEL = "claude-haiku-4-5-20251001"
LLM_INPUT_USD_PER_MTOK = 1.0
LLM_OUTPUT_USD_PER_MTOK = 5.0
LLM_CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "llm"
BRIEFING_MAX_TOKENS = 3000
QA_MAX_TOKENS = 700

# Evaluation thresholds. Chosen up front, before any real run; do not tune them to make a run pass.
EVAL_MIN_NUMERIC_ACCURACY = float(os.environ.get("EVAL_MIN_NUMERIC_ACCURACY", "0.95"))
EVAL_ABS_TOL = 0.01  # a number in a claim may differ from the source value by this much (rounding)
EVAL_REL_TOL = 0.001  # or by this fraction of the source value, whichever is larger

HISTORY_YEARS = 3  # first-time download depth; enough for 200-day averages and 1y percentiles
OVERLAP_DAYS = 7  # incremental runs re-fetch this many days to pick up Yahoo's late revisions


def get_sec_user_agent() -> str:
    """SEC EDGAR requires a declared User-Agent with contact info, e.g. "Jane Doe jane@example.com"."""
    value = os.environ.get("SEC_USER_AGENT", "").strip()
    if "@" not in value:
        raise RuntimeError("Set SEC_USER_AGENT in .env to 'Your Name your@email.com' (required by SEC EDGAR).")
    return value


def get_pinecone_settings() -> tuple[str, str]:
    """(api_key, index_name) from the environment."""
    key = os.environ.get("PINECONE_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Set PINECONE_API_KEY in .env.")
    return key, os.environ.get("PINECONE_INDEX_NAME", "market-intel")


def get_anthropic_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Set ANTHROPIC_API_KEY in .env, or use --dry-run for the mock writer.")
    return key


def get_duckdb_path() -> Path | str:
    """DuckDB location from DUCKDB_PATH (relative paths resolve against the project root)."""
    raw = os.environ.get("DUCKDB_PATH", "data/market_intel.duckdb")
    if raw == ":memory:":
        return raw
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path
