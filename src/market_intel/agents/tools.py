"""Tool functions: thin wrappers over analytics that return structured dicts (never prose).

The graph runs these, and the evaluation harness re-runs them on fresh data to get reference numbers.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from market_intel.agents.facts import flatten
from market_intel.analytics.regime import rates_snapshot, regime_summary, sector_performance
from market_intel.analytics.ticker import ticker_technicals

REGIME_KEYS = ("as_of", "regime", "score", "override", "components", "vol_regime", "trend_state", "sector_breadth")


def regime_tool(prices: pd.DataFrame) -> dict:
    summary = regime_summary(prices)
    return {k: summary[k] for k in REGIME_KEYS}


def sectors_tool(prices: pd.DataFrame) -> dict:
    """Sector returns plus rank and leader/laggard lists, computed here so the LLM never compares numbers."""
    perf = sector_performance(prices)
    ranked = {
        r["ticker"]: {**r, "rank_21d": i + 1}
        for i, r in enumerate(perf["sectors"])
        if r["ret_21d_pct"] is not None
    }
    tickers = list(ranked)
    return {
        "as_of": perf["as_of"],
        "leaders": tickers[:3],
        "laggards": tickers[-3:],
        **{t: {k: v for k, v in r.items() if k != "ticker"} for t, r in ranked.items()},
    }


def rates_tool(prices: pd.DataFrame) -> dict:
    return rates_snapshot(prices)


def drilldown_tool(prices: pd.DataFrame, tickers: list[str]) -> tuple[dict, list[str]]:
    """Technicals per ticker. Returns (results, log lines for tickers that were skipped)."""
    out, skipped = {}, []
    for ticker in tickers:
        try:
            out[ticker] = ticker_technicals(prices, ticker)
        except ValueError as exc:
            skipped.append(f"drilldown {ticker}: skipped ({exc})")
    return out, skipped


def assemble_facts(regime: dict | None, sectors: dict | None, rates: dict | None, drilldown: dict | None) -> dict[str, Any]:
    """Flatten tool outputs into {dotted.path: value}: the only numbers the writer may use."""
    facts: dict[str, Any] = {}
    for name, data in (("regime", regime), ("sectors", sectors), ("rates", rates)):
        if data:
            facts.update(flatten(name, data))
    for ticker, data in (drilldown or {}).items():
        facts.update(flatten(f"drilldown.{ticker}", data))
    return facts
