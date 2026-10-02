"""Single-ticker technicals for the drill-down tool. Pure functions, no LLM, no network."""
from __future__ import annotations

import pandas as pd

from market_intel.analytics.regime import TRADING_DAYS_YEAR, _date_str, _num, _require, _return_pct


def _rsi(series: pd.Series, period: int = 14) -> float | None:
    """Wilder's RSI (0-100)."""
    delta = series.diff().dropna()
    if len(delta) < period:
        return None
    avg_gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
    avg_loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
    return 100.0 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss)


def ticker_technicals(prices: pd.DataFrame, ticker: str) -> dict:
    """Returns, moving averages, RSI, drawdown, realized vol, and 21d performance relative to SPY."""
    s = _require(prices, ticker, 60)
    last = float(s.iloc[-1])
    sma50 = float(s.tail(50).mean())
    sma200 = float(s.tail(200).mean()) if len(s) >= 200 else None
    ret_21d = _return_pct(s, 21)
    spy_21d = _return_pct(prices["SPY"], 21) if "SPY" in prices.columns else None
    relative = None if ret_21d is None or spy_21d is None else ret_21d - spy_21d
    return {
        "as_of": _date_str(prices, [ticker]),
        "last": _num(last, 2),
        "ret_1d_pct": _return_pct(s, 1),
        "ret_5d_pct": _return_pct(s, 5),
        "ret_21d_pct": ret_21d,
        "ret_21d_vs_spy_pct": _num(relative),
        "sma_50": _num(sma50, 2),
        "sma_200": _num(sma200, 2),
        "pct_vs_sma_50": _num((last / sma50 - 1) * 100, 2),
        "rsi_14": _num(_rsi(s), 1),
        "drawdown_from_52w_high_pct": _num((last / float(s.tail(TRADING_DAYS_YEAR).max()) - 1) * 100, 2),
        "realized_vol_21d_pct": _num(s.pct_change().tail(21).std(ddof=1) * (TRADING_DAYS_YEAR**0.5) * 100, 2),
    }
