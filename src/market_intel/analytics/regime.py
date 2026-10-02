"""Market regime analytics: pure functions, DataFrames in, JSON-serializable dicts out.

`prices` is a wide adj_close DataFrame (date index, ticker columns), including the Yahoo
yield tickers. Windows are counted in trading days (rows), not calendar days. No LLM, no network.
"""
from __future__ import annotations

import pandas as pd

from market_intel.config import RATES, SECTOR_ETFS

# Heuristic thresholds: transparent and deliberately simple, not tuned on data.
VIX_LOW = 15.0
VIX_NORMAL = 20.0
VIX_ELEVATED = 30.0
BREADTH_BULLISH = 60.0  # % of sector ETFs above 50-day average
BREADTH_BEARISH = 40.0
RISK_ON_SCORE = 2
RISK_OFF_SCORE = -2
TRADING_DAYS_YEAR = 252
VOL_SCORE = {"low": 1, "normal": 0, "elevated": -1, "high": -2}


def _num(value: float | None, digits: int = 4) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def _date_str(prices: pd.DataFrame, tickers: list[str]) -> str | None:
    """Latest date with data for the given tickers (not the whole frame, since Yahoo series lag differently)."""
    cols = [t for t in tickers if t in prices.columns]
    if not cols:
        return None
    index = prices[cols].dropna(how="all").index
    return None if len(index) == 0 else pd.Timestamp(index.max()).date().isoformat()


def _return_pct(series: pd.Series, n: int) -> float | None:
    s = series.dropna()
    if len(s) <= n:
        return None
    return _num((s.iloc[-1] / s.iloc[-1 - n] - 1) * 100)


def _require(prices: pd.DataFrame, ticker: str, min_obs: int) -> pd.Series:
    if ticker not in prices.columns:
        raise ValueError(f"{ticker} not in price data")
    s = prices[ticker].dropna()
    if len(s) < min_obs:
        raise ValueError(f"{ticker} needs at least {min_obs} observations, has {len(s)}")
    return s


def sector_performance(prices: pd.DataFrame) -> dict:
    """1/5/21 trading-day returns (%) for each sector ETF, sorted by 21d return, best first."""
    rows = []
    for ticker, name in SECTOR_ETFS.items():
        if ticker not in prices.columns:
            continue
        s = prices[ticker]
        rows.append(
            {
                "ticker": ticker,
                "name": name,
                "ret_1d_pct": _return_pct(s, 1),
                "ret_5d_pct": _return_pct(s, 5),
                "ret_21d_pct": _return_pct(s, 21),
            }
        )
    rows.sort(key=lambda r: (r["ret_21d_pct"] is None, -(r["ret_21d_pct"] or 0)))
    return {"as_of": _date_str(prices, list(SECTOR_ETFS)), "sectors": rows}


def sector_breadth(prices: pd.DataFrame, window: int = 50) -> dict:
    """Share of sector ETFs trading above their `window`-day average (a proxy for market breadth)."""
    above, below = [], []
    for ticker in SECTOR_ETFS:
        if ticker not in prices.columns:
            continue
        s = prices[ticker].dropna()
        if len(s) < window:
            continue
        (above if s.iloc[-1] > s.tail(window).mean() else below).append(ticker)
    total = len(above) + len(below)
    return {
        "as_of": _date_str(prices, list(SECTOR_ETFS)),
        "window_days": window,
        "n_above": len(above),
        "n_total": total,
        "pct_above": _num(100 * len(above) / total, 2) if total else None,
        "above": above,
        "below": below,
    }


def vol_label(vix_level: float) -> str:
    if vix_level < VIX_LOW:
        return "low"
    if vix_level < VIX_NORMAL:
        return "normal"
    if vix_level < VIX_ELEVATED:
        return "elevated"
    return "high"


def vol_regime(prices: pd.DataFrame) -> dict:
    """VIX level and label, its 1-year percentile, and SPY 21-day realized vol (annualized, %)."""
    vix = _require(prices, "^VIX", 1)
    spy = _require(prices, "SPY", 22)
    level = float(vix.iloc[-1])
    window = vix.tail(TRADING_DAYS_YEAR)
    realized = spy.pct_change().tail(21).std(ddof=1) * (TRADING_DAYS_YEAR**0.5) * 100
    return {
        "as_of": _date_str(prices, ["^VIX"]),
        "vix_level": _num(level, 2),
        "label": vol_label(level),
        "vix_percentile_1y": _num((window <= level).mean() * 100, 1),
        "spy_realized_vol_21d_pct": _num(realized, 2),
    }


def trend_state(prices: pd.DataFrame) -> dict:
    """SPY vs its 50/200-day averages, plus drawdown from the 52-week high."""
    spy = _require(prices, "SPY", 200)
    last = float(spy.iloc[-1])
    sma50 = float(spy.tail(50).mean())
    sma200 = float(spy.tail(200).mean())
    if last > sma50 > sma200:
        state = "uptrend"
    elif last < sma50 < sma200:
        state = "downtrend"
    else:
        state = "mixed"
    high_52w = float(spy.tail(TRADING_DAYS_YEAR).max())
    return {
        "as_of": _date_str(prices, ["SPY"]),
        "state": state,
        "spy_last": _num(last, 2),
        "sma_50": _num(sma50, 2),
        "sma_200": _num(sma200, 2),
        "drawdown_from_52w_high_pct": _num((last / high_52w - 1) * 100, 2),
    }


def rates_snapshot(prices: pd.DataFrame) -> dict:
    """Latest Treasury yield (%) and 5-trading-day change (percentage points) per Yahoo yield ticker.

    `curve_10y_3m_proxy_pp` is 10-year minus 13-week. It is NOT the standard 10Y-2Y spread.
    """
    series_out = {}
    for ticker, name in RATES.items():
        if ticker not in prices.columns:
            continue
        s = prices[ticker].dropna()
        if s.empty:
            continue
        series_out[ticker] = {
            "name": name,
            "latest_pct": _num(s.iloc[-1]),
            "obs_date": pd.Timestamp(s.index[-1]).date().isoformat(),
            "change_5d_pp": _num(s.iloc[-1] - s.iloc[-6]) if len(s) > 5 else None,
        }
    ten_y = series_out.get("^TNX", {}).get("latest_pct")
    three_m = series_out.get("^IRX", {}).get("latest_pct")
    curve = None if ten_y is None or three_m is None else _num(ten_y - three_m)
    return {
        "as_of": _date_str(prices, list(RATES)),
        "series": series_out,
        "curve_10y_3m_proxy_pp": curve,
        "curve_10y_3m_proxy_inverted": None if curve is None else curve < 0,
    }


def regime_summary(prices: pd.DataFrame) -> dict:
    """Combine everything into risk_on / neutral / risk_off using a transparent point score.

    Score = trend (+1 up / -1 down) + volatility (low +1, normal 0, elevated -1, high -2)
          + breadth (+1 if >=60% above 50d, -1 if <=40%).
    risk_on if score >= 2, risk_off if score <= -2, else neutral.
    Override: a "high" VIX label is always risk_off.
    """
    perf = sector_performance(prices)
    breadth = sector_breadth(prices)
    vol = vol_regime(prices)
    trend = trend_state(prices)
    rates = rates_snapshot(prices)

    trend_pts = {"uptrend": 1, "downtrend": -1, "mixed": 0}[trend["state"]]
    vol_pts = VOL_SCORE[vol["label"]]
    pct_above = breadth["pct_above"]
    if pct_above is None:
        breadth_pts = 0
    elif pct_above >= BREADTH_BULLISH:
        breadth_pts = 1
    elif pct_above <= BREADTH_BEARISH:
        breadth_pts = -1
    else:
        breadth_pts = 0

    score = trend_pts + vol_pts + breadth_pts
    override = None
    if vol["label"] == "high":
        label, override = "risk_off", "VIX label is high"
    elif score >= RISK_ON_SCORE:
        label = "risk_on"
    elif score <= RISK_OFF_SCORE:
        label = "risk_off"
    else:
        label = "neutral"

    return {
        "as_of": _date_str(prices, ["SPY"]),  # equity close date; each component carries its own as_of
        "regime": label,
        "score": score,
        "override": override,
        "components": {"trend": trend_pts, "volatility": vol_pts, "breadth": breadth_pts},
        "sector_performance": perf,
        "sector_breadth": breadth,
        "vol_regime": vol,
        "trend_state": trend,
        "rates_snapshot": rates,
    }
