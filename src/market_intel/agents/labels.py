"""Plain-English names for tools, sections and fact paths. Display only: the raw paths stay the real citations."""
from __future__ import annotations

TOOL_NAMES = {
    "plan": "Plan", "regime": "Market regime", "sectors": "Sector performance", "rates": "Treasury yields",
    "drilldown": "Stock drill-down", "documents": "Filings and news", "write": "Writer", "validate": "Citation check",
}

SECTION_NAMES = {
    "market_regime": "Market regime", "sector_leaders_laggards": "Sector leaders and laggards",
    "rates_macro": "Rates", "notable_events": "Notable events", "watchlist_notes": "Watchlist notes", "risks": "Risks",
}

REGIME_FIELDS = {
    "as_of": "Data date", "regime": "Market regime", "score": "Regime score", "override": "Risk-off override",
    "components.trend": "Trend points", "components.volatility": "Volatility points", "components.breadth": "Breadth points",
    "vol_regime.as_of": "VIX data date", "vol_regime.vix_level": "VIX level", "vol_regime.label": "Volatility level",
    "vol_regime.vix_percentile_1y": "VIX 1-year percentile", "vol_regime.spy_realized_vol_21d_pct": "SPY 21-day realized volatility (%)",
    "trend_state.as_of": "SPY data date", "trend_state.state": "SPY trend", "trend_state.spy_last": "SPY close",
    "trend_state.sma_50": "SPY 50-day average", "trend_state.sma_200": "SPY 200-day average",
    "trend_state.drawdown_from_52w_high_pct": "SPY drop from 52-week high (%)",
    "sector_breadth.as_of": "Breadth data date", "sector_breadth.window_days": "Averaging window (days)",
    "sector_breadth.n_above": "Sectors above 50-day average", "sector_breadth.n_total": "Sectors tracked",
    "sector_breadth.pct_above": "Share of sectors above 50-day average (%)", "sector_breadth.above": "Sectors above 50-day average",
    "sector_breadth.below": "Sectors below 50-day average",
}
SECTOR_FIELDS = {
    "name": "name", "ret_1d_pct": "1-day return (%)", "ret_5d_pct": "5-day return (%)", "ret_21d_pct": "21-day return (%)",
    "rank_21d": "21-day rank",
}
SECTOR_TOP = {"as_of": "Sector data date", "leaders": "Top sectors (21-day)", "laggards": "Weakest sectors (21-day)"}
RATE_NAMES = {"^IRX": "13-week yield", "^FVX": "5-year yield", "^TNX": "10-year yield", "^TYX": "30-year yield"}
RATE_FIELDS = {"latest_pct": "(%)", "change_5d_pp": "5-day change (pp)", "obs_date": "data date", "name": "name"}
RATE_TOP = {"as_of": "Yield data date", "curve_10y_3m_proxy_pp": "10-year minus 13-week spread (pp, proxy)",
            "curve_10y_3m_proxy_inverted": "Curve inverted (10-year vs 13-week)"}
DRILL_FIELDS = {
    "as_of": "data date", "verdict": "verdict", "trend": "trend", "rsi_zone": "RSI zone", "vs_spy_21d": "versus SPY (21-day)",
    "last": "last close", "ret_1d_pct": "1-day return (%)", "ret_5d_pct": "5-day return (%)", "ret_21d_pct": "21-day return (%)",
    "ret_21d_vs_spy_pct": "21-day return minus SPY (pp)", "sma_50": "50-day average", "sma_200": "200-day average",
    "pct_vs_sma_50": "distance from 50-day average (%)", "rsi_14": "RSI", "drawdown_from_52w_high_pct": "drop from 52-week high (%)",
    "realized_vol_21d_pct": "21-day realized volatility (%)",
}


def friendly_label(path: str) -> str:
    """'drilldown.AAPL.rsi_14' -> 'AAPL RSI'. Unknown paths come back unchanged."""
    parts = path.split(".")
    head, rest = parts[0], parts[1:]
    if head == "regime":
        return REGIME_FIELDS.get(".".join(rest), path)
    if head == "sectors":
        if len(rest) == 1:
            return SECTOR_TOP.get(rest[0], path)
        if len(rest) == 2 and rest[1] in SECTOR_FIELDS:
            return f"{rest[0]} {SECTOR_FIELDS[rest[1]]}"
    if head == "rates":
        if len(rest) == 1:
            return RATE_TOP.get(rest[0], path)
        if len(rest) == 3 and rest[0] == "series" and rest[1] in RATE_NAMES and rest[2] in RATE_FIELDS:
            return f"{RATE_NAMES[rest[1]]} {RATE_FIELDS[rest[2]]}"
    if head == "drilldown" and len(rest) == 2 and rest[1] in DRILL_FIELDS:
        return f"{rest[0]} {DRILL_FIELDS[rest[1]]}"
    return path


def friendly_section(name: str) -> str:
    return SECTION_NAMES.get(name, name.replace("_", " ").capitalize())


def friendly_tool_line(line: str) -> str:
    """'regime: ok (neutral)' -> 'Market regime: ok (neutral)'; 'plan: drilldown, documents' -> 'Plan: Stock drill-down, Filings and news'."""
    name, _, rest = line.partition(": ")
    if name == "plan":
        rest = ", ".join(TOOL_NAMES.get(t.strip(), t.strip()) for t in rest.split(","))
    return f"{TOOL_NAMES.get(name.split(' ')[0], name)}{name[len(name.split(' ')[0]):]}: {rest}"
