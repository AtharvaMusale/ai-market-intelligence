"""Streamlit app: regime panel, sector heatmap, rates chart, cited daily brief, Q&A, evaluation results.

Run from the project root:
    PYTHONPATH=src .venv/bin/streamlit run src/market_intel/app/streamlit_app.py
The app opens DuckDB read-only and uses the free mock writer unless you choose Haiku.
"""
from __future__ import annotations

import json
from datetime import date

import altair as alt
import duckdb
import pandas as pd
import streamlit as st

from market_intel.agents.labels import friendly_tool_line
from market_intel.agents.graph import answer_question, generate_briefing, make_context
from market_intel.agents.mock import MockLLM
from market_intel.agents.tools import rates_tool, regime_tool, sectors_tool
from market_intel.analytics.loaders import load_price_matrix
from market_intel.config import ALL_PRICE_TICKERS, PROJECT_ROOT, RATES, SECTOR_ETFS, get_anthropic_key, get_duckdb_path
from market_intel.app import components as ui
from market_intel.app.theme import inject
from market_intel.llm.client import HaikuClient
from market_intel.retrieval.store import PineconeStore

EVAL_REPORT = PROJECT_ROOT / "eval_reports" / "latest.json"
REGIME_LABELS = {"risk_on": "Risk-on", "neutral": "Neutral", "risk_off": "Risk-off"}


def open_con() -> duckdb.DuckDBPyConnection | None:
    """A short-lived read-only connection, closed at the end of each page run, so refreshing data never conflicts with the app."""
    path = get_duckdb_path()
    if str(path) != ":memory:" and not path.exists():
        return None
    return duckdb.connect(str(path), read_only=True)  # the app never writes


@st.cache_data(show_spinner=False)
def load_prices(as_of: str | None) -> pd.DataFrame:
    con = open_con()
    try:
        return load_price_matrix(con, ALL_PRICE_TICKERS, as_of=as_of)
    finally:
        con.close()


FONT_MONO = "Geist Mono, ui-monospace, Menlo, monospace"


def style_chart(chart: alt.Chart) -> alt.Chart:
    """Transparent background, hairline grid and mono labels, to match the site."""
    return (chart.configure(background="transparent")
            .configure_view(strokeWidth=0)
            .configure_axis(labelColor="#a1a1a1", titleColor="#8a8a8a", gridColor="#1f1f1f", domainColor="#2e2e2e",
                            tickColor="#2e2e2e", labelFont=FONT_MONO, titleFont=FONT_MONO, labelFontSize=11)
            .configure_legend(labelColor="#a1a1a1", titleColor="#8a8a8a", labelFont=FONT_MONO, titleFont=FONT_MONO, orient="bottom", columns=2, labelLimit=240))


def heatmap(sectors: dict) -> alt.Chart | None:
    rows = [
        {"sector": f"{sectors[t]['name']} ({t})", "window": label, "return_pct": sectors[t][key]}
        for t in SECTOR_ETFS if t in sectors
        for label, key in (("1d", "ret_1d_pct"), ("5d", "ret_5d_pct"), ("21d", "ret_21d_pct"))
    ]
    if not rows:
        return None
    df = pd.DataFrame(rows)
    limit = max(abs(df["return_pct"]).max(), 1.0)
    order = [f"{sectors[t]['name']} ({t})" for t in sectors if t in SECTOR_ETFS]  # dict order is 21d rank, best first
    base = alt.Chart(df).encode(
        x=alt.X("window:O", title=None, sort=["1d", "5d", "21d"], axis=alt.Axis(orient="top", labelAngle=0)),
        y=alt.Y("sector:N", title=None, sort=order, axis=alt.Axis(labelOverlap=False, labelLimit=240)),
    )
    rect = base.mark_rect(stroke="#000", strokeWidth=2).encode(
        color=alt.Color("return_pct:Q", title="Return %", scale=alt.Scale(domain=[-limit, 0, limit], range=["#f43f5e", "#141414", "#45c26b"])),
        tooltip=["sector", "window", alt.Tooltip("return_pct:Q", format=".2f")],
    )
    text = base.mark_text(fontSize=12, font=FONT_MONO, color="#ededed").encode(text=alt.Text("return_pct:Q", format=".1f"))
    return style_chart((rect + text).properties(height=34 * len(order)))


def yields_chart(prices: pd.DataFrame) -> alt.Chart:
    cols = [t for t in RATES if t in prices.columns]
    short = {"^IRX": "13-week", "^FVX": "5-year", "^TNX": "10-year", "^TYX": "30-year"}
    df = (prices[cols].tail(252).rename(columns=short).rename_axis("date").reset_index()
          .melt("date", var_name="series", value_name="yield_pct"))
    chart = alt.Chart(df).mark_line(strokeWidth=2).encode(
        x=alt.X("date:T", title=None, axis=alt.Axis(format="%b %y")),
        y=alt.Y("yield_pct:Q", title="Yield %", scale=alt.Scale(zero=False)),
        color=alt.Color("series:N", title=None, scale=alt.Scale(range=["#22d3ee", "#a855f7", "#f43f5e", "#f59e0b"])),
        tooltip=["date:T", "series", alt.Tooltip("yield_pct:Q", format=".2f")],
    )
    return style_chart(chart.properties(height=340))


def regime_tiles(regime: dict) -> str:
    vol, trend, breadth = regime["vol_regime"], regime["trend_state"], regime["sector_breadth"]
    return ui.tiles([
        {"value": REGIME_LABELS.get(regime["regime"], regime["regime"]), "label": "Market regime",
         "sub": f"score {regime['score']}", "color": ui.REGIME_COLORS.get(regime["regime"])},
        {"value": vol["vix_level"], "label": "VIX", "sub": vol["label"]},
        {"value": trend["state"].title(), "label": "SPY trend", "sub": f"{trend['drawdown_from_52w_high_pct']}% from 52w high"},
        {"value": f"{breadth['n_above']} of {breadth['n_total']}", "label": "Sectors above 50d avg", "sub": "no sector data yet" if breadth["pct_above"] is None else f"{breadth['pct_above']}% breadth"},
        {"value": f"{vol['spy_realized_vol_21d_pct']}%", "label": "SPY 21d realized vol", "sub": f"VIX 1y percentile {vol['vix_percentile_1y']}"},
    ])


def make_writer(choice: str):
    return HaikuClient() if choice.startswith("Claude") else MockLLM()


def show_result(result: dict, con: duckdb.DuckDBPyConnection, llm) -> None:
    st.markdown(ui.result_html(result["output"], con, result.get("facts")), unsafe_allow_html=True)
    v = result["validation"]
    if "error" in v:
        st.error(v["error"])
        return
    usage = llm.tracker.summary()
    labels = [f"{v['claims_kept']} of {v['claims_total']} claims cited and verified to exist"]
    if isinstance(llm, MockLLM):
        labels.append("mock writer: free demo template")
    elif usage["calls"]:
        labels.append(f"Claude Haiku 4.5: {usage['input_tokens']} in / {usage['output_tokens']} out tokens, ~${usage['cost_usd']:.4f}"
                      + (" (cached)" if usage["cache_hits"] else ""))
    st.markdown(ui.chips(labels), unsafe_allow_html=True)
    for reason in v.get("dropped", []):
        st.warning(f"Dropped: {reason}")
    with st.expander("Tools run"):
        st.code("\n".join(friendly_tool_line(l) for l in result["tool_log"]))


def eval_tab() -> None:
    st.markdown(ui.section("04", "Evaluation", "How accurate is the briefing?",
                           "Every number in a claim is checked against values recomputed from DuckDB."), unsafe_allow_html=True)
    if not EVAL_REPORT.exists():
        st.info("No evaluation report yet. Run `PYTHONPATH=src python -m market_intel.scripts.run_eval`.")
        return
    report = json.loads(EVAL_REPORT.read_text())
    b = report["briefing"]
    passed = report.get("passed")
    st.markdown(ui.tiles([
        {"value": b["numeric_match_rate"], "label": "Numeric match rate", "sub": f"{b['numbers_matched']} of {b['numbers_checked']} numbers",
         "color": "var(--green)" if passed else "var(--rose)"},
        {"value": b["unsupported_claim_rate"], "label": "Unsupported claims"},
        {"value": b["citation_coverage"], "label": "Citation coverage"},
        {"value": f"{report['tool_routing']['passed']}/{report['tool_routing']['total']}", "label": "Questions routed correctly"},
    ]), unsafe_allow_html=True)
    st.markdown(ui.spacer() + ui.chips([
        "PASS" if passed else "FAIL", f"threshold {report['threshold_numeric_accuracy']}",
        f"writer {report['writer']}", f"generated {report['generated_at'][:10]}",
    ]), unsafe_allow_html=True)
    if report.get("answers"):
        st.dataframe(pd.DataFrame(report["answers"]).drop(columns=["failures"]), hide_index=True)
    st.caption("Checks numbers and citations against DuckDB. It does not judge interpretation or direction words.")


def main() -> None:
    st.set_page_config(page_title="Market Intelligence", page_icon=":material/change_history:", layout="wide")
    inject()
    st.markdown(ui.nav(), unsafe_allow_html=True)

    con = open_con()
    if con is None:
        st.error("No database found. Run `PYTHONPATH=src python -m market_intel.scripts.run_phase1` first.")
        return
    try:
        render(con)
    finally:
        con.close()


def render(con: duckdb.DuckDBPyConnection) -> None:

    all_prices = load_prices(None)
    latest = all_prices.index.max().date()
    with st.sidebar:
        st.markdown("#### Settings")
        as_of_date = st.date_input("As of", value=latest, min_value=all_prices.index.min().date() + pd.Timedelta(days=400), max_value=latest)
        as_of = None if as_of_date == latest else as_of_date.isoformat()
        has_key = True
        try:
            get_anthropic_key()
        except RuntimeError:
            has_key = False
        options = (["Claude Haiku 4.5 (about 1-2 cents, cached)"] if has_key else []) + ["Mock writer (free demo template)"]
        writer_choice = st.radio("Writer", options)
        if not has_key:
            st.caption("Add ANTHROPIC_API_KEY to .env to enable Haiku.")

    prices = load_prices(as_of)
    missing = [t for t in SECTOR_ETFS if t not in prices.columns]
    if missing:
        st.warning(f"No price data yet for {len(missing)} sector ETFs ({', '.join(missing)}). "
                   "Stop the app, run `PYTHONPATH=src python -m market_intel.scripts.run_phase1`, then start it again.")
    regime, sectors, rates = regime_tool(prices), sectors_tool(prices), rates_tool(prices)

    st.markdown(ui.hero(regime["as_of"]) + regime_tiles(regime) + ui.rule() + ui.spacer(), unsafe_allow_html=True)
    tab_market, tab_brief, tab_qa, tab_eval = st.tabs(["Market", "Daily brief", "Ask a question", "Evaluation"])

    with tab_market:
        st.markdown(ui.spacer() + ui.section("01", "Market", "Sector returns and rates",
                    "Sectors ranked by 21-day return. Score = trend + volatility + breadth; a high VIX forces risk-off."), unsafe_allow_html=True)
        left, right = st.columns([3, 2], gap="large")
        with left:
            chart = heatmap(sectors)
            if chart is None:
                st.markdown('<div class="empty">No sector data to show yet.</div>', unsafe_allow_html=True)
            else:
                st.altair_chart(chart, width="stretch")
            st.caption("Each row is a Vanguard sector ETF: a basket of US stocks in one sector, ranked by 21-day return. "
                       "Green is up, red is down.")
        with right:
            st.altair_chart(yields_chart(prices), width="stretch")
            st.caption(f"10Y minus 13-week proxy: {rates['curve_10y_3m_proxy_pp']} pp (not the standard 10Y-2Y spread).")

    with tab_brief:
        st.markdown(ui.spacer() + ui.section("02", "Daily brief", "The briefing, with receipts",
                    "Data claims show the value behind them. Filings and headlines link to the original."), unsafe_allow_html=True)
        if st.button("Generate briefing", key="gen_brief", type="primary"):
            llm = make_writer(writer_choice)
            with st.spinner("Running tools and writing..."):
                st.session_state["brief"] = (generate_briefing(make_context(con, llm, as_of=as_of)), llm)
        if "brief" in st.session_state:
            result, llm = st.session_state["brief"]
            show_result(result, con, llm)
        else:
            st.markdown('<div class="empty">Click Generate briefing. The mock writer is free and copies facts verbatim, so you can see the citation flow.</div>', unsafe_allow_html=True)

    with tab_qa:
        st.markdown(ui.spacer() + ui.section("03", "Ask", "Ask a question",
                    "Routed to the same tools as the briefing. Try: why did tech fall today?"), unsafe_allow_html=True)
        question = st.text_input("Question", value="why did tech fall today?", key="question", label_visibility="collapsed")
        use_pinecone = st.checkbox("Use Pinecone semantic search (needs PINECONE_API_KEY)", value=False)
        if st.button("Ask", key="ask", type="primary"):
            llm = make_writer(writer_choice)
            store = None
            if use_pinecone:
                try:
                    store = PineconeStore.connect()
                except RuntimeError as exc:
                    st.warning(f"Pinecone unavailable ({exc}); using recent stored documents.")
            with st.spinner("Routing to tools and answering..."):
                st.session_state["qa"] = (answer_question(make_context(con, llm, store, as_of), question), llm)
        if "qa" in st.session_state:
            result, llm = st.session_state["qa"]
            show_result(result, con, llm)

    with tab_eval:
        st.markdown(ui.spacer(), unsafe_allow_html=True)
        eval_tab()

    st.markdown(ui.footer(), unsafe_allow_html=True)


main()
