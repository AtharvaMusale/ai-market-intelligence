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

from market_intel.agents.briefing import render_markdown
from market_intel.agents.graph import answer_question, generate_briefing, make_context
from market_intel.agents.mock import MockLLM
from market_intel.agents.tools import rates_tool, regime_tool, sectors_tool
from market_intel.analytics.loaders import load_price_matrix
from market_intel.config import ALL_PRICE_TICKERS, PROJECT_ROOT, RATES, SECTOR_ETFS, get_anthropic_key, get_duckdb_path
from market_intel.llm.client import HaikuClient
from market_intel.retrieval.store import PineconeStore

EVAL_REPORT = PROJECT_ROOT / "eval_reports" / "latest.json"
REGIME_LABELS = {"risk_on": "Risk-on", "neutral": "Neutral", "risk_off": "Risk-off"}


@st.cache_resource
def get_con() -> duckdb.DuckDBPyConnection | None:
    path = get_duckdb_path()
    if str(path) != ":memory:" and not path.exists():
        return None
    return duckdb.connect(str(path), read_only=True)  # the app never writes


@st.cache_data(show_spinner=False)
def load_prices(as_of: str | None) -> pd.DataFrame:
    return load_price_matrix(get_con(), ALL_PRICE_TICKERS, as_of=as_of)


def heatmap(sectors: dict) -> alt.Chart:
    rows = [
        {"sector": f"{t} {sectors[t]['name']}", "window": label, "return_pct": sectors[t][key]}
        for t in SECTOR_ETFS if t in sectors
        for label, key in (("1d", "ret_1d_pct"), ("5d", "ret_5d_pct"), ("21d", "ret_21d_pct"))
    ]
    df = pd.DataFrame(rows)
    order = [f"{t} {sectors[t]['name']}" for t in sectors if t in SECTOR_ETFS]  # dict order is 21d rank, best first
    base = alt.Chart(df).encode(
        x=alt.X("window:O", title=None, sort=["1d", "5d", "21d"]),
        y=alt.Y("sector:N", title=None, sort=list(dict.fromkeys(order))),
    )
    rect = base.mark_rect().encode(
        color=alt.Color("return_pct:Q", title="Return %", scale=alt.Scale(scheme="redyellowgreen", domainMid=0)),
        tooltip=["sector", "window", alt.Tooltip("return_pct:Q", format=".2f")],
    )
    text = base.mark_text(fontSize=12).encode(text=alt.Text("return_pct:Q", format=".1f"))
    return (rect + text).properties(height=330)


def regime_panel(regime: dict) -> None:
    label = REGIME_LABELS.get(regime["regime"], regime["regime"])
    vol, trend, breadth = regime["vol_regime"], regime["trend_state"], regime["sector_breadth"]
    top = st.columns(3)
    top[0].metric("Regime", label, f"score {regime['score']}", delta_color="off", delta_arrow="off")
    top[1].metric("VIX", vol["vix_level"], vol["label"], delta_color="off", delta_arrow="off")
    top[2].metric("SPY trend", trend["state"].title(), f"{trend['drawdown_from_52w_high_pct']}% from 52w high", delta_color="off", delta_arrow="off")
    bottom = st.columns(3)
    bottom[0].metric("Breadth", f"{breadth['n_above']} of {breadth['n_total']}", "sectors above 50d avg", delta_color="off", delta_arrow="off")
    bottom[1].metric("SPY 21d realized vol", f"{vol['spy_realized_vol_21d_pct']}%", f"VIX 1y percentile {vol['vix_percentile_1y']}", delta_color="off", delta_arrow="off")
    st.caption(f"Data as of {regime['as_of']} (SPY close). Score = trend + volatility + breadth; a high VIX forces risk-off.")


def make_writer(choice: str):
    return HaikuClient() if choice.startswith("Claude") else MockLLM()


def show_result(result: dict, con: duckdb.DuckDBPyConnection, llm) -> None:
    with st.expander("Tools run", expanded=False):
        st.code("\n".join(result["tool_log"]))
    st.markdown(render_markdown(result["output"], con, result.get("facts")))
    v = result["validation"]
    if "error" in v:
        st.error(v["error"])
    else:
        st.caption(f"Citation check: {v['claims_kept']} of {v['claims_total']} claims kept; every kept claim cites a real data path or document.")
        for reason in v.get("dropped", []):
            st.warning(f"Dropped: {reason}")
    usage = llm.tracker.summary()
    if usage["calls"] and not isinstance(llm, MockLLM):
        st.caption(f"LLM: {usage['input_tokens']} in / {usage['output_tokens']} out tokens, ~${usage['cost_usd']:.4f}"
                   + (" (served from cache)" if usage["cache_hits"] else ""))


def eval_tab() -> None:
    if not EVAL_REPORT.exists():
        st.info("No evaluation report yet. Run `PYTHONPATH=src python -m market_intel.scripts.run_eval`.")
        return
    report = json.loads(EVAL_REPORT.read_text())
    b = report["briefing"]
    st.caption(f"Writer: {report['writer']} | generated {report['generated_at']} | threshold {report['threshold_numeric_accuracy']}")
    cols = st.columns(4)
    cols[0].metric("Numeric match rate", b["numeric_match_rate"], f"{b['numbers_matched']}/{b['numbers_checked']} numbers", delta_color="off", delta_arrow="off")
    cols[1].metric("Unsupported claims", b["unsupported_claim_rate"], delta_color="off", delta_arrow="off")
    cols[2].metric("Citation coverage", b["citation_coverage"], delta_color="off", delta_arrow="off")
    cols[3].metric("Tool routing", f"{report['tool_routing']['passed']}/{report['tool_routing']['total']}", delta_color="off", delta_arrow="off")
    st.success("PASS") if report.get("passed") else st.error("FAIL")
    if report.get("answers"):
        st.dataframe(pd.DataFrame(report["answers"]).drop(columns=["failures"]), hide_index=True)
    st.caption("Checks numbers and citations against DuckDB. It does not judge interpretation or direction words.")


def main() -> None:
    st.set_page_config(page_title="AI Market Intelligence", page_icon="📈", layout="wide")
    st.title("AI Market Intelligence")
    st.caption("Auditable, cited market briefing. All numbers are computed in code; the LLM only narrates and cites.")

    con = get_con()
    if con is None:
        st.error("No database found. Run `PYTHONPATH=src python -m market_intel.scripts.run_phase1` first.")
        return

    all_prices = load_prices(None)
    latest = all_prices.index.max().date()
    with st.sidebar:
        st.header("Settings")
        as_of_date = st.date_input("As of", value=latest, min_value=all_prices.index.min().date() + pd.Timedelta(days=400), max_value=latest)
        as_of = None if as_of_date == latest else as_of_date.isoformat()
        has_key = True
        try:
            get_anthropic_key()
        except RuntimeError:
            has_key = False
        options = ["Mock writer (free)"] + (["Claude Haiku 4.5 (~2 cents, cached)"] if has_key else [])
        writer_choice = st.radio("Writer", options)
        if not has_key:
            st.caption("Add ANTHROPIC_API_KEY to .env to enable Haiku.")
        st.divider()
        st.caption("Prices: Yahoo Finance via yfinance (unofficial, personal/research use). Filings: SEC EDGAR. News headlines: [GDELT Project](https://www.gdeltproject.org/). Not investment advice.")

    prices = load_prices(as_of)
    regime, sectors, rates = regime_tool(prices), sectors_tool(prices), rates_tool(prices)
    regime_panel(regime)

    tab_market, tab_brief, tab_qa, tab_eval = st.tabs(["Market", "Daily brief", "Ask a question", "Evaluation"])

    with tab_market:
        left, right = st.columns([3, 2])
        with left:
            st.subheader("Sector returns")
            st.altair_chart(heatmap(sectors), use_container_width=True)
        with right:
            st.subheader("Treasury yields (1 year)")
            yields = prices[[t for t in RATES if t in prices.columns]].tail(252).rename(columns=RATES)
            st.line_chart(yields)
            curve = rates["curve_10y_3m_proxy_pp"]
            st.caption(f"10Y minus 13-week proxy: {curve} pp (not the standard 10Y-2Y spread).")

    with tab_brief:
        if st.button("Generate briefing", key="gen_brief", type="primary"):
            llm = make_writer(writer_choice)
            with st.spinner("Running tools and writing..."):
                st.session_state["brief"] = (generate_briefing(make_context(con, llm, as_of=as_of)), llm)
        if "brief" in st.session_state:
            result, llm = st.session_state["brief"]
            show_result(result, con, llm)
        else:
            st.info("Click Generate briefing. The mock writer is free; it copies facts verbatim so you can see the citation flow.")

    with tab_qa:
        question = st.text_input("Question", value="why did tech fall today?", key="question")
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
        eval_tab()


main()
