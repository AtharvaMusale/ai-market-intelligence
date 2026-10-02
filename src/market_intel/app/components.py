"""HTML snippets for the app. Pure functions returning strings; every dynamic value is escaped.

Markup is deliberately one unindented line per block: Streamlit's markdown treats blank lines and
4-space indents as code, which would break the layout.
"""
from __future__ import annotations

from html import escape

import duckdb

from market_intel.agents.labels import friendly_label, friendly_section

REGIME_COLORS = {"risk_on": "var(--green)", "neutral": "var(--amber)", "risk_off": "var(--rose)"}


def nav() -> str:
    return ('<div class="nav"><div class="brand"><span class="brand-mark"></span>Market Intelligence</div>'
            '<span class="nav-note">Not investment advice</span></div>')


def hero(as_of: str | None) -> str:
    status = f'<div class="pill"><span class="dot"></span>Data as of {escape(as_of or "latest")}</div>'
    return (f'<div class="hero">{status}'
            '<div class="h1">What is moving the market, and <span class="spectrum">why</span>.</div>'
            '<p>Prices, rates, SEC filings and news turned into an auditable briefing. '
            'Every number is computed in code; every claim cites its source.</p></div>')


def rule() -> str:
    return '<div class="rule"></div>'


def spacer() -> str:
    return '<div class="spacer"></div>'


def section(number: str, label: str, title: str, sub: str = "") -> str:
    sub_html = f'<p class="sub">{escape(sub)}</p>' if sub else ""
    return (f'<div class="label"><span>{escape(number)}</span><span>{escape(label)}</span></div>'
            f'<div class="h2">{escape(title)}</div>{sub_html}')


def tiles(items: list[dict]) -> str:
    """Stat tiles. Each item: value, label, optional sub and color (a CSS color string)."""
    cells = "".join(
        f'<div class="tile"><div class="v"{_color(i)}>{escape(str(i["value"]))}</div>'
        f'<div class="k">{escape(i["label"])}</div>'
        + (f'<div class="s">{escape(str(i["sub"]))}</div>' if i.get("sub") else "")
        + "</div>"
        for i in items
    )
    return f'<div class="tiles" style="grid-template-columns:repeat({len(items)},minmax(0,1fr))">{cells}</div>'


def _color(item: dict) -> str:
    return f' style="color:{item["color"]}"' if item.get("color") else ""


def _chip(con: duckdb.DuckDBPyConnection, source_id: str, facts: dict | None) -> str:
    row = con.execute("SELECT url FROM documents WHERE doc_id = ?", [source_id]).fetchone()
    if row:  # documents are links to the original filing or article
        return f'<a class="chip" href="{escape(row[0], quote=True)}" target="_blank" rel="noopener noreferrer">{escape(source_id)}</a>'
    if facts and source_id in facts:  # readable label on screen; the exact data path stays available on hover
        label = f"{friendly_label(source_id)}: {facts[source_id]}"
    else:
        label = friendly_label(source_id)
    return f'<span class="chip" title="{escape(source_id, quote=True)}">{escape(label)}</span>'


def claims_list(claims: list[dict], con: duckdb.DuckDBPyConnection, facts: dict | None) -> str:
    items = "".join(
        f'<li><p>{escape(c["text"])}</p>{"".join(_chip(con, s, facts) for s in c["source_ids"])}</li>'
        for c in claims
    )
    return f'<ul class="claims">{items}</ul>'


def result_html(output: dict, con: duckdb.DuckDBPyConnection, facts: dict | None = None) -> str:
    """A briefing (numbered sections) or a Q&A answer (a single list of claims)."""
    if not output:
        return '<div class="empty">No briefing was produced. See the validation note below.</div>'
    if "sections" in output:
        parts = []
        for n, section_ in enumerate((s for s in output["sections"] if s["claims"]), start=1):
            title = friendly_section(section_["name"])
            parts.append(f'<div class="sec-title"><span>{n:02d}</span><span>{escape(title)}</span></div>'
                         + claims_list(section_["claims"], con, facts))
        return "".join(parts) or '<div class="empty">The writer returned no claims.</div>'
    if not output.get("claims"):
        return '<div class="empty">The available data cannot answer this question.</div>'
    return (f'<div class="sec-title"><span>Q</span><span>{escape(output["question"])}</span></div>'
            + claims_list(output["claims"], con, facts))


def chips(labels: list[str]) -> str:
    return "".join(f'<span class="chip">{escape(label)}</span>' for label in labels)


def footer() -> str:
    return ('<div class="footer"><span>Prices: Yahoo Finance via yfinance (unofficial, personal and research use). '
            'Filings: SEC EDGAR.</span><span>News headlines: '
            '<a href="https://www.gdeltproject.org/" target="_blank" rel="noopener noreferrer">GDELT Project</a>. '
            'Not investment advice.</span></div>')
