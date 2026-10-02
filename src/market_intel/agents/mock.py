"""Mock writer for --dry-run and tests: builds a valid, fully cited answer straight from the prompt data.

It copies FACT lines verbatim, so a dry run exercises the whole pipeline at zero cost.
"""
from __future__ import annotations

import json
import re

from market_intel.agents.labels import friendly_label
from market_intel.agents.prompts import SECTIONS
from market_intel.llm.client import CostTracker, LLMResult

FACT_RE = re.compile(r"^FACT (\S+) = (.*)$", re.MULTILINE)
DOC_RE = re.compile(r'<untrusted_document id="([^"]+)"([^>]*)>(.*?)</untrusted_document>', re.DOTALL)
ATTR_RE = re.compile(r'(\w+)="([^"]*)"')

SECTION_PICKS = {
    "market_regime": ("regime.regime", "regime.score", "regime.vol_regime.vix_level", "regime.trend_state.state"),
    "sector_leaders_laggards": ("sectors.leaders", "sectors.laggards"),
    "rates_macro": ("rates.curve_10y_3m_proxy_pp",),
    "risks": ("regime.vol_regime.label", "regime.sector_breadth.pct_above"),
}


def _side(value: str, above: str, below: str) -> str:
    return above if float(value) >= 0 else below


def qa_claims(facts: dict, docs: list[tuple[str, dict, str]]) -> list[dict]:
    """Readable template answer built from facts. Numbers are copied verbatim and each claim cites its own facts."""
    claims: list[dict] = []

    def add(text: str, *paths: str) -> None:
        if all(p in facts for p in paths):
            claims.append({"text": text, "source_ids": list(paths)})

    f = facts.get
    for t in sorted({p.split(".")[1] for p in facts if p.startswith("drilldown.")}):
        d = f"drilldown.{t}."
        add(f"Bottom line for {t}: {f(d + 'verdict')}.", d + "verdict")
        add(f"{t} last closed at {f(d + 'last')}, with returns of {f(d + 'ret_1d_pct')}% over 1 day, "
            f"{f(d + 'ret_5d_pct')}% over 5 days and {f(d + 'ret_21d_pct')}% over 21 days.",
            d + "last", d + "ret_1d_pct", d + "ret_5d_pct", d + "ret_21d_pct")
        if d + "pct_vs_sma_50" in facts and d + "drawdown_from_52w_high_pct" in facts:
            add(f"{t} trades {abs(float(f(d + 'pct_vs_sma_50')))}% {_side(f(d + 'pct_vs_sma_50'), 'above', 'below')} its 50-day average "
                f"and {abs(float(f(d + 'drawdown_from_52w_high_pct')))}% below its 52-week high.",
                d + "pct_vs_sma_50", d + "drawdown_from_52w_high_pct")
        add(f"Its RSI is {f(d + 'rsi_14')}, which is {f(d + 'rsi_zone')}, and it is {f(d + 'vs_spy_21d')} SPY by "
            f"{abs(float(f(d + 'ret_21d_vs_spy_pct', 0)))} percentage points over 21 days.",
            d + "rsi_14", d + "rsi_zone", d + "vs_spy_21d", d + "ret_21d_vs_spy_pct")
    add(f"The market regime is {f('regime.regime')}, with the VIX at {f('regime.vol_regime.vix_level')} ({f('regime.vol_regime.label')}).",
        "regime.regime", "regime.vol_regime.vix_level", "regime.vol_regime.label")
    add(f"Leading sectors over 21 days: {f('sectors.leaders')}; lagging: {f('sectors.laggards')}.",
        "sectors.leaders", "sectors.laggards")
    add(f"The 10-year yield is {f('rates.series.^TNX.latest_pct')}% and the 13-week yield is {f('rates.series.^IRX.latest_pct')}%.",
        "rates.series.^TNX.latest_pct", "rates.series.^IRX.latest_pct")
    for doc_id, attrs, body in docs[:4]:
        if attrs.get("type") == "news":
            claims.append({"text": f"{body.split('|')[0].strip()} ({attrs.get('recency', '')}).", "source_ids": [doc_id]})
        else:
            claims.append({"text": f"{attrs.get('ticker', '')} {attrs.get('desc', '')}, filed {attrs.get('date', '')} ({attrs.get('recency', '')}).",
                           "source_ids": [doc_id]})
    return claims


class MockLLM:
    def __init__(self) -> None:
        self.tracker = CostTracker()

    def complete(self, system: str, user: str, max_tokens: int) -> LLMResult:
        facts = dict(FACT_RE.findall(user))
        docs = [(i, dict(ATTR_RE.findall(a)), body) for i, a, body in DOC_RE.findall(user)]
        claim = lambda path: {"text": f"{friendly_label(path)}: {facts[path]}.", "source_ids": [path]}  # noqa: E731
        if "MODE: qa" in user:
            payload = {"claims": qa_claims(facts, docs) or [claim(p) for p in list(facts)[:5]]}
        else:
            sections = []
            for name in SECTIONS:
                if name == "notable_events":
                    claims = [{"text": d[2].split("|")[0].strip(), "source_ids": [d[0]]} for d in docs[:4]]
                elif name == "watchlist_notes":
                    picks = [p for p in facts if p.startswith("drilldown.") and p.endswith(".ret_21d_pct")][:4]
                    claims = [claim(p) for p in picks]
                else:
                    claims = [claim(p) for p in SECTION_PICKS[name] if p in facts]
                    if name == "rates_macro":
                        claims += [claim(p) for p in facts if p.startswith("rates.^") and p.endswith("latest_pct")][:3]
                sections.append({"name": name, "claims": claims})
            payload = {"sections": sections}
        result = LLMResult(json.dumps(payload))
        self.tracker.add(result)
        return result
