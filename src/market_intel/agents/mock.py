"""Mock writer for --dry-run and tests: builds a valid, fully cited answer straight from the prompt data.

It copies FACT lines verbatim, so a dry run exercises the whole pipeline at zero cost.
"""
from __future__ import annotations

import json
import re

from market_intel.agents.prompts import SECTIONS
from market_intel.llm.client import CostTracker, LLMResult

FACT_RE = re.compile(r"^FACT (\S+) = (.*)$", re.MULTILINE)
DOC_RE = re.compile(r'<untrusted_document id="([^"]+)"[^>]*>(.*?)</untrusted_document>', re.DOTALL)

SECTION_PICKS = {
    "market_regime": ("regime.regime", "regime.score", "regime.vol_regime.vix_level", "regime.trend_state.state"),
    "sector_leaders_laggards": ("sectors.leaders", "sectors.laggards"),
    "rates_macro": ("rates.curve_10y_3m_proxy_pp",),
    "risks": ("regime.vol_regime.label", "regime.sector_breadth.pct_above"),
}


class MockLLM:
    def __init__(self) -> None:
        self.tracker = CostTracker()

    def complete(self, system: str, user: str, max_tokens: int) -> LLMResult:
        facts = dict(FACT_RE.findall(user))
        docs = DOC_RE.findall(user)
        claim = lambda path: {"text": f"{path} is {facts[path]}.", "source_ids": [path]}  # noqa: E731
        if "MODE: qa" in user:
            claims = [claim(p) for p in list(facts)[:5]]
            claims += [{"text": d[1].split("|")[0].strip(), "source_ids": [d[0]]} for d in docs[:2]]
            payload = {"claims": claims}
        else:
            sections = []
            for name in SECTIONS:
                if name == "notable_events":
                    claims = [{"text": d[1].split("|")[0].strip(), "source_ids": [d[0]]} for d in docs[:4]]
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
