"""Rule-based Q&A router: picks which tools to run. No LLM call, so routing is free and testable."""
from __future__ import annotations

import re

from market_intel.config import SECTOR_ETFS, WATCHLIST

SECTOR_WORDS = {
    "tech": "VGT", "technology": "VGT", "financial": "VFH", "banks": "VFH", "energy": "VDE", "oil": "VDE",
    "health": "VHT", "healthcare": "VHT", "consumer": "VCR", "staples": "VDC", "industrial": "VIS",
    "materials": "VAW", "utilities": "VPU", "real estate": "VNQ", "communication": "VOX",
}
RATE_WORDS = ("yield", "rate", "treasury", "curve", "bond")
REGIME_WORDS = ("vix", "volatil", "regime", "risk", "market", "selloff", "sell-off", "rally", "fall", "fell", "drop", "trend", "breadth")
TEXT_WORDS = ("why", "news", "filing", "earnings", "8-k", "10-k", "10-q", "announce", "report", "event", "headline")


def route(question: str) -> dict:
    """Return {"plan": [tool names], "tickers": [...], "query": str} for a question."""
    q = question.lower()
    tickers = [
        t for t, name in WATCHLIST.items()
        if re.search(rf"\b{t.lower()}\b", q) or name.lower() in q
    ]
    plan: list[str] = []
    if any(w in q for w in SECTOR_WORDS) or any(t.lower() in q.split() for t in SECTOR_ETFS) or "sector" in q:
        plan.append("sectors")
    if any(w in q for w in RATE_WORDS):
        plan.append("rates")
    if any(w in q for w in REGIME_WORDS):
        plan.append("regime")
    if tickers:
        plan.append("drilldown")
    if tickers or any(w in q for w in TEXT_WORDS):
        plan.append("documents")
    if not plan:
        plan = ["regime", "sectors", "documents"]
    return {"plan": list(dict.fromkeys(plan)), "tickers": tickers, "query": question}
