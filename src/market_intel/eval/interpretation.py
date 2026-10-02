"""Rule-based checks of interpretive wording against the facts a claim cites.

These catch claims like "in an uptrend" when the cited trend fact says downtrend, or "at 52-week highs" when the
stock is 5% below its high. They are explicit rules, not a judgment model, and cover a small set of phrases.
"""
from __future__ import annotations

import re

NEAR_HIGH_PCT = -1.0  # "at/near a 52-week high" requires the drop from the high to be no worse than this
_NEAR_HIGH_RE = re.compile(r"\b(?:at|near)\b[^.;]{0,25}52-week high", re.IGNORECASE)


def _cited(claim: dict, facts: dict, *suffixes: str) -> list:
    # "components.trend" is a score (-1/0/1), not a trend label, so it never counts as evidence for wording.
    return [facts[s] for s in claim["source_ids"] if s in facts and s.endswith(suffixes) and ".components." not in s]


def check_interpretation(claim: dict, facts: dict) -> list[str]:
    """Problems found in the claim's interpretive wording (empty list = none found)."""
    text, issues = claim["text"].lower(), []

    for word in ("uptrend", "downtrend"):
        if re.search(rf"\b{word}\b", text):
            states = " ".join(str(v) for v in _cited(claim, facts, ".trend", ".state", ".verdict"))
            if not states:
                issues.append(f"says {word} without citing a trend fact")
            elif word not in states:
                issues.append(f"says {word} but the cited trend is: {states}")

    for word in ("overbought", "oversold"):
        if word in text:
            zones = " ".join(str(v) for v in _cited(claim, facts, ".rsi_zone", ".verdict"))  # the verdict states the zone too
            if not zones:
                issues.append(f"says {word} without citing the RSI zone")
            elif word not in zones:
                issues.append(f"says {word} but the cited RSI zone is: {zones}")

    for word, expect in (("outperform", "outperforming"), ("underperform", "underperforming")):
        if word in text:
            labels = _cited(claim, facts, ".vs_spy_21d", ".verdict")  # the verdict states relative performance too
            deltas = _cited(claim, facts, ".ret_21d_vs_spy_pct")
            if labels and expect not in " ".join(str(v) for v in labels):
                issues.append(f"says {word} but the cited label is: {labels[0]}")
            elif not labels and deltas and ((deltas[0] > 0) != (word == "outperform")):
                issues.append(f"says {word} but the cited difference is {deltas[0]}")
            elif not labels and not deltas:
                issues.append(f"says {word} without citing relative performance")

    if _NEAR_HIGH_RE.search(text):
        drops = _cited(claim, facts, ".drawdown_from_52w_high_pct")
        if not drops:
            issues.append("says at/near a 52-week high without citing the drop from the high")
        elif drops[0] < NEAR_HIGH_PCT:
            issues.append(f"says at/near a 52-week high but the stock is {drops[0]}% from it")
    return issues
