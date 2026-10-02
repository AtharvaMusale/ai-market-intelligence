"""Pull verifiable numbers and dates out of claim text."""
from __future__ import annotations

import re
from dataclasses import dataclass

DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")

# A number (not part of an identifier like ret_21d_pct), optionally signed, with $ / thousands commas / decimals / ordinal suffix. The lookahead skips
# window labels ("50-day", "52-week", "10-Year", "10y", "3m") because those name a period, not a measured value.
NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9._])(?P<sign>[-+−])?\$?(?P<num>\d[\d,]*(?:\.\d+)?)(?P<ord>st|nd|rd|th)?"
    r"(?!\d)(?!-?\s?(?:days?|weeks?|years?|months?)\b)(?![ymd]\b)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class NumberToken:
    raw: str
    value: float  # absolute value; direction words ("down 9.87%") are not parsed


def extract_dates(text: str) -> list[str]:
    return DATE_RE.findall(text)


def extract_numbers(text: str) -> list[NumberToken]:
    """Numbers in the text, with ISO dates removed first (they are checked separately)."""
    cleaned = DATE_RE.sub(" ", text)
    tokens = []
    for m in NUMBER_RE.finditer(cleaned):
        tokens.append(NumberToken(raw=m.group(0), value=float(m.group("num").replace(",", ""))))
    return tokens
