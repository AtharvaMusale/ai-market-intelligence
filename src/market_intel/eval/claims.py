"""Pull verifiable numbers and dates out of claim text."""
from __future__ import annotations

import re
from dataclasses import dataclass

DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
# Index names contain numbers that name the index, not a measured value.
INDEX_NAME_RE = re.compile(r"S&P\s?500|Nasdaq[- ]?100|Russell\s?2000|Dow\s?30", re.IGNORECASE)
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
WRITTEN_DATE_RE = re.compile(r"\b(" + "|".join(MONTHS) + r")\s+(\d{1,2}),\s+(\d{4})\b", re.IGNORECASE)

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


def iso_dates_from_text(text: str) -> list[str]:
    """'September 2, 2026' -> '2026-09-02', so a correctly reformatted date in a filing is recognized."""
    return [f"{int(y):04d}-{MONTHS.index(m.lower()) + 1:02d}-{int(d):02d}" for m, d, y in WRITTEN_DATE_RE.findall(text)]


def extract_numbers(text: str) -> list[NumberToken]:
    """Numbers in the text, with ISO dates and index names removed first (dates are checked separately)."""
    cleaned = INDEX_NAME_RE.sub(" ", DATE_RE.sub(" ", text))
    tokens = []
    for m in NUMBER_RE.finditer(cleaned):
        tokens.append(NumberToken(raw=m.group(0), value=float(m.group("num").replace(",", ""))))
    return tokens
