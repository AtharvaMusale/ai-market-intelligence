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


UP_WORDS = {"up", "rose", "gained", "gain", "gains", "climbed", "advanced", "above", "higher", "increased", "outperforming",
            "outperformed", "outperforms", "rising", "jumped", "surged", "positive", "grew"}
DOWN_WORDS = {"down", "fell", "declined", "lost", "below", "lower", "decreased", "underperforming", "underperformed",
              "underperforms", "falling", "dropped", "slid", "negative", "weaker", "shrank", "retreated", "sank"}


@dataclass(frozen=True)
class NumberToken:
    raw: str
    value: float  # absolute value
    sign: int  # explicit sign written in the text: -1, +1, or 0 if none
    start: int  # position in the text, used to find the direction word next to the number
    end: int


def extract_dates(text: str) -> list[str]:
    return DATE_RE.findall(text)


def iso_dates_from_text(text: str) -> list[str]:
    """'September 2, 2026' -> '2026-09-02', so a correctly reformatted date in a filing is recognized."""
    return [f"{int(y):04d}-{MONTHS.index(m.lower()) + 1:02d}-{int(d):02d}" for m, d, y in WRITTEN_DATE_RE.findall(text)]


def extract_numbers(text: str) -> list[NumberToken]:
    """Numbers in the text, with ISO dates and index names removed first (dates are checked separately)."""
    blank = lambda m: " " * len(m.group(0))  # noqa: E731  same length, so positions still line up with the original text
    cleaned = INDEX_NAME_RE.sub(blank, DATE_RE.sub(blank, text))
    return [
        NumberToken(
            raw=m.group(0),
            value=float(m.group("num").replace(",", "")),
            sign=-1 if m.group("sign") in ("-", "\u2212") else 1 if m.group("sign") == "+" else 0,
            start=m.start(),
            end=m.end(),
        )
        for m in NUMBER_RE.finditer(cleaned)
    ]


CLAUSE_BREAKS = {"and", "but", "while", "with", "which", "whereas", "though"}


def _direction_of(word: str) -> str | None:
    return "up" if word in UP_WORDS else "down" if word in DOWN_WORDS else None


def direction_around(text: str, start: int, end: int) -> str | None:
    """'up' or 'down' for the number at text[start:end].

    A direction word right after the number wins ("3.03% above its average"). Otherwise the nearest one just before it
    counts ("up 0.53% in one day"), without looking back across a clause break like "and" or "but".
    """
    after = re.findall(r"[A-Za-z']+", text[end : end + 22].lower())[:2]
    for word in after:
        if (d := _direction_of(word)):
            return d
    for word in reversed(re.findall(r"[A-Za-z']+", text[max(0, start - 30) : start].lower())[-3:]):
        if word in CLAUSE_BREAKS:
            return None
        if (d := _direction_of(word)):
            return d
    return None
