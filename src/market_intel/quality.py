"""Text-quality helpers shared by ingestion and retrieval. Leaf module (imports nothing from the project)."""
from __future__ import annotations

import re

# Marketplace spam and classifieds that GDELT matches on company names.
JUNK_RE = re.compile(
    r"\bfor sale\b|\bunlocked\b|wholesale|coupon|free shipping|\bdiscount\b|\bbuy now\b|\bpieces?\b"
    r"|\d\s?(?:cad|usd|aud)\b",
    re.IGNORECASE,
)
ITEM_RE = re.compile(r"Item\s+\d\.\d{2}")


def is_junk_title(title: str) -> bool:
    return bool(JUNK_RE.search(title or ""))


def trim_to_first_item(text: str) -> str:
    """8-K filings open with a long cover page; start at the first 'Item x.xx' where the news is."""
    match = ITEM_RE.search(text)
    return text[match.start():] if match else text


# SEC 8-K item codes in plain English. 9.01 (exhibits) is skipped when a more informative item is present.
ITEM_MEANINGS = {
    "1.01": "material agreement", "1.05": "cybersecurity incident", "2.01": "acquisition or disposal of assets",
    "2.02": "earnings results", "2.03": "new debt obligation", "5.02": "executive or director change",
    "5.07": "shareholder vote results", "7.01": "Regulation FD disclosure", "8.01": "other event announcement",
    "9.01": "financial statements and exhibits",
}
ITEMS_IN_TITLE_RE = re.compile(r"items ([\d.,]+)")


def describe_document(doc_type: str, title: str) -> str:
    """Plain-English label like '8-K current report (earnings results)'."""
    if doc_type == "news":
        return "news headline"
    if doc_type == "10-Q":
        return "10-Q quarterly report"
    if doc_type == "10-K":
        return "10-K annual report"
    match = ITEMS_IN_TITLE_RE.search(title or "")
    codes = [c for c in (match.group(1).split(",") if match else []) if c in ITEM_MEANINGS]
    informative = [c for c in codes if c != "9.01"] or codes
    return f"{doc_type} current report ({ITEM_MEANINGS[informative[0]]})" if informative else f"{doc_type} current report"


def recency_label(days_old: int) -> str:
    """Words, not numbers, so the writer can say how stale a document is without doing arithmetic."""
    if days_old <= 7:
        return "from this week"
    if days_old <= 30:
        return "from the last month"
    if days_old <= 90:
        return "1 to 3 months old"
    return "over 3 months old"
