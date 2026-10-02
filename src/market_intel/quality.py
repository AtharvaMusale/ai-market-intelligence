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
