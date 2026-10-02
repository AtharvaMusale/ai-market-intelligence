"""Turn tool results into FACT lines with stable dotted paths the validator can check."""
from __future__ import annotations

import re
from typing import Any

SCALARS = (str, int, float, bool)


def flatten(prefix: str, obj: Any) -> dict[str, Any]:
    """Flatten nested dicts/lists to {dotted.path: scalar}. Lists of {"ticker": ...} are keyed by ticker."""
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            out.update(flatten(f"{prefix}.{key}", value))
    elif isinstance(obj, list):
        if obj and all(isinstance(i, dict) and "ticker" in i for i in obj):
            for item in obj:
                out.update(flatten(f"{prefix}.{item['ticker']}", {k: v for k, v in item.items() if k != "ticker"}))
        elif all(isinstance(i, str) for i in obj):
            if obj:
                out[prefix] = ",".join(obj)
        else:
            for i, item in enumerate(obj):
                out.update(flatten(f"{prefix}.{i}", item))
    elif isinstance(obj, SCALARS):
        out[prefix] = obj
    return out  # None values are skipped on purpose: a missing number must not become a claim


def fact_lines(facts: dict[str, Any]) -> str:
    return "\n".join(f"FACT {path} = {value}" for path, value in facts.items())


def sanitize_excerpt(text: str, limit: int = 400) -> str:
    """Make retrieved text safe to embed: no tag lookalikes, no control characters, bounded length."""
    text = re.sub(r"[<>]", " ", text)
    return re.sub(r"\s+", " ", text).strip()[:limit]


def document_blocks(docs: list[dict]) -> str:
    """Wrap documents in tags the system prompt declares untrusted."""
    return "\n".join(
        f'<untrusted_document id="{d["doc_id"]}" ticker="{d["ticker"]}" type="{d["doc_type"]}" date="{d["date"]}">'
        f'{sanitize_excerpt(d.get("title") or "", 200)} | {sanitize_excerpt(d.get("excerpt") or "")}'
        "</untrusted_document>"
        for d in docs
    )
