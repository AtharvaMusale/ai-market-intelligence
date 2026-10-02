"""Pinecone storage and search using Pinecone's hosted embeddings (the index embeds text for us).

Free-tier facts (Pinecone pricing page, checked 2026-10-02): up to 5 indexes, 2 GB storage,
2M write units and 1M read units per month, 5M embedding tokens per month, AWS us-east-1 only.
Text upserts are limited to 96 records per request.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Any

from market_intel.config import get_pinecone_settings

logger = logging.getLogger(__name__)

NAMESPACE = "market-intel"
EMBED_MODEL = "llama-text-embed-v2"
TEXT_FIELD = "chunk_text"
UPSERT_BATCH = 96


def build_filter(ticker: str | None = None, date_from: date | None = None, date_to: date | None = None) -> dict:
    """Pinecone metadata filter. Dates are compared as YYYYMMDD integers (stored as `date_int`)."""
    flt: dict[str, Any] = {}
    if ticker:
        flt["ticker"] = {"$eq": ticker}
    date_range: dict[str, int] = {}
    if date_from:
        date_range["$gte"] = int(date_from.strftime("%Y%m%d"))
    if date_to:
        date_range["$lte"] = int(date_to.strftime("%Y%m%d"))
    if date_range:
        flt["date_int"] = date_range
    return flt


class PineconeStore:
    def __init__(self, index: Any, namespace: str = NAMESPACE) -> None:
        self._index = index
        self._namespace = namespace

    @classmethod
    def connect(cls, create_index: bool = False) -> "PineconeStore":
        """Connect to the index. Creating it is an account change, so it only happens when asked."""
        from pinecone import Pinecone  # imported here so tests and dry runs need no Pinecone

        api_key, index_name = get_pinecone_settings()
        pc = Pinecone(api_key=api_key)
        if not pc.has_index(index_name):
            if not create_index:
                raise RuntimeError(f"Index {index_name!r} not found. Re-run with --create-index to create it.")
            pc.create_index_for_model(
                name=index_name,
                cloud="aws",
                region="us-east-1",  # the only region on the free plan
                embed={"model": EMBED_MODEL, "field_map": {"text": TEXT_FIELD}},
            )
        return cls(pc.Index(host=pc.describe_index(index_name).host))

    def upsert(self, records: list[dict]) -> int:
        """Upsert records (each has `_id`, `chunk_text`, and flat metadata). Same ID means overwrite."""
        for i in range(0, len(records), UPSERT_BATCH):
            self._index.upsert_records(namespace=self._namespace, records=records[i : i + UPSERT_BATCH])
        return len(records)

    def search(
        self,
        query: str,
        ticker: str | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
        top_k: int = 5,
    ) -> list[dict]:
        """Semantic search with optional ticker and date-range filters. Returns hits with their source URL."""
        kwargs: dict[str, Any] = {
            "namespace": self._namespace,
            "top_k": top_k,
            "inputs": {"text": query},
        }
        flt = build_filter(ticker, date_from, date_to)
        if flt:
            kwargs["filter"] = flt
        response = self._index.search(**kwargs)
        hits = []
        for hit in response.result.hits:
            fields = hit.fields
            hits.append(
                {
                    "chunk_id": hit.id,
                    "score": hit.score,
                    "text": fields.get(TEXT_FIELD, ""),
                    **{k: fields.get(k) for k in ("doc_id", "ticker", "date", "source", "doc_type", "title", "url")},
                }
            )
        return hits
