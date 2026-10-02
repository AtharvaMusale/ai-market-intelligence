"""Sync DuckDB documents to Pinecone: chunk new documents, then push only chunks not yet pushed.

DuckDB remembers what was pushed (`chunks.pushed_at`), so re-running spends no embedding tokens
on text that has not changed.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import duckdb
import pandas as pd

from market_intel.config import EMBED_TOKEN_BUDGET_PER_RUN
from market_intel.db import upsert_df
from market_intel.retrieval.chunking import chunk_id, chunk_text, content_hash
from market_intel.retrieval.store import TEXT_FIELD, PineconeStore

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN = 4  # rough estimate, used only for the budget guard


def build_chunks(con: duckdb.DuckDBPyConnection) -> int:
    """Create chunk rows for documents that have none yet."""
    docs = con.execute(
        "SELECT doc_id, text FROM documents WHERE doc_id NOT IN (SELECT doc_id FROM chunks)"
    ).fetchall()
    rows = [
        {
            "chunk_id": chunk_id(doc_id, idx),
            "doc_id": doc_id,
            "chunk_idx": idx,
            "text": piece,
            "content_hash": content_hash(piece),
            "pushed_at": pd.NaT,
        }
        for doc_id, text in docs
        for idx, piece in enumerate(chunk_text(text or ""))
    ]
    return upsert_df(con, "chunks", pd.DataFrame(rows, columns=["chunk_id", "doc_id", "chunk_idx", "text", "content_hash", "pushed_at"]))


def sync_to_pinecone(
    con: duckdb.DuckDBPyConnection,
    store: PineconeStore | None,
    token_budget: int = EMBED_TOKEN_BUDGET_PER_RUN,
    dry_run: bool = False,
) -> dict:
    """Push pending chunks. With dry_run (or no store) nothing is sent; the plan is returned instead."""
    new_chunks = build_chunks(con)
    pending = con.execute(
        """
        SELECT c.chunk_id, c.doc_id, c.text, d.ticker, d.published_at, d.source, d.doc_type, d.title, d.url
        FROM chunks c JOIN documents d USING (doc_id)
        WHERE c.pushed_at IS NULL
        ORDER BY c.chunk_id
        """
    ).df()
    est_tokens = int(pending["text"].str.len().sum() // CHARS_PER_TOKEN) if len(pending) else 0
    result = {"new_chunks": new_chunks, "pending_chunks": len(pending), "estimated_tokens": est_tokens, "pushed": 0}
    if est_tokens > token_budget:
        raise RuntimeError(
            f"Estimated {est_tokens} embedding tokens exceeds the per-run budget of {token_budget}. "
            "Narrow the watchlist or raise EMBED_TOKEN_BUDGET_PER_RUN deliberately."
        )
    if dry_run or store is None or pending.empty:
        return result

    records = [
        {
            "_id": row.chunk_id,
            TEXT_FIELD: row.text,
            "doc_id": row.doc_id,
            "ticker": row.ticker,
            "date": row.published_at.date().isoformat(),
            "date_int": int(row.published_at.strftime("%Y%m%d")),
            "source": row.source,
            "doc_type": row.doc_type,
            "title": row.title or "",
            "url": row.url,
        }
        for row in pending.itertuples()
    ]
    store.upsert(records)
    con.execute(
        "UPDATE chunks SET pushed_at = ? WHERE list_contains(?, chunk_id)",
        [datetime.now(timezone.utc).replace(tzinfo=None), pending["chunk_id"].tolist()],
    )
    result["pushed"] = len(records)
    return result
