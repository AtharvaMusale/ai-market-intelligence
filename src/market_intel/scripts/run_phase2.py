"""Phase 2 runner: ingest news + filings into DuckDB, then sync chunks to Pinecone, or search.

Run from the project root:
    PYTHONPATH=src python -m market_intel.scripts.run_phase2 --dry-run
    PYTHONPATH=src python -m market_intel.scripts.run_phase2 --create-index
    PYTHONPATH=src python -m market_intel.scripts.run_phase2 --query "iPhone demand" --ticker AAPL
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import date

from market_intel.config import WATCHLIST
from market_intel.db import connect
from market_intel.ingest.filings import ingest_filings
from market_intel.ingest.gdelt import ingest_news
from market_intel.retrieval.indexer import sync_to_pinecone
from market_intel.retrieval.store import PineconeStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tickers", nargs="+", help="subset of the watchlist")
    parser.add_argument("--skip-ingest", action="store_true", help="use documents already in DuckDB")
    parser.add_argument("--dry-run", action="store_true", help="do not call Pinecone; show what would be pushed")
    parser.add_argument("--create-index", action="store_true", help="create the Pinecone index if missing")
    parser.add_argument("--query", help="search instead of syncing")
    parser.add_argument("--ticker", help="filter search by ticker")
    parser.add_argument("--from", dest="date_from", type=date.fromisoformat, help="YYYY-MM-DD")
    parser.add_argument("--to", dest="date_to", type=date.fromisoformat, help="YYYY-MM-DD")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.query:
        store = PineconeStore.connect()
        hits = store.search(args.query, args.ticker, args.date_from, args.date_to)
        print(json.dumps(hits, indent=2, default=str))
        return 0

    con = connect()
    watchlist = {t: WATCHLIST[t] for t in args.tickers} if args.tickers else WATCHLIST
    if not args.skip_ingest:
        print("News headlines from the GDELT Project (https://www.gdeltproject.org/)")
        print(f"new news docs: {ingest_news(con, watchlist)}")
        print(f"new filings:   {ingest_filings(con, watchlist)}")

    store = None if args.dry_run else PineconeStore.connect(create_index=args.create_index)
    print(json.dumps(sync_to_pinecone(con, store, dry_run=args.dry_run), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
