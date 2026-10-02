"""Phase 1 runner: ingest prices into DuckDB, then print the regime summary as JSON.

Run from the project root:
    PYTHONPATH=src python -m market_intel.scripts.run_phase1
    PYTHONPATH=src python -m market_intel.scripts.run_phase1 --skip-ingest
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from market_intel.analytics.loaders import load_price_matrix
from market_intel.analytics.regime import regime_summary
from market_intel.config import ALL_PRICE_TICKERS
from market_intel.db import connect
from market_intel.ingest.prices import ingest_prices


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-ingest", action="store_true", help="use data already in DuckDB")
    parser.add_argument("--as-of", help="YYYY-MM-DD: compute as if this were today (no look-ahead)")
    parser.add_argument("--db", help="override DUCKDB_PATH")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    con = connect(args.db)

    if not args.skip_ingest:
        ingest_prices(con)

    prices = load_price_matrix(con, ALL_PRICE_TICKERS, as_of=args.as_of)
    if prices.empty:
        print("No price data in DuckDB. Run without --skip-ingest first.", file=sys.stderr)
        return 1

    print(json.dumps(regime_summary(prices), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
