"""Ask a market question, answered from the same tools as the briefing.

Run from the project root:
    PYTHONPATH=src python -m market_intel.scripts.ask "why did tech fall today?" --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging

from market_intel.agents.briefing import render_markdown
from market_intel.agents.graph import answer_question, make_context
from market_intel.agents.mock import MockLLM
from market_intel.db import connect
from market_intel.llm.client import HaikuClient
from market_intel.retrieval.store import PineconeStore

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question")
    parser.add_argument("--dry-run", action="store_true", help="use the mock writer (no API call, no cost)")
    parser.add_argument("--no-search", action="store_true", help="skip Pinecone; use recent stored documents")
    parser.add_argument("--as-of", help="YYYY-MM-DD: only use data available on that date")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    con = connect()
    llm = MockLLM() if args.dry_run else HaikuClient()
    store = None
    if not args.no_search:
        try:
            store = PineconeStore.connect()
        except RuntimeError as exc:
            logger.warning("Pinecone unavailable (%s); using recent stored documents instead", exc)
    result = answer_question(make_context(con, llm, store, args.as_of), args.question)

    print("Tools run:\n  " + "\n  ".join(result["tool_log"]))
    print(render_markdown(result["output"], con))
    print(f"\nValidation: {json.dumps(result['validation'])}")
    print(f"LLM usage: {json.dumps(llm.tracker.summary())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
