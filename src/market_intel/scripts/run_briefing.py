"""Generate the daily briefing.

Run from the project root:
    PYTHONPATH=src python -m market_intel.scripts.run_briefing --dry-run     # mock writer, zero cost
    PYTHONPATH=src python -m market_intel.scripts.run_briefing               # real Haiku call (needs ANTHROPIC_API_KEY)
"""
from __future__ import annotations

import argparse
import json
import logging

from market_intel.agents.briefing import render_markdown
from market_intel.agents.graph import generate_briefing, make_context
from market_intel.agents.mock import MockLLM
from market_intel.db import connect
from market_intel.llm.client import HaikuClient


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="use the mock writer (no API call, no cost)")
    parser.add_argument("--as-of", help="YYYY-MM-DD: only use data available on that date")
    parser.add_argument("--json", action="store_true", help="print the validated JSON instead of Markdown")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    con = connect(read_only=True)
    llm = MockLLM() if args.dry_run else HaikuClient()
    result = generate_briefing(make_context(con, llm, as_of=args.as_of))

    print("Tools run:\n  " + "\n  ".join(result["tool_log"]))
    print(json.dumps(result["output"], indent=2) if args.json else render_markdown(result["output"], con))
    print(f"\nValidation: {json.dumps(result['validation'])}")
    print(f"LLM usage: {json.dumps(llm.tracker.summary())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
