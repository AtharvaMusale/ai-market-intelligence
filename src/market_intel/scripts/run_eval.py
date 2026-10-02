"""Evaluate the daily briefing (and optionally the question set) against DuckDB numbers.

Run from the project root:
    PYTHONPATH=src python -m market_intel.scripts.run_eval --dry-run        # pipeline check only (mock writer)
    PYTHONPATH=src python -m market_intel.scripts.run_eval                  # real Haiku briefing
    PYTHONPATH=src python -m market_intel.scripts.run_eval --run-questions  # also answer the fixed questions
Exit code 1 if numeric accuracy is below the threshold.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from market_intel.agents.graph import answer_question, generate_briefing, make_context
from market_intel.agents.mock import MockLLM
from market_intel.config import EVAL_ABS_TOL, EVAL_MIN_NUMERIC_ACCURACY, EVAL_REL_TOL, PROJECT_ROOT
from market_intel.db import connect
from market_intel.eval.questions import check_routing, load_questions
from market_intel.eval.verify import check_threshold, evaluate_output, reference_facts
from market_intel.llm.client import HaikuClient


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="mock writer: copies facts verbatim, so accuracy is trivially 1.0")
    parser.add_argument("--run-questions", action="store_true", help="also answer the fixed questions with the writer")
    parser.add_argument("--as-of", help="YYYY-MM-DD")
    parser.add_argument("--min-numeric", type=float, default=EVAL_MIN_NUMERIC_ACCURACY)
    parser.add_argument("--db", help="override DUCKDB_PATH")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "eval_reports" / "latest.json"))
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    con = connect(args.db, read_only=True)
    llm = MockLLM() if args.dry_run else HaikuClient()
    ctx = make_context(con, llm, as_of=args.as_of)
    facts, docs = reference_facts(con, args.as_of)

    briefing = generate_briefing(ctx)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "writer": "mock" if args.dry_run else "claude-haiku-4-5-20251001",
        "as_of": args.as_of,
        "tolerance": {"abs": EVAL_ABS_TOL, "rel": EVAL_REL_TOL},
        "threshold_numeric_accuracy": args.min_numeric,
        "briefing": evaluate_output(briefing.get("output", {}), briefing["validation"], facts, docs),
        "tool_routing": check_routing(load_questions()),
    }
    if args.run_questions:
        answers = []
        for q in load_questions():
            result = answer_question(ctx, q["question"])
            answers.append({"id": q["id"], **evaluate_output(result.get("output", {}), result["validation"], facts, docs)})
        report["answers"] = answers

    passed, message = check_threshold(report["briefing"], args.min_numeric)
    report["passed"] = passed
    report["llm_usage"] = llm.tracker.summary()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))

    b, r = report["briefing"], report["tool_routing"]
    if args.dry_run:
        print("NOTE: mock writer copies facts verbatim; these numbers test the harness, not the model.")
    print(f"Briefing: {b['claims_written']} claims, {b['numbers_checked']} numbers checked")
    print(f"  numeric match rate:     {b['numeric_match_rate']}")
    print(f"  unsupported-claim rate: {b['unsupported_claim_rate']}")
    print(f"  citation coverage:      {b['citation_coverage']}")
    for f in b["failures"]:
        print(f"  UNMATCHED {f['unmatched']} in: {f['claim']}")
    print(f"Tool routing: {r['passed']}/{r['total']} questions routed as expected")
    for res in r["results"]:
        if not res["passed"]:
            print(f"  ROUTING MISS {res['id']}: plan={res['plan']} missing={res['missing_tools']} tickers_ok={res['tickers_ok']}")
    if "answers" in report:
        rates = [a["numeric_match_rate"] for a in report["answers"] if a["numeric_match_rate"] is not None]
        print(f"Question answers: mean numeric match {sum(rates) / len(rates):.4f}" if rates else "Question answers: no numbers")
    print(f"{'PASS' if passed else 'FAIL'}: {message}")
    print(f"LLM usage: {json.dumps(report['llm_usage'])}\nReport: {args.out}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
