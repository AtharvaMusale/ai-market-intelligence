"""Evaluate the briefing across many past dates (point-in-time replay), to see accuracy over time, not one lucky run.

Run from the project root:
    PYTHONPATH=src python -m market_intel.scripts.run_eval_history --days 20                      # free: mock writer
    PYTHONPATH=src python -m market_intel.scripts.run_eval_history --days 20 --haiku --confirm-cost  # real Haiku, paid

Each date uses only data available on that date. With --haiku the run costs about 1.6 cents per date and will not
start without --confirm-cost.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from market_intel.agents.graph import generate_briefing, make_context
from market_intel.agents.mock import MockLLM
from market_intel.config import ALL_PRICE_TICKERS, EVAL_MIN_NUMERIC_ACCURACY, PROJECT_ROOT
from market_intel.analytics.loaders import load_price_matrix
from market_intel.db import connect
from market_intel.eval.verify import check_threshold, evaluate_output, reference_facts
from market_intel.llm.client import HaikuClient

EST_COST_PER_DATE_USD = 0.016  # measured: about 6.8k input + 1.8k output tokens per briefing


def replay_dates(con, days: int) -> list[str]:
    """The last `days` trading dates that have SPY data, oldest first."""
    prices = load_price_matrix(con, ["SPY"])
    return [d.date().isoformat() for d in prices["SPY"].dropna().index[-days:]]


def summarize(rows: list[dict], min_numeric: float) -> dict:
    rates = [r["numeric_match_rate"] for r in rows if r["numeric_match_rate"] is not None]
    return {
        "dates": len(rows),
        "mean_numeric_match": round(sum(rates) / len(rates), 4) if rates else None,
        "min_numeric_match": min(rates) if rates else None,
        "dates_below_threshold": sum(1 for r in rows if not check_threshold(r, min_numeric)[0]),
        "direction_errors": sum(r["direction_errors"] for r in rows),
        "direction_checked": sum(r["direction_checked"] for r in rows),
        "interpretation_issues": sum(r["interpretation_issues"] for r in rows),
        "mean_citation_coverage": round(sum(r["citation_coverage"] for r in rows if r["citation_coverage"] is not None) / max(len(rows), 1), 4),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--days", type=int, default=20)
    parser.add_argument("--haiku", action="store_true", help="use the real writer (paid)")
    parser.add_argument("--confirm-cost", action="store_true", help="required with --haiku: you accept the estimated cost")
    parser.add_argument("--min-numeric", type=float, default=EVAL_MIN_NUMERIC_ACCURACY)
    parser.add_argument("--db", help="override DUCKDB_PATH")
    parser.add_argument("--out", help="default: eval_reports/history.json (Haiku) or history-mock.json (mock, git-ignored)")
    args = parser.parse_args(argv)
    args.out = args.out or str(PROJECT_ROOT / "eval_reports" / ("history.json" if args.haiku else "history-mock.json"))

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    con = connect(args.db, read_only=True)
    dates = replay_dates(con, args.days)
    if args.haiku:
        estimate = EST_COST_PER_DATE_USD * len(dates)
        if not args.confirm_cost:
            print(f"Refusing to call Haiku: {len(dates)} dates would cost about ${estimate:.2f}. Re-run with --confirm-cost to accept.")
            return 2
        print(f"Real Haiku replay of {len(dates)} dates, estimated ${estimate:.2f}.")
    else:
        print("NOTE: mock writer copies facts verbatim, so scores test the harness, not the model.")

    llm = HaikuClient() if args.haiku else MockLLM()
    rows = []
    for as_of in dates:
        facts, docs = reference_facts(con, as_of)
        result = generate_briefing(make_context(con, llm, as_of=as_of))
        row = {"as_of": as_of, **evaluate_output(result.get("output", {}), result["validation"], facts, docs)}
        row.pop("failures"), rows.append(row)
        print(f"{as_of}  numbers {row['numbers_matched']}/{row['numbers_checked']}  match {row['numeric_match_rate']}  "
              f"direction errors {row['direction_errors']}  interpretation issues {row['interpretation_issues']}")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "writer": "claude-haiku-4-5-20251001" if args.haiku else "mock",
        "threshold_numeric_accuracy": args.min_numeric,
        "summary": summarize(rows, args.min_numeric),
        "dates": rows,
        "llm_usage": llm.tracker.summary(),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2))
    print(f"LLM usage: {json.dumps(report['llm_usage'])}\nReport: {args.out}")
    return 0 if report["summary"]["dates_below_threshold"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
