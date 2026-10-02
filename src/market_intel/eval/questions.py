"""Fixed question set with expected tool usage, checked against the router (no LLM, no cost)."""
from __future__ import annotations

import json
from pathlib import Path

from market_intel.agents.router import route

QUESTIONS_PATH = Path(__file__).with_name("questions.json")


def load_questions() -> list[dict]:
    return json.loads(QUESTIONS_PATH.read_text())


def check_routing(questions: list[dict]) -> dict:
    """A question passes when every expected tool is in the plan and the tickers found match exactly."""
    results = []
    for q in questions:
        routed = route(q["question"])
        missing = sorted(set(q["expected_tools"]) - set(routed["plan"]))
        tickers_ok = sorted(routed["tickers"]) == sorted(q.get("expected_tickers", []))
        results.append({"id": q["id"], "question": q["question"], "plan": routed["plan"],
                        "missing_tools": missing, "tickers_ok": tickers_ok, "passed": not missing and tickers_ok})
    passed = sum(r["passed"] for r in results)
    return {"tool_usage_accuracy": round(passed / len(results), 4), "passed": passed, "total": len(results),
            "results": results}
