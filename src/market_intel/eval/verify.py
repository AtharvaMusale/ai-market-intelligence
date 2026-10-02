"""Verify a validated briefing/answer against numbers recomputed from DuckDB.

Reference numbers are rebuilt by running the same analytics tools on fresh data, so a claim is checked
against the data, not against the prompt the writer saw.
"""
from __future__ import annotations

import duckdb

from market_intel.agents.tools import assemble_facts, drilldown_tool, rates_tool, regime_tool, sectors_tool
from market_intel.analytics.loaders import load_price_matrix
from market_intel.config import ALL_PRICE_TICKERS, EVAL_ABS_TOL, EVAL_REL_TOL, WATCHLIST
from market_intel.eval.claims import extract_dates, extract_numbers, iso_dates_from_text


def reference_facts(con: duckdb.DuckDBPyConnection, as_of: str | None = None) -> tuple[dict, dict[str, str]]:
    """(facts recomputed from DuckDB, {doc_id: title + text}) for checking claims."""
    prices = load_price_matrix(con, ALL_PRICE_TICKERS, as_of=as_of)
    drilldown, _ = drilldown_tool(prices, list(WATCHLIST))
    facts = assemble_facts(regime_tool(prices), sectors_tool(prices), rates_tool(prices), drilldown)
    docs = {
        doc_id: f"{title or ''} {text or ''} {published.date().isoformat()} {' '.join(iso_dates_from_text(text or ''))}"
        for doc_id, title, text, published in con.execute(
            "SELECT doc_id, title, text, published_at FROM documents"
        ).fetchall()
    }
    return facts, docs


def _close(a: float, b: float, abs_tol: float, rel_tol: float) -> bool:
    return abs(a - b) <= max(abs_tol, rel_tol * abs(b))


def verify_claim(claim: dict, facts: dict, docs: dict[str, str], abs_tol: float = EVAL_ABS_TOL, rel_tol: float = EVAL_REL_TOL) -> dict:
    """Check every number and date in a claim against the values its own source_ids point to."""
    numeric_values: list[float] = []
    text_values: list[str] = []
    for sid in claim["source_ids"]:
        if sid in facts:
            value = facts[sid]
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                numeric_values.append(abs(float(value)))
            else:
                text_values.append(str(value))
        elif sid in docs:
            text_values.append(docs[sid])
            numeric_values += [t.value for t in extract_numbers(docs[sid])]

    checks = [
        {"raw": t.raw, "kind": "number", "matched": any(_close(t.value, v, abs_tol, rel_tol) for v in numeric_values)}
        for t in extract_numbers(claim["text"])
    ] + [
        {"raw": d, "kind": "date", "matched": any(d in tv for tv in text_values)}
        for d in extract_dates(claim["text"])
    ]
    return {"text": claim["text"], "source_ids": claim["source_ids"], "checks": checks,
            "all_matched": all(c["matched"] for c in checks)}


def all_claims(output: dict) -> list[dict]:
    if "sections" in output:
        return [c for s in output["sections"] for c in s["claims"]]
    return output.get("claims", [])


def evaluate_output(output: dict, validation: dict, facts: dict, docs: dict[str, str]) -> dict:
    """Metrics for one briefing or answer.

    numeric_match_rate: matched numbers+dates / all numbers+dates found in claim text.
    unsupported_claim_rate: (claims dropped by the validator + claims with an unmatched number) / all claims written.
    citation_coverage: claims with valid source IDs / all claims written.
    """
    results = [verify_claim(c, facts, docs) for c in all_claims(output)]
    checks = [c for r in results for c in r["checks"]]
    matched = sum(c["matched"] for c in checks)
    dropped = validation.get("claims_dropped", 0)
    written = len(results) + dropped
    bad_claims = [r for r in results if not r["all_matched"]]
    return {
        "claims_written": written,
        "claims_kept": len(results),
        "numbers_checked": len(checks),
        "numbers_matched": matched,
        "numeric_match_rate": round(matched / len(checks), 4) if checks else None,
        "unsupported_claim_rate": round((dropped + len(bad_claims)) / written, 4) if written else None,
        "citation_coverage": round(len(results) / written, 4) if written else None,
        "failures": [
            {"claim": r["text"], "unmatched": [c["raw"] for c in r["checks"] if not c["matched"]]}
            for r in bad_claims
        ],
    }


def check_threshold(report: dict, min_numeric_accuracy: float) -> tuple[bool, str]:
    """Pass/fail on numeric accuracy. No numbers to check counts as a failure, not a pass."""
    rate = report.get("numeric_match_rate")
    if rate is None:
        return False, "no numbers were found to verify"
    ok = rate >= min_numeric_accuracy
    return ok, f"numeric match rate {rate:.4f} {'>=' if ok else '<'} threshold {min_numeric_accuracy:.4f}"
