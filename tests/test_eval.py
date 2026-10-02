"""Tests for the evaluation harness: known-good and known-bad briefings, thresholds, routing."""
import json

from conftest import synthetic_prices, to_long

from market_intel.agents.graph import generate_briefing, make_context
from market_intel.agents.mock import MockLLM
from market_intel.db import connect, upsert_df
from market_intel.eval.claims import extract_dates, extract_numbers
from market_intel.eval.questions import check_routing, load_questions
from market_intel.eval.verify import check_threshold, evaluate_output, reference_facts, verify_claim
from market_intel.scripts.run_eval import main


def claim(text, *ids):
    return {"text": text, "source_ids": list(ids)}


def test_extract_numbers_skips_window_labels_and_dates():
    text = "SPY 770.22 is above its 50-day MA of 762.24 and 52-week high on 2026-10-02; 10-Year at 5.2%, ranked 11th, $1,234.5"
    assert [t.value for t in extract_numbers(text)] == [770.22, 762.24, 5.2, 11.0, 1234.5]
    assert extract_dates(text) == ["2026-10-02"]


def test_correct_claim_passes_and_planted_wrong_number_is_caught(seeded_con):
    facts, docs = reference_facts(seeded_con)
    vix = facts["regime.vol_regime.vix_level"]
    good = verify_claim(claim(f"VIX is at {vix}.", "regime.vol_regime.vix_level"), facts, docs)
    bad = verify_claim(claim(f"VIX is at {vix + 5}.", "regime.vol_regime.vix_level"), facts, docs)
    assert good["all_matched"] and not bad["all_matched"]


def test_tolerance_allows_rounding_but_not_real_differences(seeded_con):
    facts, docs = reference_facts(seeded_con)
    v = facts["sectors.XLK.ret_21d_pct"]
    assert verify_claim(claim(f"XLK {round(v, 2)}%.", "sectors.XLK.ret_21d_pct"), facts, docs)["all_matched"]
    assert not verify_claim(claim(f"XLK {v + 0.5}%.", "sectors.XLK.ret_21d_pct"), facts, docs)["all_matched"]


def test_number_must_match_a_cited_source_not_any_source(seeded_con):
    facts, docs = reference_facts(seeded_con)
    vix = facts["regime.vol_regime.vix_level"]
    wrong_citation = verify_claim(claim(f"VIX is {vix}.", "regime.score"), facts, docs)
    assert not wrong_citation["all_matched"]


def test_sign_is_ignored_so_down_9_percent_matches_negative_value(seeded_con):
    facts, docs = reference_facts(seeded_con)
    path = next(p for p, v in facts.items() if p.endswith("ret_21d_pct") and isinstance(v, float) and v < 0)
    assert verify_claim(claim(f"Down {abs(facts[path])}%.", path), facts, docs)["all_matched"]


def test_dates_and_document_numbers(seeded_con):
    facts, docs = reference_facts(seeded_con)
    as_of = facts["regime.as_of"]
    assert verify_claim(claim(f"As of {as_of}.", "regime.as_of"), facts, docs)["all_matched"]
    assert not verify_claim(claim("As of 1999-01-01.", "regime.as_of"), facts, docs)["all_matched"]


def test_metrics_count_validator_drops_as_unsupported(seeded_con):
    facts, docs = reference_facts(seeded_con)
    vix = facts["regime.vol_regime.vix_level"]
    output = {"claims": [claim(f"VIX {vix}.", "regime.vol_regime.vix_level"), claim("VIX 99.9.", "regime.vol_regime.vix_level")]}
    report = evaluate_output(output, {"claims_dropped": 2}, facts, docs)
    assert report["claims_written"] == 4 and report["numbers_checked"] == 2 and report["numbers_matched"] == 1
    assert report["numeric_match_rate"] == 0.5
    assert report["unsupported_claim_rate"] == 0.75  # 2 dropped + 1 wrong number, out of 4
    assert report["citation_coverage"] == 0.5
    assert report["failures"][0]["unmatched"] == ["99.9"]


def test_threshold_fails_below_and_when_no_numbers():
    assert check_threshold({"numeric_match_rate": 0.9}, 0.95)[0] is False
    assert check_threshold({"numeric_match_rate": 0.95}, 0.95)[0] is True
    assert check_threshold({"numeric_match_rate": None}, 0.95)[0] is False


def test_mock_briefing_scores_perfectly_end_to_end(seeded_con):
    result = generate_briefing(make_context(seeded_con, MockLLM()))
    facts, docs = reference_facts(seeded_con)
    report = evaluate_output(result["output"], result["validation"], facts, docs)
    assert report["numeric_match_rate"] == 1.0 and report["unsupported_claim_rate"] == 0.0


def test_question_set_routes_as_expected():
    questions = load_questions()
    assert len(questions) >= 10
    report = check_routing(questions)
    assert report["tool_usage_accuracy"] == 1.0, [r for r in report["results"] if not r["passed"]]


def test_run_eval_script_exit_codes(tmp_path):
    db = tmp_path / "t.duckdb"
    con = connect(db)
    upsert_df(con, "prices", to_long(synthetic_prices()))
    con.close()
    out = tmp_path / "report.json"
    assert main(["--dry-run", "--db", str(db), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["passed"] is True
