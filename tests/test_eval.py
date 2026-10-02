import pytest
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
    v = facts["sectors.VGT.ret_21d_pct"]
    assert verify_claim(claim(f"VGT {round(v, 2)}%.", "sectors.VGT.ret_21d_pct"), facts, docs)["all_matched"]
    assert not verify_claim(claim(f"VGT {v + 0.5}%.", "sectors.VGT.ret_21d_pct"), facts, docs)["all_matched"]


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


def test_index_names_and_written_out_dates_are_not_false_alarms(seeded_con):
    from market_intel.eval.claims import iso_dates_from_text

    assert [t.value for t in extract_numbers("S&P 500 closed at 769.1; the Nasdaq 100 and Russell 2000 lagged")] == [769.1]
    assert iso_dates_from_text("Date of earliest event reported: September 2, 2026") == ["2026-09-02"]
    from datetime import datetime

    import pandas as pd

    from market_intel.db import upsert_df

    upsert_df(seeded_con, "documents", pd.DataFrame([{
        "doc_id": "sec-test", "ticker": "NVDA", "source": "sec-edgar", "doc_type": "8-K", "title": "NVDA 8-K",
        "url": "https://sec.gov/x", "published_at": datetime(2026, 9, 3), "fetched_at": datetime(2026, 9, 3),
        "text": "Date of earliest event reported: September 2, 2026"}]))
    facts, docs = reference_facts(seeded_con)
    assert verify_claim(claim("Announced 2026-09-02.", "sec-test"), facts, docs)["all_matched"]
    assert not verify_claim(claim("Announced 2026-09-05.", "sec-test"), facts, docs)["all_matched"]  # a wrong date still fails


def test_distance_from_200_day_average_is_computed_in_code(seeded_con):
    from market_intel.analytics.loaders import load_price_matrix
    from market_intel.analytics.ticker import ticker_technicals

    t = ticker_technicals(load_price_matrix(seeded_con, ["AAPL", "SPY"]), "AAPL")
    assert round((t["last"] / t["sma_200"] - 1) * 100, 1) == pytest.approx(t["pct_vs_sma_200"], abs=0.2)


def _facts():
    return {"drilldown.AAPL.trend": "downtrend", "drilldown.AAPL.rsi_zone": "neutral", "drilldown.AAPL.vs_spy_21d": "underperforming SPY",
            "drilldown.AAPL.verdict": "uptrend, outperforming SPY over 21 days, RSI overbought", "drilldown.AAPL.drawdown_from_52w_high_pct": -5.0,
            "drilldown.AAPL.ret_21d_pct": -9.0, "drilldown.AAPL.ret_21d_vs_spy_pct": -3.2, "drilldown.AAPL.pct_vs_sma_200": 14.99,
            "drilldown.AAPL.sma_200": 288.79, "regime.components.trend": 1}


def test_direction_check_catches_wrong_sign_but_not_level_facts_or_clause_order():
    f = _facts()
    wrong = verify_claim(claim("AAPL is up 9% over 21 days.", "drilldown.AAPL.ret_21d_pct"), f, {})
    right = verify_claim(claim("AAPL is down 9% over 21 days.", "drilldown.AAPL.ret_21d_pct"), f, {})
    assert not wrong["all_matched"] and wrong["direction"][0]["said"] == "up" and right["all_matched"]
    # A level fact: "below its 200-day average of 288.79" describes position, not the sign of 288.79.
    level = verify_claim(claim("AAPL fell below its 200-day average of 288.79.", "drilldown.AAPL.sma_200"), f, {})
    assert level["all_matched"] and level["direction"] == []
    # The direction word after the number wins, and a clause break stops the look-back.
    two = verify_claim(claim("AAPL trades 14.99% above its 200-day average and 5.0% below its 52-week high.",
                             "drilldown.AAPL.pct_vs_sma_200", "drilldown.AAPL.drawdown_from_52w_high_pct"), f, {})
    assert two["all_matched"] and len(two["direction"]) == 2


def test_interpretation_rules():
    f = _facts()
    bad_trend = verify_claim(claim("AAPL is in an uptrend.", "drilldown.AAPL.trend"), f, {})
    assert any("uptrend" in i for i in bad_trend["interpretation"])
    uncited = verify_claim(claim("AAPL is in a downtrend.", "drilldown.AAPL.ret_21d_pct"), f, {})
    assert any("without citing a trend fact" in i for i in uncited["interpretation"])
    assert not verify_claim(claim("AAPL is in a downtrend.", "drilldown.AAPL.trend"), f, {})["interpretation"]
    # the score component is not a trend label
    assert verify_claim(claim("Combining an uptrend.", "regime.components.trend"), f, {})["interpretation"]
    # the verdict fact states RSI zone and relative performance, so quoting it is valid evidence
    ok = verify_claim(claim("AAPL: uptrend, outperforming SPY, RSI overbought.", "drilldown.AAPL.verdict"), f, {})
    assert ok["interpretation"] == []
    assert any("oversold" in i for i in verify_claim(claim("RSI shows oversold conditions.", "drilldown.AAPL.rsi_zone"), f, {})["interpretation"])
    assert any("52-week high" in i for i in verify_claim(claim("AAPL is at its 52-week high.", "drilldown.AAPL.drawdown_from_52w_high_pct"), f, {})["interpretation"])
    assert any("underperform" in i for i in verify_claim(claim("AAPL is outperforming SPY.", "drilldown.AAPL.vs_spy_21d"), f, {})["interpretation"])


def test_new_metrics_appear_in_the_report(seeded_con):
    facts, docs = reference_facts(seeded_con)
    out = {"claims": [claim("AAPL is up 9%.", "drilldown.AAPL.ret_21d_pct")]}
    facts["drilldown.AAPL.ret_21d_pct"] = -9.0
    rep = evaluate_output(out, {"claims_dropped": 0}, facts, docs)
    assert rep["direction_checked"] == 1 and rep["direction_errors"] == 1 and rep["unsupported_claim_rate"] == 1.0
    assert "direction: said up, data is down" in rep["failures"][0]["unmatched"][0]


def test_history_replay_is_point_in_time_and_refuses_unconfirmed_spend(tmp_path, capsys):
    from market_intel.scripts.run_eval_history import main as history_main, replay_dates, summarize

    db = tmp_path / "h.duckdb"
    con = connect(db)
    upsert_df(con, "prices", to_long(synthetic_prices()))
    con.close()
    out = tmp_path / "history.json"
    assert history_main(["--days", "5", "--db", str(db), "--out", str(out)]) == 0  # mock writer, free
    report = json.loads(out.read_text())
    assert report["writer"] == "mock" and len(report["dates"]) == 5
    assert report["dates"] == sorted(report["dates"], key=lambda r: r["as_of"])
    assert report["summary"]["dates_below_threshold"] == 0
    # asking for Haiku without --confirm-cost must stop before any call and return a distinct exit code
    assert history_main(["--days", "5", "--haiku", "--db", str(db), "--out", str(out)]) == 2
    assert "Refusing to call Haiku" in capsys.readouterr().out
    assert summarize([{"numeric_match_rate": 0.9, "direction_errors": 1, "direction_checked": 4,
                       "interpretation_issues": 2, "citation_coverage": 1.0}], 0.95)["dates_below_threshold"] == 1
