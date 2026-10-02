"""Offline tests for the agents: mocked LLM, in-memory DuckDB, no network."""
import json
from types import SimpleNamespace

import pytest

from market_intel.agents.facts import flatten
from market_intel.agents.graph import answer_question, generate_briefing, make_context
from market_intel.agents.mock import MockLLM
from market_intel.agents.router import route
from market_intel.agents.briefing import render_markdown
from market_intel.analytics.ticker import ticker_technicals
from market_intel.llm.client import CostTracker, HaikuClient, LLMResult, estimate_cost
from market_intel.retrieval.recent import recent_documents


class ScriptedLLM:
    """Returns a fixed reply and records the prompt it was given."""

    def __init__(self, reply: dict | str):
        self.reply = reply if isinstance(reply, str) else json.dumps(reply)
        self.tracker = CostTracker()
        self.last_user = ""

    def complete(self, system, user, max_tokens):
        self.last_user = user
        return LLMResult(self.reply)


def test_briefing_end_to_end_with_mock(seeded_con):
    result = generate_briefing(make_context(seeded_con, MockLLM()))
    sections = {s["name"]: s["claims"] for s in result["output"]["sections"]}
    assert list(sections) == ["market_regime", "sector_leaders_laggards", "rates_macro",
                              "notable_events", "watchlist_notes", "risks"]
    assert all(c["source_ids"] for claims in sections.values() for c in claims)
    assert result["validation"]["claims_dropped"] == 0
    assert result["validation"]["citation_coverage"] == 1.0
    log = " ".join(result["tool_log"])
    assert all(t in log for t in ("regime:", "sectors:", "rates:", "drilldown:", "documents:"))


def test_validator_drops_invented_and_uncited_claims(seeded_con):
    llm = ScriptedLLM({"sections": [{"name": "market_regime", "claims": [
        {"text": "VIX is calm.", "source_ids": ["regime.vol_regime.vix_level"]},
        {"text": "Made-up number.", "source_ids": ["regime.not.a.real.path"]},
        {"text": "No source at all.", "source_ids": []},
    ]}]})
    result = generate_briefing(make_context(seeded_con, llm))
    assert result["validation"]["claims_kept"] == 1
    assert result["validation"]["claims_dropped"] == 2
    assert result["output"]["sections"][0]["claims"][0]["source_ids"] == ["regime.vol_regime.vix_level"]


def test_unparseable_output_is_reported_not_raised(seeded_con):
    result = generate_briefing(make_context(seeded_con, ScriptedLLM("sorry, no json")))
    assert "error" in result["validation"] and result["output"] == {}


def test_prompt_contains_code_computed_numbers_and_boxes_untrusted_text(seeded_con):
    llm = ScriptedLLM({"sections": []})
    generate_briefing(make_context(seeded_con, llm))
    prompt = llm.last_user
    assert "FACT regime.vol_regime.vix_level = 16.0" in prompt
    assert '<untrusted_document id="sec-0001"' in prompt
    # The planted closing tag inside the filing text must not survive and break out of the box.
    assert prompt.count("</untrusted_document>") == prompt.count("<untrusted_document ")
    assert "IGNORE ALL PREVIOUS INSTRUCTIONS" in prompt  # present only as boxed data


def test_documents_respect_as_of_no_lookahead(seeded_con):
    docs = recent_documents(seeded_con, ["MSFT"], as_of="2026-01-01")
    assert docs == []
    assert recent_documents(seeded_con, ["MSFT"], as_of="2030-06-01")[0]["doc_id"] == "news-0002"


def test_router_picks_tools():
    assert {"sectors", "regime", "documents"} <= set(route("why did tech fall today?")["plan"])
    aapl = route("How is AAPL doing?")
    assert aapl["tickers"] == ["AAPL"] and {"drilldown", "documents"} <= set(aapl["plan"])
    assert "rates" in route("what are treasury yields doing")["plan"]
    assert route("hello")["plan"] == ["regime", "sectors", "documents"]


def test_qa_runs_only_routed_tools_and_cites(seeded_con):
    result = answer_question(make_context(seeded_con, MockLLM()), "How is AAPL doing?")
    log = " ".join(result["tool_log"])
    assert "drilldown:" in log and "rates:" not in log
    assert result["output"]["tools_used"] == result["plan"]
    assert all(c["source_ids"] for c in result["output"]["claims"])
    assert "AAPL" in render_markdown(result["output"], seeded_con) or result["output"]["claims"]


def test_render_links_document_ids(seeded_con):
    out = {"as_of": "x", "sections": [{"name": "notable_events", "claims": [{"text": "Apple.", "source_ids": ["news-0001", "regime.regime"]}]}]}
    md = render_markdown(out, seeded_con)
    assert "[news-0001](https://a.com/1)" in md and "`regime.regime`" in md


def test_flatten_keys_ticker_lists_and_skips_none():
    flat = flatten("s", {"rows": [{"ticker": "XLK", "ret": 1.5, "x": None}], "above": ["A", "B"], "n": 3})
    assert flat == {"s.rows.XLK.ret": 1.5, "s.above": "A,B", "s.n": 3}


def test_ticker_technicals_keys_and_ranges(seeded_con):
    from market_intel.analytics.loaders import load_price_matrix
    t = ticker_technicals(load_price_matrix(seeded_con, ["AAPL", "SPY"]), "AAPL")
    assert 0 <= t["rsi_14"] <= 100 and t["sma_200"] is not None
    json.dumps(t)


def test_haiku_client_caches_and_tracks_cost(tmp_path):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text='{"ok": 1}')],
            usage=SimpleNamespace(input_tokens=1000, output_tokens=500), stop_reason="end_turn",
        )

    client = HaikuClient(cache_dir=tmp_path, client=SimpleNamespace(messages=SimpleNamespace(create=create)))
    first = client.complete("sys", "user", 100)
    second = client.complete("sys", "user", 100)
    assert len(calls) == 1 and calls[0]["model"] == "claude-haiku-4-5-20251001" and calls[0]["max_tokens"] == 100
    assert first.cost_usd == pytest.approx(0.0035) == estimate_cost(1000, 500)
    assert second.from_cache and second.cost_usd == 0
    assert client.tracker.summary()["calls"] == 2 and client.tracker.cache_hits == 1


def test_validator_accepts_fact_label_prefix_but_not_invented_ids(seeded_con):
    llm = ScriptedLLM({"claims": [
        {"text": "VIX is 16.0.", "source_ids": ["FACT regime.vol_regime.vix_level"]},
        {"text": "Invented.", "source_ids": ["FACT regime.nope"]},
    ]})
    result = answer_question(make_context(seeded_con, llm), "what is the vix?")
    assert [c["source_ids"] for c in result["output"]["claims"]] == [["regime.vol_regime.vix_level"]]
    assert result["validation"]["claims_dropped"] == 1
