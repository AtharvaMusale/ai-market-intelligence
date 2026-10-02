"""Headless smoke test of the Streamlit app (mock writer, temp database, no network)."""
from pathlib import Path

import pandas as pd
import pytest
from conftest import synthetic_prices, to_long
from streamlit.testing.v1 import AppTest

from market_intel.db import connect, upsert_df

APP = str(Path(__file__).resolve().parents[1] / "src" / "market_intel" / "app" / "streamlit_app.py")


@pytest.fixture
def app(tmp_path, monkeypatch):
    db = tmp_path / "app.duckdb"
    con = connect(db)
    wide = synthetic_prices()
    upsert_df(con, "prices", to_long(wide))
    last = wide.index[-1].to_pydatetime()
    upsert_df(con, "documents", pd.DataFrame([{
        "doc_id": "news-0001", "ticker": "AAPL", "source": "gdelt:a.com", "doc_type": "news",
        "title": "Apple shares rise", "url": "https://a.com/1", "published_at": last,
        "fetched_at": last, "text": "Apple shares rise"}]))
    con.close()
    monkeypatch.setenv("DUCKDB_PATH", str(db))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from streamlit import cache_data, cache_resource
    cache_data.clear(); cache_resource.clear()
    return AppTest.from_file(APP, default_timeout=60).run()


def test_app_renders_panels_without_exception(app):
    assert not app.exception
    assert [t.label for t in app.tabs] == ["Market", "Daily brief", "Ask a question", "Evaluation"]
    assert any("Regime" == m.label for m in app.metric)


def test_generate_briefing_shows_cited_claims(app):
    app.button(key="gen_brief").click().run()
    assert not app.exception
    text = " ".join(m.value for m in app.markdown)
    assert "Daily market briefing" in text and "regime.regime = " in text  # data citations show their values
    assert "[news-0001](https://a.com/1)" in text  # document citations are links


def test_question_box_runs_routed_tools(app):
    app.text_input(key="question").set_value("How is AAPL doing?").run()
    app.button(key="ask").click().run()
    assert not app.exception
    assert any("How is AAPL doing?" in m.value for m in app.markdown)
