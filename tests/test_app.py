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
    text = " ".join(m.value for m in app.markdown)
    assert "Market regime" in text and "Data as of" in text


def test_generate_briefing_shows_cited_claims(app):
    app.button(key="gen_brief").click().run()
    assert not app.exception
    text = " ".join(m.value for m in app.markdown)
    assert "Market regime: neutral" in text  # data citations show a readable label and the value
    assert 'title="regime.regime"' in text  # the raw path stays available on hover for auditing
    assert 'href="https://a.com/1"' in text and ">news-0001<" in text  # document citations are links
    assert "claims cited and verified" in text


def test_question_box_runs_routed_tools(app):
    app.text_input(key="question").set_value("How is AAPL doing?").run()
    app.button(key="ask").click().run()
    assert not app.exception
    assert any("How is AAPL doing?" in m.value for m in app.markdown)


def test_haiku_is_default_writer_only_when_key_is_set(app, monkeypatch):
    assert app.radio[0].value.startswith("Mock")  # no key in this fixture
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    with_key = AppTest.from_file(APP, default_timeout=60).run()
    assert with_key.radio[0].value.startswith("Claude")


def test_html_output_escapes_untrusted_text(seeded_con):
    from market_intel.app.components import result_html

    nasty = {"sections": [{"name": "notable_events", "claims": [
        {"text": "<script>alert(1)</script> Apple", "source_ids": ["news-0001", "<img src=x onerror=alert(1)>"]}]}]}
    html = result_html(nasty, seeded_con, {})
    assert "<script>" not in html and "<img" not in html and "&lt;script&gt;" in html


def test_app_survives_missing_sector_data(tmp_path, monkeypatch):
    """A database with no sector ETF prices must show a message, not a traceback."""
    db = tmp_path / "partial.duckdb"
    con = connect(db)
    wide = synthetic_prices()[["SPY", "^VIX"]]
    upsert_df(con, "prices", to_long(wide))
    con.close()
    monkeypatch.setenv("DUCKDB_PATH", str(db))
    from streamlit import cache_data, cache_resource
    cache_data.clear(); cache_resource.clear()
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any("No price data yet" in w.value for w in at.warning)


def test_app_releases_the_database_so_data_refresh_can_write(app, tmp_path):
    """After a page run the database must be writable again (no lingering read lock)."""
    import os

    con = connect(os.environ["DUCKDB_PATH"])  # a write connection; raises if the app still holds a lock
    con.execute("SELECT count(*) FROM prices")
    con.close()
