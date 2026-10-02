"""Offline tests for text ingestion and retrieval: no network, no Pinecone, in-memory DuckDB."""
from datetime import date, datetime

import pandas as pd
import pytest
from pinecone.models.vectors.search import Hit

from market_intel.db import connect, upsert_df
from market_intel.ingest.filings import extract_mdna, html_to_text, parse_submissions
from market_intel.ingest.gdelt import parse_articles
from market_intel.retrieval.chunking import chunk_id, chunk_text
from market_intel.retrieval.indexer import sync_to_pinecone
from market_intel.retrieval.store import UPSERT_BATCH, PineconeStore, build_filter

NOW = datetime(2026, 10, 2, 12, 0)


def test_chunk_text_overlaps_and_covers_text():
    text = " ".join(f"word{i}" for i in range(400))
    chunks = chunk_text(text, size=300, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)
    assert chunks[0].startswith("word0") and chunks[-1].endswith("word399")
    tail_words = set(chunks[0].split()[-5:])
    assert tail_words & set(chunks[1].split()[:15])  # consecutive chunks share words


def test_chunk_text_empty_and_ids_are_deterministic():
    assert chunk_text("   ") == []
    assert chunk_id("sec-0001", 3) == "sec-0001:3"


def test_parse_articles_keeps_titles_with_keyword_and_stores_no_body():
    payload = {
        "articles": [
            {"url": "https://a.com/1", "title": "Apple shares rise", "seendate": "20260930T123000Z", "domain": "a.com"},
            {"url": "https://b.com/2", "title": "Unrelated fruit news", "seendate": "20260930T130000Z", "domain": "b.com"},
            {"url": "https://c.com/3", "title": "", "seendate": "20260930T130000Z"},
        ]
    }
    df = parse_articles(payload, "AAPL", "Apple", NOW)
    assert list(df["title"]) == ["Apple shares rise"]
    row = df.iloc[0]
    assert row["text"] == row["title"]  # headline only
    assert row["source"] == "gdelt:a.com"
    assert row["published_at"] == datetime(2026, 9, 30, 12, 30)
    assert row["doc_id"].startswith("news-")


def test_parse_submissions_filters_form_and_date():
    payload = {
        "filings": {
            "recent": {
                "form": ["8-K", "4", "10-Q", "8-K"],
                "filingDate": ["2026-09-30", "2026-09-29", "2026-08-01", "2025-01-01"],
                "accessionNumber": ["0001-26-000001", "0001-26-000002", "0001-26-000003", "0001-25-000004"],
                "primaryDocument": ["a.htm", "b.xml", "c.htm", "d.htm"],
                "items": ["2.02,9.01", "", "", ""],
            }
        }
    }
    filings = parse_submissions(payload, 320193, "AAPL", since=date(2026, 7, 1))
    assert [f["form"] for f in filings] == ["8-K", "10-Q"]
    assert filings[0]["doc_id"] == "sec-0001-26-000001"
    assert filings[0]["url"] == "https://www.sec.gov/Archives/edgar/data/320193/000126000001/a.htm"
    assert "items 2.02,9.01" in filings[0]["title"]


def test_html_to_text_skips_scripts_and_xbrl_header():
    html = "<html><script>evil()</script><ix:header>hidden</ix:header><p>Revenue  grew</p><p>10%</p></html>"
    assert html_to_text(html) == "Revenue grew 10%"


def test_extract_mdna_prefers_last_heading():
    text = "TOC Management's Discussion and Analysis page 5. " + "x " * 50 + "Management's Discussion and Analysis Real content here"
    assert extract_mdna(text, 100).startswith("Management's Discussion and Analysis Real content")


def test_build_filter():
    assert build_filter() == {}
    flt = build_filter("AAPL", date(2026, 9, 1), date(2026, 9, 30))
    assert flt == {"ticker": {"$eq": "AAPL"}, "date_int": {"$gte": 20260901, "$lte": 20260930}}


class FakeIndex:
    """Stands in for a Pinecone index and records what it was sent."""

    def __init__(self):
        self.batches, self.search_kwargs = [], None

    def upsert_records(self, namespace, records):
        self.batches.append(records)

    def search(self, **kwargs):
        self.search_kwargs = kwargs
        # The SDK's own Hit class, so this test breaks if we read hits the wrong way.
        hit = Hit(id_="news-1:0", score_=0.9, fields={"chunk_text": "Apple shares rise", "ticker": "AAPL", "url": "https://a.com/1"})
        return type("Resp", (), {"result": type("Res", (), {"hits": [hit]})()})()


def _seed_documents(con, n_news=3):
    rows = [
        {
            "doc_id": f"news-{i}", "ticker": "AAPL", "source": "gdelt:a.com", "doc_type": "news",
            "title": f"Apple story {i}", "url": f"https://a.com/{i}",
            "published_at": datetime(2026, 9, 30, 12, 0), "fetched_at": NOW, "text": f"Apple story {i}",
        }
        for i in range(n_news)
    ]
    upsert_df(con, "documents", pd.DataFrame(rows))


def test_sync_pushes_once_then_nothing_and_uses_deterministic_ids():
    con = connect(":memory:")
    _seed_documents(con)
    index = FakeIndex()
    store = PineconeStore(index)

    first = sync_to_pinecone(con, store)
    second = sync_to_pinecone(con, store)

    assert first["pushed"] == 3 and second["pushed"] == 0
    sent = [r for batch in index.batches for r in batch]
    assert sorted(r["_id"] for r in sent) == ["news-0:0", "news-1:0", "news-2:0"]
    assert sent[0]["date_int"] == 20260930 and sent[0]["url"].startswith("https://a.com/")


def test_sync_dry_run_sends_nothing_and_budget_guard_trips():
    con = connect(":memory:")
    _seed_documents(con)
    index = FakeIndex()
    result = sync_to_pinecone(con, PineconeStore(index), dry_run=True)
    assert result["pending_chunks"] == 3 and result["pushed"] == 0 and index.batches == []
    with pytest.raises(RuntimeError):
        sync_to_pinecone(con, PineconeStore(index), token_budget=1)


def test_upsert_batches_at_96():
    index = FakeIndex()
    PineconeStore(index).upsert([{"_id": str(i), "chunk_text": "x"} for i in range(UPSERT_BATCH + 4)])
    assert [len(b) for b in index.batches] == [UPSERT_BATCH, 4]


def test_search_applies_filter_and_returns_source_url():
    index = FakeIndex()
    hits = PineconeStore(index).search("iphone demand", ticker="AAPL", date_from=date(2026, 9, 1))
    assert index.search_kwargs["filter"] == {"ticker": {"$eq": "AAPL"}, "date_int": {"$gte": 20260901}}
    assert index.search_kwargs["inputs"] == {"text": "iphone demand"}
    assert hits[0]["url"] == "https://a.com/1" and hits[0]["score"] == 0.9


def test_gdelt_retries_then_succeeds():
    import requests
    from market_intel.ingest.gdelt import _get_with_retry

    class Flaky:
        calls = 0

        def get(self, url, **kwargs):
            Flaky.calls += 1
            if Flaky.calls < 3:
                raise requests.HTTPError("429")
            return "ok"

    waits = []
    assert _get_with_retry(Flaky(), {}, sleep=waits.append) == "ok"
    assert len(waits) == 2


def test_ingest_news_makes_one_request_and_assigns_tickers():
    from market_intel.ingest.gdelt import ingest_news

    class OneShot:
        calls = 0

        def get(self, url, **kwargs):
            OneShot.calls += 1
            payload = {"articles": [
                {"url": "https://a.com/1", "title": "Apple shares rise", "seendate": "20260930T123000Z", "domain": "a.com"},
                {"url": "https://b.com/2", "title": "Microsoft earnings beat", "seendate": "20260930T130000Z", "domain": "b.com"},
            ]}
            return type("R", (), {"json": lambda self: payload})()

    con = connect(":memory:")
    added = ingest_news(con, {"AAPL": "Apple", "MSFT": "Microsoft"}, session=OneShot())
    assert OneShot.calls == 1 and added == 2
    assert dict(con.execute("SELECT ticker, count(*) FROM documents GROUP BY 1").fetchall()) == {"AAPL": 1, "MSFT": 1}
    assert ingest_news(con, {"AAPL": "Apple", "MSFT": "Microsoft"}, session=OneShot()) == 0  # re-run adds nothing
