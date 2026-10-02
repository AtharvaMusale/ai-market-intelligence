"""Offline tests on synthetic data. No network, no real database file."""
import json
import logging

import numpy as np
import pandas as pd
import pytest

from market_intel.analytics.loaders import load_price_matrix
from market_intel.analytics.regime import (
    rates_snapshot,
    regime_summary,
    sector_breadth,
    sector_performance,
    trend_state,
    vol_regime,
)
from market_intel.config import SECTOR_ETFS
from market_intel.db import connect, upsert_df
from market_intel.ingest.prices import normalize_download

N = 300


def make_prices(vix_last: float = 16.0, spy_daily: float = 0.0005) -> pd.DataFrame:
    """Sector i grows at a constant daily rate 0.1% * (i - 5), so ranking is known in advance."""
    idx = pd.bdate_range("2024-01-02", periods=N)
    data = {
        t: 100 * np.cumprod(np.full(N, 1 + 0.001 * (i - 5))) for i, t in enumerate(SECTOR_ETFS)
    }
    data["SPY"] = 100 * (1 + spy_daily) ** np.arange(N)
    vix = np.full(N, 20.0)
    vix[-1] = vix_last
    data["^VIX"] = vix
    data["^TNX"] = np.linspace(4.0, 4.3, N)  # 10-year yield, %
    data["^IRX"] = np.linspace(4.5, 4.4, N)  # 13-week yield, %
    return pd.DataFrame(data, index=idx)


def test_sector_performance_keys_and_sorting():
    result = sector_performance(make_prices())
    assert set(result) == {"as_of", "sectors"}
    assert set(result["sectors"][0]) == {"ticker", "name", "ret_1d_pct", "ret_5d_pct", "ret_21d_pct"}
    assert [r["ticker"] for r in result["sectors"]] == list(reversed(SECTOR_ETFS))
    returns = [r["ret_21d_pct"] for r in result["sectors"]]
    assert returns == sorted(returns, reverse=True)


@pytest.mark.parametrize(
    "vix,label",
    [(12.0, "low"), (14.99, "low"), (15.0, "normal"), (19.99, "normal"),
     (20.0, "elevated"), (29.99, "elevated"), (30.0, "high"), (45.0, "high")],
)
def test_vix_thresholds(vix, label):
    assert vol_regime(make_prices(vix_last=vix))["label"] == label


def test_vol_regime_keys():
    result = vol_regime(make_prices())
    assert {"vix_level", "label", "vix_percentile_1y", "spy_realized_vol_21d_pct"} <= set(result)


def test_high_vix_forces_risk_off_even_in_uptrend():
    summary = regime_summary(make_prices(vix_last=35.0))
    assert summary["trend_state"]["state"] == "uptrend"
    assert summary["regime"] == "risk_off"
    assert summary["override"] is not None


def test_calm_uptrend_is_risk_on_and_normal_vix_is_neutral():
    assert regime_summary(make_prices(vix_last=12.0))["regime"] == "risk_on"
    assert regime_summary(make_prices(vix_last=16.0))["regime"] == "neutral"


def test_trend_and_breadth():
    prices = make_prices()
    assert trend_state(prices)["state"] == "uptrend"
    assert trend_state(make_prices(spy_daily=-0.0005))["state"] == "downtrend"
    breadth = sector_breadth(prices)
    assert breadth["n_total"] == len(SECTOR_ETFS)
    assert breadth["n_above"] == len(breadth["above"])


def test_rates_snapshot_and_json_serializable():
    rates = rates_snapshot(make_prices())
    assert rates["series"]["^TNX"]["latest_pct"] == 4.3
    assert rates["curve_10y_3m_proxy_pp"] == -0.1  # 4.3 - 4.4
    assert rates["curve_10y_3m_proxy_inverted"] is True
    json.dumps(regime_summary(make_prices()))  # raises if not serializable


def test_rates_snapshot_handles_missing_tickers():
    rates = rates_snapshot(make_prices().drop(columns=["^TNX", "^IRX"]))
    assert rates["series"] == {}
    assert rates["curve_10y_3m_proxy_pp"] is None


def _price_rows(closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": "SPY",
            "date": pd.bdate_range("2024-01-02", periods=len(closes)),
            "open": 1.0, "high": 1.0, "low": 1.0, "close": closes,
            "adj_close": closes,
            "volume": 100,
        }
    )


def test_upsert_is_idempotent_and_replaces():
    con = connect(":memory:")
    df = _price_rows([4.0, 4.1])
    upsert_df(con, "prices", df)
    upsert_df(con, "prices", df)
    assert con.execute("SELECT count(*) FROM prices").fetchone()[0] == 2

    df.loc[0, "adj_close"] = 9.9
    upsert_df(con, "prices", df)
    assert con.execute("SELECT count(*) FROM prices").fetchone()[0] == 2
    assert con.execute("SELECT adj_close FROM prices WHERE date = '2024-01-02'").fetchone()[0] == 9.9


def test_upsert_rejects_unknown_table():
    with pytest.raises(ValueError):
        upsert_df(connect(":memory:"), "nope; DROP TABLE prices", pd.DataFrame())


def test_loader_respects_as_of():
    con = connect(":memory:")
    upsert_df(con, "prices", _price_rows([1.0, 2.0, 3.0, 4.0, 5.0]))
    wide = load_price_matrix(con, ["SPY"], as_of="2024-01-04")
    assert wide.index.max() == pd.Timestamp("2024-01-04")
    assert wide["SPY"].iloc[-1] == 3.0


def test_normalize_download_warns_on_empty_ticker(caplog):
    idx = pd.bdate_range("2024-01-02", periods=3, name="Date")
    fields = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    cols = pd.MultiIndex.from_product([["SPY"], fields])
    raw = pd.DataFrame(np.ones((3, len(fields))), index=idx, columns=cols)
    with caplog.at_level(logging.WARNING):
        out = normalize_download(raw, ["SPY", "VGT"])
    assert set(out["ticker"]) == {"SPY"}
    assert "No price data returned for VGT" in caplog.text
