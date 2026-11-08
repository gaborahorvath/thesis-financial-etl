import numpy as np
import pandas as pd
import pytest

from src.transformer import FinancialDataTransformer
from src.loader import PostgresDataLoader


@pytest.fixture
def sample_raw_ohlcv() -> pd.DataFrame:
    """Determinisztikus tesztadathalmaz érvényes és szándékosan hibás sorokkal."""
    dates = pd.bdate_range(start="2026-01-01", periods=40)
    prices = [100.0 + i * 1.5 for i in range(40)]
    df = pd.DataFrame({
        "ticker": ["AAPL"] * 40,
        "trade_date": dates,
        "open_price": prices,
        "high_price": [p + 2.0 for p in prices],
        "low_price": [p - 1.5 for p in prices],
        "close_price": [p + 0.5 for p in prices],
        "adj_close": [p + 0.5 for p in prices],
        "volume": [10_000_000 + i * 1000 for i in range(40)]
    })
    # Szándékos anomáliák hozzáadása:
    # 1. Duplikált kereskedési nap
    dup_row = df.iloc[[0]].copy()
    # 2. Negatív árfolyam anomália
    neg_row = df.iloc[[1]].copy()
    neg_row["close_price"] = -50.0
    # 3. Hiányzó (NaN) árfolyam
    nan_row = df.iloc[[2]].copy()
    nan_row["open_price"] = np.nan

    return pd.concat([df, dup_row, neg_row, nan_row], ignore_index=True)


def test_clean_ohlcv_removes_anomalies_and_duplicates(sample_raw_ohlcv: pd.DataFrame) -> None:
    """TC-01: Ellenőrzi a duplikátumok, NaN és negatív árak kiszűrését (FK2, NFK3)."""
    transformer = FinancialDataTransformer()
    cleaned = transformer.clean_ohlcv_data(sample_raw_ohlcv)

    assert len(cleaned) == 40
    assert cleaned.isna().sum().sum() == 0
    assert (cleaned["close_price"] > 0).all()
    assert cleaned.duplicated(subset=["ticker", "trade_date"]).sum() == 0


def test_compute_sma_ema_mathematical_accuracy(sample_raw_ohlcv: pd.DataFrame) -> None:
    """TC-02: Ellenőrzi az SMA_20, SMA_50, SMA_200 és EMA_12/26 pontosságát (FK3)."""
    transformer = FinancialDataTransformer()
    cleaned = transformer.clean_ohlcv_data(sample_raw_ohlcv)
    result = transformer.compute_sma_ema(cleaned)

    expected_sma20 = cleaned["close_price"].iloc[:20].mean()
    assert np.isclose(result["sma_20"].iloc[19], expected_sma20, atol=1e-4)
    assert all(col in result.columns for col in ["sma_20", "sma_50", "sma_200", "ema_12", "ema_26"])


def test_compute_rsi_bounds_and_uptrend_behavior(sample_raw_ohlcv: pd.DataFrame) -> None:
    """TC-03: Ellenőrzi, hogy az RSI_14 a [0, 100] tartományban marad és trendkövető (FK3)."""
    transformer = FinancialDataTransformer(rsi_period=14)
    cleaned = transformer.clean_ohlcv_data(sample_raw_ohlcv)
    rsi_series = transformer.compute_rsi(cleaned)

    assert rsi_series.between(0.0, 100.0).all()
    # Szigorúan növekvő árfolyamnál az RSI értéke 70 feletti (túlvett)
    assert rsi_series.iloc[-1] > 70.0


def test_compute_macd_algebraic_identity(sample_raw_ohlcv: pd.DataFrame) -> None:
    """TC-04: Ellenőrzi a MACD = EMA_12 - EMA_26 és Hist = MACD - Signal azonosságokat."""
    transformer = FinancialDataTransformer()
    cleaned = transformer.clean_ohlcv_data(sample_raw_ohlcv)
    with_ma = transformer.compute_sma_ema(cleaned)
    result = transformer.compute_macd(with_ma)

    diff_macd = np.abs(result["macd"] - (result["ema_12"] - result["ema_26"])).max()
    diff_hist = np.abs(result["macd_hist"] - (result["macd"] - result["macd_signal"])).max()
    assert diff_macd <= 1e-3
    assert diff_hist <= 1e-3


def test_clean_fundamentals_handles_nulls_and_types() -> None:
    """TC-05: Ellenőrzi a vállalati fundamentális adatok típuskényszerítését."""
    transformer = FinancialDataTransformer()
    raw_fund = pd.DataFrame([
        {
            "ticker": " aapl ",
            "company_name": "Apple Inc.",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "pe_ratio": "28.45678",
            "eps": "None",
            "market_cap": "3200000000000",
            "updated_at": "2026-10-30T22:00:00Z"
        }
    ])
    cleaned = transformer.clean_fundamentals(raw_fund)
    assert cleaned.loc[0, "ticker"] == "AAPL"
    assert np.isclose(cleaned.loc[0, "pe_ratio"], 28.4568)
    assert pd.isna(cleaned.loc[0, "eps"])
    assert cleaned.loc[0, "market_cap"] == 3_200_000_000_000


def test_idempotent_upsert_prevents_duplicates(sample_raw_ohlcv: pd.DataFrame, tmp_path) -> None:
    """TC-06: Ellenőrzi, hogy a kétszer egymás után futtatott Upsert nem duplikál sorokat (FK4, H3)."""
    db_file = tmp_path / "test_idempotency.db"
    loader = PostgresDataLoader(connection_uri=f"sqlite:///{db_file}", batch_size=20)
    transformer = FinancialDataTransformer()

    fund_df = transformer.clean_fundamentals(pd.DataFrame([{
        "ticker": "AAPL", "company_name": "Apple Inc.", "sector": "Technology",
        "industry": "Hardware", "pe_ratio": 25.0, "eps": 6.5,
        "market_cap": 3000000000000, "updated_at": "2026-10-30T22:00:00Z"
    }]))
    metrics_df = transformer.transform_all(sample_raw_ohlcv)

    loader.upsert_instruments(fund_df)
    first_count = loader.upsert_daily_metrics(metrics_df)
    second_count = loader.upsert_daily_metrics(metrics_df)

    with loader._db_engine.connect() as conn:
        total_in_db = conn.execute(loader.daily_metrics_table.select()).fetchall()

    loader.close_connection()
    assert first_count == 40
    assert second_count == 40
    assert len(total_in_db) == 40