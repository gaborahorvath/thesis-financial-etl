import logging
import os
import sys
import time
import psutil

from src.extractor import MarketDataExtractor
from src.transformer import FinancialDataTransformer
from src.loader import PostgresDataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
logger = logging.getLogger("ETL_Pipeline_Main")


def run_etl_pipeline() -> None:
    """Végrehajtja a teljes Extract -> Transform -> Load munkafolyamatot."""
    start_time = time.perf_counter()
    process = psutil.Process(os.getpid())

    api_key = os.environ.get("ALPHAVANTAGE_API_KEY", "DEMO")
    # Ha nincs megadva felhős SUPABASE_DB_URI, lokális SQLite adatbázissal tesztelhető PyCharmban
    db_uri = os.environ.get("SUPABASE_DB_URI", "sqlite:///local_portfolio_test.db")

    loader = PostgresDataLoader(connection_uri=db_uri, batch_size=1000)

    try:
        logger.info("=== [1/3 EXTRACT] Piaci és fundamentális adatok lekérése ===")
        extractor = MarketDataExtractor(api_key=api_key)
        raw_fundamentals_df = extractor.extract_company_fundamentals()
        raw_ohlcv_df = extractor.extract_ohlcv_history(period="2y")

        logger.info("=== [2/3 TRANSFORM] Adattisztítás és SMA/EMA/RSI/MACD számítás ===")
        transformer = FinancialDataTransformer(rsi_period=14, macd_params=(12, 26, 9))
        clean_fundamentals_df = transformer.clean_fundamentals(raw_fundamentals_df)
        metrics_df = transformer.transform_all(raw_ohlcv_df)

        logger.info("=== [3/3 LOAD] Kötegelt idempotens Upsert az adatbázisba ===")
        loader.upsert_instruments(clean_fundamentals_df)
        rows_loaded = loader.upsert_daily_metrics(metrics_df)

        elapsed_sec = time.perf_counter() - start_time
        peak_ram_mb = process.memory_info().rss / (1024 ** 2)

        loader.log_audit_event(
            status="SUCCESS",
            rows_processed=rows_loaded,
            execution_time_sec=elapsed_sec,
            peak_ram_mb=peak_ram_mb
        )
        logger.info(
            f"=== ETL PIPELINE SIKERES | Rekordok: {rows_loaded} | "
            f"Futási idő: {elapsed_sec:.2f} mp (KPI-1 <= 180s) | "
            f"Csúcs RAM: {peak_ram_mb:.1f} MB (KPI-2 <= 250MB) ==="
        )

    except Exception as exc:
        elapsed_sec = time.perf_counter() - start_time
        peak_ram_mb = process.memory_info().rss / (1024 ** 2)
        logger.error(f"ETL Pipeline kritikus hiba: {exc}")
        loader.log_audit_event(
            status="FAILED",
            rows_processed=0,
            execution_time_sec=elapsed_sec,
            peak_ram_mb=peak_ram_mb,
            error_message=str(exc)
        )
        sys.exit(1)
    finally:
        loader.close_connection()


if __name__ == "__main__":
    run_etl_pipeline()