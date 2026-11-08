import logging
from datetime import datetime
from typing import Dict, Optional

import pandas as pd
from sqlalchemy import (
    BigInteger, Column, Date, DateTime, Engine, ForeignKey, Integer,
    MetaData, Numeric, String, Table, create_engine, select
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

logger = logging.getLogger(__name__)


class PostgresDataLoader:
    """
    Felhőalapú PaaS PostgreSQL adatbázis-betöltő osztály kötegelt Upsert
    támogatással (FK4, NFK3, NFK4).
    """

    def __init__(self, connection_uri: str, batch_size: int = 1000) -> None:
        self._connection_uri: str = connection_uri
        self._batch_size: int = batch_size
        self._metadata = MetaData()
        self._define_tables()
        self._db_engine: Engine = self.create_engine_connection()

    def _define_tables(self) -> None:
        """Definiálja a 2.4.5. fejezet ERD modelljének megfelelő SQLAlchemy táblákat."""
        self.instruments_table = Table(
            "instruments",
            self._metadata,
            Column("instrument_id", Integer, primary_key=True, autoincrement=True),
            Column("ticker", String(16), unique=True, nullable=False),
            Column("company_name", String(128)),
            Column("sector", String(64)),
            Column("industry", String(64)),
            Column("pe_ratio", Numeric(10, 4)),
            Column("eps", Numeric(10, 4)),
            Column("market_cap", BigInteger),
            Column("updated_at", DateTime(timezone=True))
        )

        self.daily_metrics_table = Table(
            "daily_metrics",
            self._metadata,
            Column("instrument_id", Integer, ForeignKey("instruments.instrument_id"), primary_key=True),
            Column("trade_date", Date, primary_key=True),
            Column("open_price", Numeric(12, 4), nullable=False),
            Column("high_price", Numeric(12, 4), nullable=False),
            Column("low_price", Numeric(12, 4), nullable=False),
            Column("close_price", Numeric(12, 4), nullable=False),
            Column("adj_close", Numeric(12, 4), nullable=False),
            Column("volume", BigInteger, nullable=False),
            Column("sma_20", Numeric(12, 4)),
            Column("sma_50", Numeric(12, 4)),
            Column("sma_200", Numeric(12, 4)),
            Column("ema_12", Numeric(12, 4)),
            Column("ema_26", Numeric(12, 4)),
            Column("rsi_14", Numeric(8, 4)),
            Column("macd", Numeric(12, 4)),
            Column("macd_signal", Numeric(12, 4)),
            Column("macd_hist", Numeric(12, 4))
        )

        self.audit_log_table = Table(
            "etl_audit_log",
            self._metadata,
            Column("run_id", Integer, primary_key=True, autoincrement=True),
            Column("run_timestamp", DateTime(timezone=True), nullable=False),
            Column("status", String(32), nullable=False),
            Column("rows_processed", Integer, nullable=False),
            Column("execution_time_sec", Numeric(10, 3), nullable=False),
            Column("peak_ram_mb", Numeric(10, 2), nullable=False),
            Column("error_message", String(512))
        )

    def create_engine_connection(self) -> Engine:
        """
        Felépíti az adatbázis-kapcsolatot. PostgreSQL esetén kikényszeríti a
        TLS/SSL titkosítást (NFK4).
        """
        connect_args = {}
        if self._connection_uri.startswith("postgresql") and "sslmode" not in self._connection_uri:
            connect_args["sslmode"] = "require"

        engine = create_engine(
            self._connection_uri,
            pool_pre_ping=True,
            connect_args=connect_args
        )
        self._metadata.create_all(engine)
        return engine

    def upsert_instruments(self, instruments_df: pd.DataFrame) -> int:
        """
        Idempotens módon frissíti vagy beszúrja a vállalati törzsadatokat
        az `instruments` táblába (ON CONFLICT (ticker) DO UPDATE).
        """
        if instruments_df.empty:
            return 0

        records = instruments_df.to_dict(orient="records")
        dialect_name = self._db_engine.dialect.name
        insert_fn = pg_insert if dialect_name == "postgresql" else sqlite_insert

        with self._db_engine.begin() as conn:
            stmt = insert_fn(self.instruments_table).values(records)
            update_cols = {
                col.name: stmt.excluded[col.name]
                for col in self.instruments_table.columns
                if col.name not in ("instrument_id", "ticker")
            }
            upsert_stmt = stmt.on_conflict_do_update(
                index_elements=["ticker"],
                set_=update_cols
            )
            conn.execute(upsert_stmt)

        logger.info(f"Sikeresen szinkronizálva {len(records)} instrumentum törzsadata.")
        return len(records)

    def _get_ticker_id_map(self) -> Dict[str, int]:
        """Lekérdezi a ticker -> instrument_id hozzárendelést az idegen kulcsokhoz."""
        with self._db_engine.connect() as conn:
            rows = conn.execute(
                select(self.instruments_table.c.ticker, self.instruments_table.c.instrument_id)
            ).fetchall()
        return {row[0]: row[1] for row in rows}

    def upsert_daily_metrics(self, metrics_df: pd.DataFrame) -> int:
        """
        Kötegelt (batch) idempotens Upsert a `daily_metrics` ténytáblába
        az `(instrument_id, trade_date)` összetett elsődleges kulcs mentén (FK4).
        """
        if metrics_df.empty:
            return 0

        ticker_map = self._get_ticker_id_map()
        df = metrics_df.copy()
        df["ticker"] = df["ticker"].astype(str)
        df["instrument_id"] = df["ticker"].map(ticker_map)
        df = df.dropna(subset=["instrument_id"]).copy()
        df["instrument_id"] = df["instrument_id"].astype(int)

        target_cols = [col.name for col in self.daily_metrics_table.columns]
        records = df[target_cols].to_dict(orient="records")

        dialect_name = self._db_engine.dialect.name
        insert_fn = pg_insert if dialect_name == "postgresql" else sqlite_insert

        total_upserted = 0
        with self._db_engine.begin() as conn:
            for start_idx in range(0, len(records), self._batch_size):
                batch = records[start_idx:start_idx + self._batch_size]
                stmt = insert_fn(self.daily_metrics_table).values(batch)
                update_cols = {
                    col.name: stmt.excluded[col.name]
                    for col in self.daily_metrics_table.columns
                    if col.name not in ("instrument_id", "trade_date")
                }
                upsert_stmt = stmt.on_conflict_do_update(
                    index_elements=["instrument_id", "trade_date"],
                    set_=update_cols
                )
                conn.execute(upsert_stmt)
                total_upserted += len(batch)

        logger.info(f"Kötegelt Upsert kész: {total_upserted} napi rekord rögzítve.")
        return total_upserted

    def log_audit_event(
        self,
        status: str,
        rows_processed: int,
        execution_time_sec: float,
        peak_ram_mb: float,
        error_message: Optional[str] = None
    ) -> None:
        """Rögzíti az ETL futás KPI metrikáit az `etl_audit_log` táblában."""
        with self._db_engine.begin() as conn:
            conn.execute(
                self.audit_log_table.insert().values(
                    run_timestamp=datetime.utcnow(),
                    status=status,
                    rows_processed=rows_processed,
                    execution_time_sec=round(execution_time_sec, 3),
                    peak_ram_mb=round(peak_ram_mb, 2),
                    error_message=error_message[:512] if error_message else None
                )
            )

    def close_connection(self) -> None:
        """Lezárja az SQLAlchemy kapcsolat-poolt."""
        if self._db_engine is not None:
            self._db_engine.dispose()
            logger.info("Adatbázis-kapcsolat lezárva.")