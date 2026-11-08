-- 1. Vállalati törzs- és fundamentális dimenziótábla
CREATE TABLE IF NOT EXISTS public.instruments (
    instrument_id SERIAL PRIMARY KEY,
    ticker VARCHAR(16) NOT NULL UNIQUE,
    company_name VARCHAR(128),
    sector VARCHAR(64),
    industry VARCHAR(64),
    pe_ratio NUMERIC(10, 4),
    eps NUMERIC(10, 4),
    market_cap BIGINT,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 2. Napi OHLCV árfolyamok és technikai indikátorok ténytáblája
CREATE TABLE IF NOT EXISTS public.daily_metrics (
    instrument_id INTEGER NOT NULL REFERENCES public.instruments(instrument_id) ON DELETE CASCADE,
    trade_date DATE NOT NULL,
    open_price NUMERIC(12, 4) NOT NULL CHECK (open_price > 0),
    high_price NUMERIC(12, 4) NOT NULL CHECK (high_price > 0),
    low_price NUMERIC(12, 4) NOT NULL CHECK (low_price > 0),
    close_price NUMERIC(12, 4) NOT NULL CHECK (close_price > 0),
    adj_close NUMERIC(12, 4) NOT NULL CHECK (adj_close > 0),
    volume BIGINT NOT NULL CHECK (volume >= 0),
    sma_20 NUMERIC(12, 4),
    sma_50 NUMERIC(12, 4),
    sma_200 NUMERIC(12, 4),
    ema_12 NUMERIC(12, 4),
    ema_26 NUMERIC(12, 4),
    rsi_14 NUMERIC(8, 4) CHECK (rsi_14 BETWEEN 0 AND 100),
    macd NUMERIC(12, 4),
    macd_signal NUMERIC(12, 4),
    macd_hist NUMERIC(12, 4),
    PRIMARY KEY (instrument_id, trade_date)
);

-- B-fa index az időtartomány-alapú BI lekérdezések gyorsítására
CREATE INDEX IF NOT EXISTS idx_daily_metrics_date
    ON public.daily_metrics (trade_date DESC, instrument_id);

-- 3. Technikai auditnapló tábla (KPI-1, KPI-2, KPI-3 monitorozásához)
CREATE TABLE IF NOT EXISTS public.etl_audit_log (
    run_id SERIAL PRIMARY KEY,
    run_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    status VARCHAR(32) NOT NULL,
    rows_processed INTEGER NOT NULL,
    execution_time_sec NUMERIC(10, 3) NOT NULL,
    peak_ram_mb NUMERIC(10, 2) NOT NULL,
    error_message VARCHAR(512)
);

-- 4. Szerepköralapú hozzáférés-szabályozás (RBAC - NFK4 követelmény)
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'etl_writer_role') THEN
        CREATE ROLE etl_writer_role NOLOGIN;
    END IF;
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'bi_reader_role') THEN
        CREATE ROLE bi_reader_role NOLOGIN;
    END IF;
END
$$;

GRANT USAGE ON SCHEMA public TO etl_writer_role, bi_reader_role;
GRANT SELECT, INSERT, UPDATE ON public.instruments, public.daily_metrics, public.etl_audit_log TO etl_writer_role;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO etl_writer_role;

-- A Power BI szerepkör kizárólag olvasási (SELECT) jogosultságot kap
GRANT SELECT ON public.instruments, public.daily_metrics, public.etl_audit_log TO bi_reader_role;