import logging
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FinancialDataTransformer:
    """
    Állapotmentes adattisztító és technikai indikátor-számító osztály (FK2 és FK3).
    """

    def __init__(
        self,
        rsi_period: int = 14,
        macd_params: Tuple[int, int, int] = (12, 26, 9)
    ) -> None:
        self.rsi_period: int = rsi_period
        self.macd_params: Tuple[int, int, int] = macd_params

    def clean_fundamentals(self, raw_fund_df: pd.DataFrame) -> pd.DataFrame:
        """
        Tisztítja és normalizálja a vállalati fundamentális törzsadatokat
        az `instruments` adatbázis-tábla sémájának megfelelően.
        """
        if raw_fund_df is None or raw_fund_df.empty:
            raise ValueError("A bemeneti fundamentális DataFrame üres.")

        df = raw_fund_df.copy()
        df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
        df = df.dropna(subset=["ticker"]).drop_duplicates(subset=["ticker"], keep="last")

        # Alpha Vantage "None" vagy "-" szövegek konvertálása valós numerikus értékre
        for num_col in ["pe_ratio", "eps"]:
            df[num_col] = pd.to_numeric(df[num_col], errors="coerce").round(4)

        df["market_cap"] = (
            pd.to_numeric(df["market_cap"], errors="coerce")
            .fillna(0)
            .astype("int64")
        )
        df["company_name"] = df["company_name"].fillna("Unknown").astype(str).str.slice(0, 128)
        df["sector"] = df["sector"].fillna("Unclassified").astype(str).str.slice(0, 64)
        df["industry"] = df["industry"].fillna("Unclassified").astype(str).str.slice(0, 64)
        df["updated_at"] = pd.to_datetime(df["updated_at"], utc=True)

        return df[[
            "ticker", "company_name", "sector", "industry",
            "pe_ratio", "eps", "market_cap", "updated_at"
        ]].reset_index(drop=True)

    def clean_ohlcv_data(self, raw_price_df: pd.DataFrame) -> pd.DataFrame:
        """
        Nyers OHLCV idősorok tisztítása, duplikátumszűrése, érvénytelen (<=0) árak
        kizárása és memóriabarát adattípusok kikényszerítése (FK2, NFK2, NFK3).
        """
        if raw_price_df is None or raw_price_df.empty:
            raise ValueError("A bemeneti OHLCV DataFrame üres.")

        df = raw_price_df.copy()

        # Ticker és dátum normalizálás (YYYY-MM-DD)
        df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
        df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.date

        price_cols = ["open_price", "high_price", "low_price", "close_price", "adj_close"]
        for col in price_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

        # Hiányzó kulcsmezők vagy definiálatlan árak szűrése
        df = df.dropna(subset=["ticker", "trade_date"] + price_cols + ["volume"])

        # Tartományvalidáció: kizárólag pozitív árfolyamok és nemnegatív forgalom maradhat
        valid_mask = (
            (df["open_price"] > 0)
            & (df["high_price"] > 0)
            & (df["low_price"] > 0)
            & (df["close_price"] > 0)
            & (df["adj_close"] > 0)
            & (df["high_price"] >= df["low_price"])
            & (df["volume"] >= 0)
        )
        df = df.loc[valid_mask].copy()

        # Duplikátumok eltávolítása és időrendi rendezés
        df = (
            df.drop_duplicates(subset=["ticker", "trade_date"], keep="last")
            .sort_values(by=["ticker", "trade_date"], ascending=[True, True])
            .reset_index(drop=True)
        )

        # Memóriaoptimalizált adattípusok beállítása
        df["ticker"] = df["ticker"].astype("category")
        for col in price_cols:
            df[col] = df[col].astype("float64").round(4)
        df["volume"] = df["volume"].astype("int64")

        return df

    def compute_sma_ema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Kiszámítja a 20, 50, 200 napos egyszerű (SMA) és a 12, 26 napos exponenciális
        (EMA) mozgóátlagokat vektorizált csúszóablakos műveletekkel (FK3).
        """
        result = df.copy()
        grouped_close = result.groupby("ticker", observed=True)["close_price"]

        result["sma_20"] = grouped_close.transform(
            lambda s: s.rolling(window=20, min_periods=1).mean()
        ).round(4)
        result["sma_50"] = grouped_close.transform(
            lambda s: s.rolling(window=50, min_periods=1).mean()
        ).round(4)
        result["sma_200"] = grouped_close.transform(
            lambda s: s.rolling(window=200, min_periods=1).mean()
        ).round(4)

        result["ema_12"] = grouped_close.transform(
            lambda s: s.ewm(span=12, adjust=False).mean()
        ).round(4)
        result["ema_26"] = grouped_close.transform(
            lambda s: s.ewm(span=26, adjust=False).mean()
        ).round(4)

        return result

    def compute_rsi(self, df: pd.DataFrame) -> pd.Series:
        """
        Vektorizáltan kiszámítja a 14 periódusú Relatív Erősség Indexet (RSI_14)
        Wilder-féle exponenciális simítással a [0, 100] zárt intervallumon (FK3).
        """

        def _calc_single_rsi(series: pd.Series) -> pd.Series:
            delta = series.diff()
            gain = delta.clip(lower=0.0)
            loss = -delta.clip(upper=0.0)

            avg_gain = gain.ewm(alpha=1.0 / self.rsi_period, min_periods=1, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1.0 / self.rsi_period, min_periods=1, adjust=False).mean()

            rs = avg_gain / avg_loss.replace(0.0, np.nan)
            rsi = 100.0 - (100.0 / (1.0 + rs))
            # Ha avg_loss == 0 és avg_gain > 0 -> RSI = 100; ha nincs elmozdulás -> 50
            rsi = np.where((avg_loss == 0.0) & (avg_gain > 0.0), 100.0, rsi)
            rsi = np.where((avg_loss == 0.0) & (avg_gain == 0.0), 50.0, rsi)
            return pd.Series(rsi, index=series.index).fillna(50.0).clip(lower=0.0, upper=100.0).round(4)

        return df.groupby("ticker", observed=True)["close_price"].transform(_calc_single_rsi)

    def compute_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Vektorizáltan előállítja a MACD vonalat (EMA_12 - EMA_26), a 9 napos
        exponenciális jelzővonalat (MACD_Signal) és a MACD hisztogramot (FK3).
        """
        result = df.copy()
        fast_span, slow_span, signal_span = self.macd_params
        grouped_close = result.groupby("ticker", observed=True)["close_price"]

        if "ema_12" not in result.columns or "ema_26" not in result.columns:
            ema_fast = grouped_close.transform(lambda s: s.ewm(span=fast_span, adjust=False).mean())
            ema_slow = grouped_close.transform(lambda s: s.ewm(span=slow_span, adjust=False).mean())
        else:
            ema_fast = result["ema_12"]
            ema_slow = result["ema_26"]

        result["macd"] = (ema_fast - ema_slow).round(4)
        result["macd_signal"] = (
            result.groupby("ticker", observed=True)["macd"]
            .transform(lambda s: s.ewm(span=signal_span, adjust=False).mean())
            .round(4)
        )
        result["macd_hist"] = (result["macd"] - result["macd_signal"]).round(4)

        return result

    def transform_all(self, raw_price_df: pd.DataFrame) -> pd.DataFrame:
        """
        A teljes adattisztítási és indikátorszámítási munkafolyamatot összefogó
        vezérlőmetódus, amely a `daily_metrics` táblába tölthető DataFrame-et ad vissza.
        """
        cleaned_df = self.clean_ohlcv_data(raw_price_df).reset_index(drop=True)
        with_ma_df = self.compute_sma_ema(cleaned_df).reset_index(drop=True)
        with_ma_df["rsi_14"] = self.compute_rsi(with_ma_df)
        final_df = self.compute_macd(with_ma_df)

        return final_df