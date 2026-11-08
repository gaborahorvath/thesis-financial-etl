import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

try:
    import yfinance as yf
except ImportError:
    yf = None

logger = logging.getLogger(__name__)

# A mintavállalkozás 45 nemzetközi részvényből és ETF-ből álló portfólió-univerzuma
DEFAULT_PORTFOLIO_TICKERS: List[str] = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "BRK-B", "JPM", "V",
    "UNH", "XOM", "MA", "PG", "JNJ", "HD", "MRK", "COST", "ABBV", "CVX",
    "CRM", "BAC", "WMT", "AMD", "NFLX", "PEP", "KO", "TMO", "LIN", "ADBE",
    "DIS", "CSCO", "ABT", "MCD", "WFC", "INTC", "VZ", "CMCSA", "IBM", "QCOM",
    "SPY", "QQQ", "DIA", "IWM", "VTI"
]


class MarketDataExtractor:
    """
    Külső pénzügyi REST API-kból (Yahoo Finance, Alpha Vantage) történő
    automatizált adatkinyerést megvalósító osztály (FK1 követelmény).
    """

    ALPHAVANTAGE_BASE_URL = "https://www.alphavantage.co/query"

    def __init__(
        self,
        api_key: Optional[str] = None,
        tickers: Optional[List[str]] = None,
        timeout_sec: int = 15,
        max_retries: int = 3
    ) -> None:
        self._api_key: str = api_key or os.environ.get("ALPHAVANTAGE_API_KEY", "")
        self._tickers: List[str] = tickers or DEFAULT_PORTFOLIO_TICKERS
        self._timeout_sec: int = timeout_sec
        self._max_retries: int = max_retries
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "PortfolioETLPlatform/1.0 (GDE-BSc-Thesis)",
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate"
        })

    @staticmethod
    def _mask_secret(text: str, secret: str) -> str:
        """Maszkolja az API-kulcsot a naplóüzenetekben (NFK4 információbiztonság)."""
        if not secret or len(secret) < 4:
            return text
        return text.replace(secret, "***MASKED_API_KEY***")

    def _request_with_retry(self, endpoint: str, params: Dict[str, str]) -> Dict:
        """
        HTTP GET lekérdezés exponenciális visszalépéses (exponential backoff)
        újrapróbálkozással és Alpha Vantage HTTP 200 Soft Error ellenőrzéssel.
        """
        for attempt in range(1, self._max_retries + 1):
            try:
                response = self._session.get(
                    endpoint,
                    params=params,
                    timeout=self._timeout_sec
                )

                # Kliensoldali (4xx, kivéve 429) hiba esetén nem érdemes újrapróbálkozni
                if 400 <= response.status_code < 500 and response.status_code != 429:
                    safe_url = self._mask_secret(response.url, self._api_key)
                    logger.error(f"Kliensoldali HTTP {response.status_code} hiba: {safe_url}")
                    response.raise_for_status()

                # HTTP 429 vagy 5xx szerverhiba esetén exponenciális várakozás
                if response.status_code == 429 or response.status_code >= 500:
                    raise requests.exceptions.RequestException(
                        f"Ideiglenes szerver/kvóta hiba: HTTP {response.status_code}"
                    )

                payload = response.json()

                # Alpha Vantage specifikus HTTP 200 "Soft Error" ellenőrzés
                if "Error Message" in payload:
                    raise ValueError(f"Alpha Vantage API hiba: {payload['Error Message']}")

                if "Note" in payload or "Information" in payload:
                    note_msg = payload.get("Note") or payload.get("Information")
                    logger.warning(f"Alpha Vantage Rate Limit (Soft Error): {note_msg}")
                    raise requests.exceptions.RetryError("API rate limit elérve (Soft Error)")

                return payload

            except (requests.exceptions.RequestException, ValueError) as exc:
                safe_err = self._mask_secret(str(exc), self._api_key)
                if attempt == self._max_retries:
                    logger.warning(
                        f"Sikertelen API lekérdezés {self._max_retries} próba után: {safe_err}"
                    )
                    return {}
                sleep_sec = 2 ** attempt
                logger.info(
                    f"Újrapróbálkozás ({attempt}/{self._max_retries}) {sleep_sec} mp múlva..."
                )
                time.sleep(sleep_sec)

        return {}

    def extract_ohlcv_history(self, period: str = "2y") -> pd.DataFrame:
        """
        Letölti a portfólió-univerzum napi OHLCV árfolyamadatait.
        Elsődlegesen a Yahoo Finance kötegelt végpontját használja, hálózati
        korlátozás vagy tesztüzemmód esetén determinisztikus adatsort állít elő.
        """
        logger.info(f"Napi OHLCV idősorok kinyerése {len(self._tickers)} instrumentumra...")
        records: List[pd.DataFrame] = []

        if yf is not None:
            try:
                raw_data = yf.download(
                    tickers=self._tickers,
                    period=period,
                    interval="1d",
                    group_by="ticker",
                    auto_adjust=False,
                    progress=False,
                    threads=False
                )
                if not raw_data.empty:
                    for ticker in self._tickers:
                        df_t = raw_data[ticker].copy() if len(self._tickers) > 1 else raw_data.copy()
                        df_t = df_t.dropna(how="all").reset_index()
                        if not df_t.empty:
                            df_t["ticker"] = ticker
                            df_t = df_t.rename(columns={
                                "Date": "trade_date",
                                "Open": "open_price",
                                "High": "high_price",
                                "Low": "low_price",
                                "Close": "close_price",
                                "Adj Close": "adj_close",
                                "Volume": "volume"
                            })
                            records.append(df_t[[
                                "ticker", "trade_date", "open_price", "high_price",
                                "low_price", "close_price", "adj_close", "volume"
                            ]])
            except Exception as exc:
                logger.warning(f"yfinance letöltési hiba ({exc}), tartalék generátor aktiválása.")

        if not records:
            logger.info("Tartalék OHLCV idősor generálása reprodukálható tesztfutáshoz...")
            records = self._generate_fallback_ohlcv(days=504)

        return pd.concat(records, ignore_index=True)

    def extract_company_fundamentals(self, max_api_calls: int = 5) -> pd.DataFrame:
        """
        Vállalati fundamentális törzsadatok (P/E, EPS, Market Cap, Szektor) kinyerése
        az Alpha Vantage OVERVIEW végpontjáról, gördülő kvótakezeléssel.
        """
        logger.info("Vállalati fundamentális törzsadatok kinyerése...")
        fundamentals_list: List[Dict] = []

        for idx, ticker in enumerate(self._tickers):
            payload = {}
            if self._api_key and self._api_key != "DEMO" and idx < max_api_calls:
                params = {
                    "function": "OVERVIEW",
                    "symbol": ticker,
                    "apikey": self._api_key
                }
                payload = self._request_with_retry(self.ALPHAVANTAGE_BASE_URL, params)

            if payload and "Symbol" in payload:
                fundamentals_list.append({
                    "ticker": payload.get("Symbol", ticker),
                    "company_name": payload.get("Name", f"{ticker} Corp."),
                    "sector": payload.get("Sector", "Technology"),
                    "industry": payload.get("Industry", "Software & Services"),
                    "pe_ratio": payload.get("PERatio", None),
                    "eps": payload.get("DilutedEPSTTM", None),
                    "market_cap": payload.get("MarketCapitalization", None),
                    "updated_at": datetime.utcnow()
                })
            else:
                # Reprodukálható törzsadat, ha az ingyenes napi kvóta (25 kérés/nap) betelt
                sectors = ["Technology", "Financials", "Healthcare", "Consumer", "Energy", "ETF"]
                fundamentals_list.append({
                    "ticker": ticker,
                    "company_name": f"{ticker} Holdings Inc.",
                    "sector": "ETF" if ticker in {"SPY", "QQQ", "DIA", "IWM", "VTI"} else sectors[idx % 5],
                    "industry": "Index Fund" if ticker in {"SPY", "QQQ", "DIA", "IWM", "VTI"} else "Global Equities",
                    "pe_ratio": round(15.0 + (idx % 20) * 1.15, 4),
                    "eps": round(2.50 + (idx % 10) * 0.75, 4),
                    "market_cap": int(50_000_000_000 + idx * 12_500_000_000),
                    "updated_at": datetime.utcnow()
                })

        return pd.DataFrame(fundamentals_list)

    def _generate_fallback_ohlcv(self, days: int = 504) -> List[pd.DataFrame]:
        """Determinisztikus OHLCV adatsor előállítása offline/CI teszteléshez."""
        end_date = datetime.utcnow().date()
        dates = pd.bdate_range(end=end_date, periods=days)
        dfs: List[pd.DataFrame] = []

        for idx, ticker in enumerate(self._tickers):
            rng = np.random.default_rng(seed=1000 + idx)
            daily_returns = rng.normal(loc=0.0005, scale=0.015, size=len(dates))
            close_prices = (100.0 + idx * 5.0) * np.cumprod(1.0 + daily_returns)
            open_prices = close_prices * (1.0 + rng.normal(0, 0.004, size=len(dates)))
            high_prices = np.maximum(open_prices, close_prices) * (1.0 + np.abs(rng.normal(0, 0.006, size=len(dates))))
            low_prices = np.minimum(open_prices, close_prices) * (1.0 - np.abs(rng.normal(0, 0.006, size=len(dates))))
            volumes = rng.integers(low=1_000_000, high=50_000_000, size=len(dates))

            dfs.append(pd.DataFrame({
                "ticker": ticker,
                "trade_date": dates,
                "open_price": np.round(open_prices, 4),
                "high_price": np.round(high_prices, 4),
                "low_price": np.round(low_prices, 4),
                "close_price": np.round(close_prices, 4),
                "adj_close": np.round(close_prices, 4),
                "volume": volumes
            }))
        return dfs