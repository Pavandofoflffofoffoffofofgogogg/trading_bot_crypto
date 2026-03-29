"""
Data Engine - Fetches historical and real-time OHLCV data via ccxt.
Supports multiple exchanges, caching, and incremental updates.
"""
import logging
import time
from datetime import datetime, timezone
from typing import Optional

import ccxt
import pandas as pd

from config import settings

logger = logging.getLogger(__name__)


class DataEngine:
    """Fetches and manages market data from crypto exchanges."""

    def __init__(
        self,
        exchange_id: str = settings.EXCHANGE_ID,
        api_key: str = settings.API_KEY,
        api_secret: str = settings.API_SECRET,
        sandbox: bool = settings.SANDBOX_MODE,
    ):
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })
        if sandbox:
            self.exchange.set_sandbox_mode(True)

        self._cache: dict[str, pd.DataFrame] = {}
        logger.info("DataEngine initialized with exchange=%s sandbox=%s", exchange_id, sandbox)

    def fetch_ohlcv(
        self,
        symbol: str = settings.SYMBOL,
        timeframe: str = settings.TIMEFRAME,
        limit: int = settings.LOOKBACK_BARS,
        since: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV candles and return a clean DataFrame.

        Args:
            symbol: Trading pair (e.g. 'BTC/USDT')
            timeframe: Candle interval (e.g. '1h', '15m', '1d')
            limit: Number of candles to fetch
            since: Start timestamp in milliseconds

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        logger.info("Fetching %d bars of %s %s", limit, symbol, timeframe)

        all_candles = []
        fetched = 0
        max_per_request = min(limit, 1000)

        while fetched < limit:
            batch_limit = min(max_per_request, limit - fetched)
            try:
                candles = self.exchange.fetch_ohlcv(
                    symbol, timeframe, since=since, limit=batch_limit
                )
            except ccxt.NetworkError as e:
                logger.error("Network error fetching data: %s", e)
                time.sleep(5)
                continue
            except ccxt.ExchangeError as e:
                logger.error("Exchange error: %s", e)
                raise

            if not candles:
                break

            all_candles.extend(candles)
            fetched += len(candles)
            since = candles[-1][0] + 1  # Next ms after last candle

            if len(candles) < batch_limit:
                break  # No more data available

            time.sleep(self.exchange.rateLimit / 1000)

        df = pd.DataFrame(
            all_candles,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        self._cache[f"{symbol}_{timeframe}"] = df
        logger.info("Fetched %d candles for %s", len(df), symbol)
        return df

    def fetch_latest_candle(
        self,
        symbol: str = settings.SYMBOL,
        timeframe: str = settings.TIMEFRAME,
    ) -> pd.Series:
        """Fetch the most recent completed candle."""
        df = self.fetch_ohlcv(symbol, timeframe, limit=2)
        return df.iloc[-2]  # Second-to-last is the most recent completed candle

    def fetch_ticker(self, symbol: str = settings.SYMBOL) -> dict:
        """Fetch current ticker (bid/ask/last price)."""
        try:
            return self.exchange.fetch_ticker(symbol)
        except ccxt.BaseError as e:
            logger.error("Error fetching ticker for %s: %s", symbol, e)
            raise

    def fetch_order_book(self, symbol: str = settings.SYMBOL, limit: int = 10) -> dict:
        """Fetch the current order book."""
        try:
            return self.exchange.fetch_order_book(symbol, limit=limit)
        except ccxt.BaseError as e:
            logger.error("Error fetching order book: %s", e)
            raise

    def get_balance(self) -> dict:
        """Fetch account balance."""
        try:
            balance = self.exchange.fetch_balance()
            return {
                "total": balance.get("total", {}),
                "free": balance.get("free", {}),
                "used": balance.get("used", {}),
            }
        except ccxt.BaseError as e:
            logger.error("Error fetching balance: %s", e)
            raise

    def get_cached(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Return cached data if available."""
        return self._cache.get(f"{symbol}_{timeframe}")

    def save_to_csv(self, df: pd.DataFrame, filename: str) -> None:
        """Persist DataFrame to CSV in the data directory."""
        path = settings.DATA_DIR / filename
        df.to_csv(path, index=False)
        logger.info("Saved data to %s", path)

    def load_from_csv(self, filename: str) -> pd.DataFrame:
        """Load DataFrame from CSV in the data directory."""
        path = settings.DATA_DIR / filename
        df = pd.read_csv(path, parse_dates=["timestamp"])
        logger.info("Loaded %d rows from %s", len(df), path)
        return df
