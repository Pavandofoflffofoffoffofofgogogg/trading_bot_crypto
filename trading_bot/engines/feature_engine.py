"""
Feature Engine - Computes technical indicators and advanced features.
All features are computed as new columns on the OHLCV DataFrame.
"""
import logging

import numpy as np
import pandas as pd
import ta

from config import settings

logger = logging.getLogger(__name__)


class FeatureEngine:
    """Generates ML features from OHLCV data using technical analysis."""

    def compute_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all features on the input DataFrame.
        Input must have columns: open, high, low, close, volume.
        Returns a new DataFrame with feature columns added.
        """
        df = df.copy()
        logger.info("Computing features on %d rows", len(df))

        df = self._add_trend_indicators(df)
        df = self._add_momentum_indicators(df)
        df = self._add_volatility_indicators(df)
        df = self._add_volume_indicators(df)
        df = self._add_price_features(df)
        df = self._add_statistical_features(df)
        df = self._add_lag_features(df)
        df = self._add_market_regime(df)

        # Drop rows with NaN from indicator warm-up
        initial_len = len(df)
        df = df.dropna().reset_index(drop=True)
        logger.info("Features computed. Rows: %d -> %d (dropped %d NaN rows)",
                     initial_len, len(df), initial_len - len(df))
        return df

    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Moving averages and trend indicators."""
        # Simple Moving Averages
        for window in settings.FEATURE_WINDOWS:
            df[f"sma_{window}"] = ta.trend.sma_indicator(df["close"], window=window)
            df[f"ema_{window}"] = ta.trend.ema_indicator(df["close"], window=window)

        # MACD
        macd = ta.trend.MACD(
            df["close"],
            window_fast=settings.MACD_FAST,
            window_slow=settings.MACD_SLOW,
            window_sign=settings.MACD_SIGNAL,
        )
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_histogram"] = macd.macd_diff()

        # ADX - Average Directional Index
        adx = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14)
        df["adx"] = adx.adx()
        df["adx_pos"] = adx.adx_pos()
        df["adx_neg"] = adx.adx_neg()

        # Ichimoku
        ichimoku = ta.trend.IchimokuIndicator(df["high"], df["low"])
        df["ichimoku_a"] = ichimoku.ichimoku_a()
        df["ichimoku_b"] = ichimoku.ichimoku_b()

        return df

    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """RSI, Stochastic, Williams %R, etc."""
        # RSI
        df["rsi"] = ta.momentum.rsi(df["close"], window=settings.RSI_PERIOD)

        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(
            df["high"], df["low"], df["close"], window=14, smooth_window=3
        )
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()

        # Williams %R
        df["williams_r"] = ta.momentum.williams_r(df["high"], df["low"], df["close"], lbp=14)

        # ROC - Rate of Change
        df["roc"] = ta.momentum.roc(df["close"], window=10)

        # CCI - Commodity Channel Index
        df["cci"] = ta.trend.cci(df["high"], df["low"], df["close"], window=20)

        return df

    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bollinger Bands, ATR, Keltner Channel."""
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(
            df["close"],
            window=settings.BOLLINGER_PERIOD,
            window_dev=settings.BOLLINGER_STD,
        )
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_middle"] = bb.bollinger_mavg()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_width"] = bb.bollinger_wband()
        df["bb_pct"] = bb.bollinger_pband()

        # ATR - Average True Range
        df["atr"] = ta.volatility.average_true_range(
            df["high"], df["low"], df["close"], window=settings.ATR_PERIOD
        )

        # Keltner Channel
        kc = ta.volatility.KeltnerChannel(df["high"], df["low"], df["close"], window=20)
        df["kc_upper"] = kc.keltner_channel_hband()
        df["kc_lower"] = kc.keltner_channel_lband()

        return df

    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume-based indicators."""
        # On-Balance Volume
        df["obv"] = ta.volume.on_balance_volume(df["close"], df["volume"])

        # Volume-Weighted Average Price (approx via rolling)
        df["vwap"] = (df["volume"] * (df["high"] + df["low"] + df["close"]) / 3).cumsum() / df["volume"].cumsum()

        # Money Flow Index
        df["mfi"] = ta.volume.money_flow_index(
            df["high"], df["low"], df["close"], df["volume"], window=14
        )

        # Chaikin Money Flow
        df["cmf"] = ta.volume.chaikin_money_flow(
            df["high"], df["low"], df["close"], df["volume"], window=20
        )

        # Volume SMA ratio
        df["volume_sma_20"] = df["volume"].rolling(window=20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma_20"]

        return df

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Price-derived features."""
        # Returns at various periods
        for period in [1, 3, 5, 10]:
            df[f"return_{period}"] = df["close"].pct_change(period)

        # Log returns
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))

        # Price relative to SMAs
        for window in settings.FEATURE_WINDOWS:
            df[f"price_sma_{window}_ratio"] = df["close"] / df[f"sma_{window}"]

        # High-Low spread
        df["hl_spread"] = (df["high"] - df["low"]) / df["close"]

        # Close position within bar
        df["close_position"] = (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-10)

        # Gap (open vs previous close)
        df["gap"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)

        return df

    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rolling statistical features."""
        for window in [10, 20]:
            df[f"volatility_{window}"] = df["log_return"].rolling(window=window).std()
            df[f"skew_{window}"] = df["log_return"].rolling(window=window).skew()
            df[f"kurtosis_{window}"] = df["log_return"].rolling(window=window).kurt()
            df[f"zscore_{window}"] = (
                (df["close"] - df["close"].rolling(window=window).mean())
                / df["close"].rolling(window=window).std()
            )

        return df

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lagged versions of key indicators for temporal context."""
        lag_cols = ["rsi", "macd", "macd_histogram", "adx", "bb_pct", "volume_ratio", "atr"]
        for col in lag_cols:
            if col in df.columns:
                for lag in [1, 2, 3, 5]:
                    df[f"{col}_lag_{lag}"] = df[col].shift(lag)

        # Lagged returns
        for lag in [1, 2, 3]:
            df[f"close_lag_{lag}"] = df["close"].shift(lag)
            df[f"volume_lag_{lag}"] = df["volume"].shift(lag)

        # Return momentum (acceleration)
        if "return_1" in df.columns:
            df["return_acceleration"] = df["return_1"] - df["return_1"].shift(1)

        return df

    def _add_market_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect market regime using volatility clustering and trend strength.

        Regimes:
          0 = low volatility / ranging
          1 = trending up
          2 = trending down
          3 = high volatility / choppy
        """
        # Volatility regime via rolling std percentile
        vol_window = 20
        vol = df["close"].pct_change().rolling(vol_window).std()
        vol_median = vol.rolling(100).median()
        high_vol = vol > vol_median * 1.5

        # Trend detection via EMA crossover
        if "ema_7" in df.columns and "ema_21" in df.columns:
            uptrend = (df["ema_7"] > df["ema_21"]) & (df["adx"] > 25) if "adx" in df.columns else (df["ema_7"] > df["ema_21"])
            downtrend = (df["ema_7"] < df["ema_21"]) & (df["adx"] > 25) if "adx" in df.columns else (df["ema_7"] < df["ema_21"])
        else:
            ema_fast = ta.trend.ema_indicator(df["close"], window=7)
            ema_slow = ta.trend.ema_indicator(df["close"], window=21)
            uptrend = ema_fast > ema_slow
            downtrend = ema_fast < ema_slow

        # Assign regimes
        regime = pd.Series(0, index=df.index)  # Default: ranging
        regime[uptrend & ~high_vol] = 1   # Trending up
        regime[downtrend & ~high_vol] = 2  # Trending down
        regime[high_vol] = 3               # High volatility

        df["market_regime"] = regime
        df["volatility_regime"] = (vol > vol_median).astype(int)

        # Regime duration (how many bars in current regime)
        df["regime_duration"] = df["market_regime"].groupby(
            (df["market_regime"] != df["market_regime"].shift()).cumsum()
        ).cumcount() + 1

        return df

    def get_feature_names(self, df: pd.DataFrame) -> list[str]:
        """Return the list of feature column names (excludes OHLCV + timestamp + labels)."""
        exclude = {"timestamp", "open", "high", "low", "close", "volume",
                   "forward_return", "label", "close_lag_1", "close_lag_2",
                   "close_lag_3", "volume_lag_1", "volume_lag_2", "volume_lag_3"}
        return [col for col in df.columns if col not in exclude]
