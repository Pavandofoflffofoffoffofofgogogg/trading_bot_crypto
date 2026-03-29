"""
Labeling Engine - Creates target variables for supervised learning.
Supports multiple labeling strategies: fixed horizon, triple barrier, etc.
"""
import logging

import numpy as np
import pandas as pd

from config import settings

logger = logging.getLogger(__name__)


class LabelingEngine:
    """Creates ML target labels from price data."""

    def fixed_horizon_label(
        self,
        df: pd.DataFrame,
        forward_period: int = settings.FORWARD_PERIOD,
        profit_threshold: float = settings.PROFIT_THRESHOLD,
        loss_threshold: float = settings.LOSS_THRESHOLD,
    ) -> pd.DataFrame:
        """
        Label based on forward return over a fixed horizon.

        Labels:
            1 = BUY  (forward return > profit_threshold)
            0 = HOLD (return between thresholds)
           -1 = SELL (forward return < loss_threshold)

        Args:
            df: DataFrame with 'close' column
            forward_period: Number of bars to look ahead
            profit_threshold: Minimum return for BUY label
            loss_threshold: Maximum return for SELL label (negative)

        Returns:
            DataFrame with 'forward_return' and 'label' columns added
        """
        df = df.copy()
        df["forward_return"] = df["close"].shift(-forward_period) / df["close"] - 1

        conditions = [
            df["forward_return"] > profit_threshold,
            df["forward_return"] < loss_threshold,
        ]
        choices = [1, -1]
        df["label"] = np.select(conditions, choices, default=0)

        # Drop rows where we can't compute forward return
        df = df.dropna(subset=["forward_return"]).reset_index(drop=True)

        label_counts = df["label"].value_counts().to_dict()
        logger.info(
            "Fixed horizon labels (period=%d): BUY=%d, HOLD=%d, SELL=%d",
            forward_period,
            label_counts.get(1, 0),
            label_counts.get(0, 0),
            label_counts.get(-1, 0),
        )
        return df

    def triple_barrier_label(
        self,
        df: pd.DataFrame,
        take_profit: float = settings.TAKE_PROFIT_PCT,
        stop_loss: float = settings.STOP_LOSS_PCT,
        max_holding_period: int = 20,
    ) -> pd.DataFrame:
        """
        Triple Barrier Method (Lopez de Prado).
        A trade is labeled based on which barrier is hit first:
        - Upper barrier (take profit) -> 1
        - Lower barrier (stop loss)   -> -1
        - Time barrier (max holding)  -> 0

        Args:
            df: DataFrame with 'close' column
            take_profit: Upper barrier as fraction
            stop_loss: Lower barrier as fraction
            max_holding_period: Maximum bars before time barrier
        """
        df = df.copy()
        labels = []

        for i in range(len(df)):
            if i + max_holding_period >= len(df):
                labels.append(np.nan)
                continue

            entry_price = df["close"].iloc[i]
            upper = entry_price * (1 + take_profit)
            lower = entry_price * (1 - stop_loss)

            label = 0  # Default: time barrier hit
            for j in range(1, max_holding_period + 1):
                future_price = df["close"].iloc[i + j]
                if future_price >= upper:
                    label = 1
                    break
                elif future_price <= lower:
                    label = -1
                    break

            labels.append(label)

        df["label"] = labels
        df = df.dropna(subset=["label"]).reset_index(drop=True)
        df["label"] = df["label"].astype(int)

        label_counts = df["label"].value_counts().to_dict()
        logger.info(
            "Triple barrier labels: WIN=%d, LOSS=%d, NEUTRAL=%d",
            label_counts.get(1, 0),
            label_counts.get(-1, 0),
            label_counts.get(0, 0),
        )
        return df

    def binary_label(
        self,
        df: pd.DataFrame,
        forward_period: int = settings.FORWARD_PERIOD,
        threshold: float = 0.0,
    ) -> pd.DataFrame:
        """
        Simple binary label: 1 if price goes up, 0 if down.
        Useful for binary classification models.
        """
        df = df.copy()
        df["forward_return"] = df["close"].shift(-forward_period) / df["close"] - 1
        df["label"] = (df["forward_return"] > threshold).astype(int)
        df = df.dropna(subset=["forward_return"]).reset_index(drop=True)

        pos = df["label"].sum()
        neg = len(df) - pos
        logger.info("Binary labels: UP=%d, DOWN=%d (ratio=%.2f)", pos, neg, pos / max(neg, 1))
        return df
