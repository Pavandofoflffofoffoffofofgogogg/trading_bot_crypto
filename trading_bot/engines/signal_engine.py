"""
Signal Engine - Converts model predictions into actionable trade signals.
Applies confidence filters and signal confirmation logic.
"""
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from config import settings

logger = logging.getLogger(__name__)


class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class TradeSignal:
    """Represents a trade signal with metadata."""
    signal_type: SignalType
    symbol: str
    confidence: float
    price: float
    timestamp: datetime
    features: Optional[dict] = None
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "signal_type": self.signal_type.value,
            "symbol": self.symbol,
            "confidence": round(self.confidence, 4),
            "price": self.price,
            "timestamp": self.timestamp.isoformat(),
            "reason": self.reason,
        }


class SignalEngine:
    """Converts ML predictions into trade signals with filtering and cooldown."""

    def __init__(
        self,
        probability_threshold: float = settings.PROBABILITY_THRESHOLD,
        cooldown_bars: int = settings.COOLDOWN_BARS,
    ):
        self.probability_threshold = probability_threshold
        self.cooldown_bars = cooldown_bars
        self.signal_history: list[TradeSignal] = []
        self._bars_since_last_trade: int = cooldown_bars  # Start ready to trade
        self._last_signal_type: Optional[SignalType] = None

    def update_bar(self) -> None:
        """Call once per new bar to advance the cooldown counter."""
        self._bars_since_last_trade += 1

    def is_on_cooldown(self) -> bool:
        """Check if we're still in the post-trade cooldown window."""
        return self._bars_since_last_trade < self.cooldown_bars

    def generate_signal(
        self,
        probabilities: np.ndarray,
        symbol: str,
        current_price: float,
        class_labels: list = None,
    ) -> TradeSignal:
        """
        Generate a trade signal from model output probabilities.

        Args:
            probabilities: Array of class probabilities (e.g., [P(SELL), P(HOLD), P(BUY)])
            symbol: Trading pair
            current_price: Current market price
            class_labels: Ordered class labels matching probability indices

        Returns:
            TradeSignal object
        """
        if class_labels is None:
            class_labels = [-1, 0, 1]  # SELL, HOLD, BUY

        # Handle both binary and multi-class
        if len(probabilities) == 2:
            # Binary: [P(DOWN), P(UP)]
            buy_prob = probabilities[1]
            sell_prob = probabilities[0]
            hold_prob = 0.0
        else:
            # Multi-class: [P(SELL), P(HOLD), P(BUY)]
            label_to_idx = {label: i for i, label in enumerate(class_labels)}
            buy_prob = probabilities[label_to_idx.get(1, 2)]
            sell_prob = probabilities[label_to_idx.get(-1, 0)]
            hold_prob = probabilities[label_to_idx.get(0, 1)]

        # Determine signal
        max_prob = max(buy_prob, sell_prob, hold_prob)
        now = datetime.now(timezone.utc)

        # Enforce cooldown — force HOLD if too soon after last trade
        if self.is_on_cooldown():
            signal = TradeSignal(
                signal_type=SignalType.HOLD,
                symbol=symbol,
                confidence=max_prob,
                price=current_price,
                timestamp=now,
                reason=f"Cooldown active ({self._bars_since_last_trade}/{self.cooldown_bars} bars)",
            )
            self.signal_history.append(signal)
            logger.info("Signal suppressed by cooldown: %d/%d bars",
                        self._bars_since_last_trade, self.cooldown_bars)
            return signal

        if buy_prob >= self.probability_threshold and buy_prob == max_prob:
            signal = TradeSignal(
                signal_type=SignalType.BUY,
                symbol=symbol,
                confidence=buy_prob,
                price=current_price,
                timestamp=now,
                reason=f"BUY probability {buy_prob:.2%} exceeds threshold {self.probability_threshold:.2%}",
            )
        elif sell_prob >= self.probability_threshold and sell_prob == max_prob:
            signal = TradeSignal(
                signal_type=SignalType.SELL,
                symbol=symbol,
                confidence=sell_prob,
                price=current_price,
                timestamp=now,
                reason=f"SELL probability {sell_prob:.2%} exceeds threshold {self.probability_threshold:.2%}",
            )
        else:
            signal = TradeSignal(
                signal_type=SignalType.HOLD,
                symbol=symbol,
                confidence=max_prob,
                price=current_price,
                timestamp=now,
                reason=f"No signal exceeds threshold {self.probability_threshold:.2%}",
            )

        # Reset cooldown on actionable signal
        if signal.signal_type != SignalType.HOLD:
            self._bars_since_last_trade = 0
            self._last_signal_type = signal.signal_type

        self.signal_history.append(signal)
        logger.info(
            "Signal: %s %s @ %.2f (confidence=%.4f) — %s",
            signal.signal_type.value, symbol, current_price, signal.confidence, signal.reason,
        )
        return signal

    def confirm_signal(
        self,
        signal: TradeSignal,
        df: pd.DataFrame,
    ) -> bool:
        """
        Apply additional confirmation filters before executing a trade.
        Uses technical indicators to filter out low-quality signals.

        Returns True if signal is confirmed.
        """
        if signal.signal_type == SignalType.HOLD:
            return False

        last_row = df.iloc[-1]
        confirmations = 0
        total_checks = 0

        if signal.signal_type == SignalType.BUY:
            # RSI not overbought
            if "rsi" in last_row:
                total_checks += 1
                if last_row["rsi"] < 70:
                    confirmations += 1

            # Price above SMA (trend alignment)
            if "sma_21" in last_row:
                total_checks += 1
                if signal.price > last_row["sma_21"]:
                    confirmations += 1

            # MACD histogram positive
            if "macd_histogram" in last_row:
                total_checks += 1
                if last_row["macd_histogram"] > 0:
                    confirmations += 1

        elif signal.signal_type == SignalType.SELL:
            # RSI not oversold
            if "rsi" in last_row:
                total_checks += 1
                if last_row["rsi"] > 30:
                    confirmations += 1

            # Price below SMA
            if "sma_21" in last_row:
                total_checks += 1
                if signal.price < last_row["sma_21"]:
                    confirmations += 1

            # MACD histogram negative
            if "macd_histogram" in last_row:
                total_checks += 1
                if last_row["macd_histogram"] < 0:
                    confirmations += 1

        # Require at least 2/3 confirmations
        confirmed = total_checks == 0 or (confirmations / total_checks >= 0.66)
        logger.info(
            "Signal confirmation: %d/%d checks passed -> %s",
            confirmations, total_checks, "CONFIRMED" if confirmed else "REJECTED",
        )
        return confirmed

    def get_signal_summary(self) -> pd.DataFrame:
        """Return signal history as a DataFrame."""
        if not self.signal_history:
            return pd.DataFrame()
        return pd.DataFrame([s.to_dict() for s in self.signal_history])
