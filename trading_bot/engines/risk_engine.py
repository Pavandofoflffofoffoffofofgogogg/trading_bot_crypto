"""
Risk Engine - Position sizing, stop-loss, drawdown control, and risk limits.
Enforces risk management rules before any trade is executed.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from config import settings
from engines.signal_engine import TradeSignal, SignalType

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents an open position."""
    symbol: str
    side: str  # "long" or "short"
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    entry_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def notional_value(self) -> float:
        return self.entry_price * self.quantity


class RiskEngine:
    """Manages risk: position sizing, stop-losses, and portfolio exposure."""

    def __init__(
        self,
        max_position_pct: float = settings.MAX_POSITION_SIZE_PCT,
        max_drawdown_pct: float = settings.MAX_DRAWDOWN_PCT,
        stop_loss_pct: float = settings.STOP_LOSS_PCT,
        take_profit_pct: float = settings.TAKE_PROFIT_PCT,
        max_open_positions: int = settings.MAX_OPEN_POSITIONS,
        daily_loss_limit_pct: float = settings.DAILY_LOSS_LIMIT_PCT,
    ):
        self.max_position_pct = max_position_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_open_positions = max_open_positions
        self.daily_loss_limit_pct = daily_loss_limit_pct

        self.open_positions: dict[str, Position] = {}
        self.peak_portfolio_value: float = 0.0
        self.daily_pnl: float = 0.0
        self.total_pnl: float = 0.0
        self.is_halted: bool = False

    def calculate_position_size(
        self,
        signal: TradeSignal,
        portfolio_value: float,
        current_price: float,
    ) -> float:
        """
        Calculate position size using fixed-fraction risk model.
        Scales position by signal confidence.

        Returns:
            Quantity to trade (in base currency units).
        """
        # Base allocation: max_position_pct of portfolio
        max_allocation = portfolio_value * self.max_position_pct

        # Scale by confidence (higher confidence = closer to max allocation)
        confidence_scalar = min(signal.confidence, 1.0)
        allocation = max_allocation * confidence_scalar

        quantity = allocation / current_price

        logger.info(
            "Position sizing: portfolio=$%.2f allocation=$%.2f qty=%.6f (confidence=%.2f)",
            portfolio_value, allocation, quantity, confidence_scalar,
        )
        return quantity

    def calculate_stop_loss(self, entry_price: float, side: str) -> float:
        """Calculate stop-loss price."""
        if side == "long":
            return entry_price * (1 - self.stop_loss_pct)
        else:
            return entry_price * (1 + self.stop_loss_pct)

    def calculate_take_profit(self, entry_price: float, side: str) -> float:
        """Calculate take-profit price."""
        if side == "long":
            return entry_price * (1 + self.take_profit_pct)
        else:
            return entry_price * (1 - self.take_profit_pct)

    def check_risk_limits(
        self,
        signal: TradeSignal,
        portfolio_value: float,
    ) -> tuple[bool, str]:
        """
        Check all risk limits before allowing a trade.

        Returns:
            (is_allowed, reason)
        """
        # Check if trading is halted
        if self.is_halted:
            return False, "Trading halted due to drawdown limit breach"

        # Check max open positions
        if len(self.open_positions) >= self.max_open_positions:
            return False, f"Max open positions ({self.max_open_positions}) reached"

        # Check if already in a position for this symbol
        if signal.symbol in self.open_positions:
            return False, f"Already have open position in {signal.symbol}"

        # Check daily loss limit
        if portfolio_value > 0:
            daily_loss_ratio = abs(min(self.daily_pnl, 0)) / portfolio_value
            if daily_loss_ratio >= self.daily_loss_limit_pct:
                return False, f"Daily loss limit ({self.daily_loss_limit_pct:.1%}) reached"

        # Check drawdown
        if self.peak_portfolio_value > 0:
            drawdown = (self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value
            if drawdown >= self.max_drawdown_pct:
                self.is_halted = True
                return False, f"Max drawdown ({self.max_drawdown_pct:.1%}) breached — trading halted"

        return True, "Risk checks passed"

    def open_position(
        self,
        signal: TradeSignal,
        quantity: float,
    ) -> Position:
        """Register a new open position."""
        side = "long" if signal.signal_type == SignalType.BUY else "short"
        position = Position(
            symbol=signal.symbol,
            side=side,
            entry_price=signal.price,
            quantity=quantity,
            stop_loss=self.calculate_stop_loss(signal.price, side),
            take_profit=self.calculate_take_profit(signal.price, side),
        )
        self.open_positions[signal.symbol] = position
        logger.info(
            "Opened %s position: %s qty=%.6f @ %.2f SL=%.2f TP=%.2f",
            side, signal.symbol, quantity, signal.price,
            position.stop_loss, position.take_profit,
        )
        return position

    def close_position(
        self,
        symbol: str,
        exit_price: float,
    ) -> float:
        """Close a position and return the realized PnL."""
        if symbol not in self.open_positions:
            logger.warning("No open position for %s", symbol)
            return 0.0

        pos = self.open_positions.pop(symbol)

        if pos.side == "long":
            pnl = (exit_price - pos.entry_price) * pos.quantity
        else:
            pnl = (pos.entry_price - exit_price) * pos.quantity

        self.daily_pnl += pnl
        self.total_pnl += pnl

        logger.info(
            "Closed %s %s @ %.2f (entry=%.2f) PnL=$%.2f",
            pos.side, symbol, exit_price, pos.entry_price, pnl,
        )
        return pnl

    def check_stop_loss_take_profit(
        self,
        symbol: str,
        current_price: float,
    ) -> Optional[str]:
        """
        Check if current price triggers stop-loss or take-profit.

        Returns:
            'stop_loss', 'take_profit', or None
        """
        if symbol not in self.open_positions:
            return None

        pos = self.open_positions[symbol]

        if pos.side == "long":
            if current_price <= pos.stop_loss:
                return "stop_loss"
            if current_price >= pos.take_profit:
                return "take_profit"
        else:
            if current_price >= pos.stop_loss:
                return "stop_loss"
            if current_price <= pos.take_profit:
                return "take_profit"

        return None

    def update_portfolio_peak(self, portfolio_value: float) -> None:
        """Update the peak portfolio value for drawdown tracking."""
        if portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = portfolio_value

    def reset_daily_pnl(self) -> None:
        """Reset daily PnL counter (call at start of each trading day)."""
        self.daily_pnl = 0.0

    def get_status(self) -> dict:
        """Return current risk engine status."""
        return {
            "open_positions": len(self.open_positions),
            "positions": {
                sym: {
                    "side": p.side,
                    "entry_price": p.entry_price,
                    "quantity": p.quantity,
                    "stop_loss": p.stop_loss,
                    "take_profit": p.take_profit,
                }
                for sym, p in self.open_positions.items()
            },
            "daily_pnl": round(self.daily_pnl, 2),
            "total_pnl": round(self.total_pnl, 2),
            "is_halted": self.is_halted,
            "peak_portfolio_value": round(self.peak_portfolio_value, 2),
        }
