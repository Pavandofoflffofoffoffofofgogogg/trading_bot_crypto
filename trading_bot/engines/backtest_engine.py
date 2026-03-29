"""
Backtest Engine - Simulates trading strategy on historical data.
Tracks portfolio value, trades, and performance metrics.
"""
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from config import settings
from engines.signal_engine import SignalType

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """Record of a single backtest trade."""
    entry_idx: int
    exit_idx: int
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    return_pct: float
    exit_reason: str


@dataclass
class BacktestResult:
    """Complete backtest results."""
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    equity_curve: list[float]
    trades: list[BacktestTrade]

    def summary(self) -> str:
        return (
            f"\n{'='*50}\n"
            f"  BACKTEST RESULTS\n"
            f"{'='*50}\n"
            f"  Total Return:     {self.total_return:>10.2%}\n"
            f"  Annual Return:    {self.annual_return:>10.2%}\n"
            f"  Sharpe Ratio:     {self.sharpe_ratio:>10.2f}\n"
            f"  Max Drawdown:     {self.max_drawdown:>10.2%}\n"
            f"  Win Rate:         {self.win_rate:>10.2%}\n"
            f"  Profit Factor:    {self.profit_factor:>10.2f}\n"
            f"  Total Trades:     {self.total_trades:>10d}\n"
            f"  Winning Trades:   {self.winning_trades:>10d}\n"
            f"  Losing Trades:    {self.losing_trades:>10d}\n"
            f"  Avg Win:          {self.avg_win:>10.2%}\n"
            f"  Avg Loss:         {self.avg_loss:>10.2%}\n"
            f"{'='*50}"
        )


class BacktestEngine:
    """Simulates a trading strategy on historical data."""

    def __init__(
        self,
        initial_capital: float = settings.INITIAL_CAPITAL,
        commission_pct: float = settings.COMMISSION_PCT,
        stop_loss_pct: float = settings.STOP_LOSS_PCT,
        take_profit_pct: float = settings.TAKE_PROFIT_PCT,
        slippage_pct: float = settings.SLIPPAGE_PCT,
        cooldown_bars: int = settings.COOLDOWN_BARS,
    ):
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.slippage_pct = slippage_pct
        self.cooldown_bars = cooldown_bars

    def run(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
        position_size_pct: float = settings.MAX_POSITION_SIZE_PCT,
    ) -> BacktestResult:
        """
        Run backtest simulation.

        Args:
            df: DataFrame with OHLCV data (must align with predictions)
            predictions: Array of predicted labels (1=BUY, -1=SELL, 0=HOLD)
            probabilities: Optional prediction probabilities for confidence scaling
            position_size_pct: Fraction of capital per trade

        Returns:
            BacktestResult with full performance metrics
        """
        capital = self.initial_capital
        equity_curve = [capital]
        trades: list[BacktestTrade] = []

        in_position = False
        position_side = None
        entry_price = 0.0
        entry_idx = 0
        quantity = 0.0
        bars_since_trade = self.cooldown_bars  # Start ready

        for i in range(len(predictions)):
            current_price = df["close"].iloc[i]
            bars_since_trade += 1

            # Check stop-loss / take-profit if in position
            if in_position:
                if position_side == "long":
                    pnl_pct = (current_price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - current_price) / entry_price

                exit_reason = None
                if pnl_pct <= -self.stop_loss_pct:
                    exit_reason = "stop_loss"
                elif pnl_pct >= self.take_profit_pct:
                    exit_reason = "take_profit"
                elif (position_side == "long" and predictions[i] == -1) or \
                     (position_side == "short" and predictions[i] == 1):
                    exit_reason = "signal_reversal"

                if exit_reason:
                    # Apply slippage on exit (sell lower, buy-to-cover higher)
                    if position_side == "long":
                        exit_price = current_price * (1 - self.slippage_pct)
                    else:
                        exit_price = current_price * (1 + self.slippage_pct)

                    if position_side == "long":
                        pnl_pct = (exit_price - entry_price) / entry_price
                    else:
                        pnl_pct = (entry_price - exit_price) / entry_price

                    pnl = pnl_pct * quantity * entry_price
                    commission = abs(quantity * exit_price * self.commission_pct)
                    pnl -= commission
                    capital += pnl

                    trades.append(BacktestTrade(
                        entry_idx=entry_idx,
                        exit_idx=i,
                        side=position_side,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        quantity=quantity,
                        pnl=pnl,
                        return_pct=pnl_pct,
                        exit_reason=exit_reason,
                    ))
                    in_position = False

            # Open new position (with cooldown enforcement)
            if not in_position and predictions[i] != 0 and bars_since_trade >= self.cooldown_bars:
                allocation = capital * position_size_pct
                if probabilities is not None:
                    confidence = max(probabilities[i])
                    allocation *= confidence

                # Apply slippage: buy higher, sell lower
                if predictions[i] == 1:
                    fill_price = current_price * (1 + self.slippage_pct)
                else:
                    fill_price = current_price * (1 - self.slippage_pct)

                quantity = allocation / fill_price
                commission = allocation * self.commission_pct
                capital -= commission

                entry_price = fill_price
                entry_idx = i
                position_side = "long" if predictions[i] == 1 else "short"
                in_position = True
                bars_since_trade = 0

            equity_curve.append(capital)

        # Close any remaining position at the end
        if in_position:
            current_price = df["close"].iloc[-1]
            if position_side == "long":
                pnl_pct = (current_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - current_price) / entry_price

            pnl = pnl_pct * quantity * entry_price
            commission = abs(quantity * current_price * self.commission_pct)
            pnl -= commission
            capital += pnl

            trades.append(BacktestTrade(
                entry_idx=entry_idx,
                exit_idx=len(df) - 1,
                side=position_side,
                entry_price=entry_price,
                exit_price=current_price,
                quantity=quantity,
                pnl=pnl,
                return_pct=pnl_pct,
                exit_reason="end_of_data",
            ))

        return self._compute_metrics(equity_curve, trades)

    def _compute_metrics(
        self,
        equity_curve: list[float],
        trades: list[BacktestTrade],
    ) -> BacktestResult:
        """Calculate performance metrics from equity curve and trades."""
        equity = np.array(equity_curve)

        # Returns
        total_return = (equity[-1] - equity[0]) / equity[0]
        n_periods = len(equity) - 1
        annual_factor = 365 * 24  # Assuming hourly bars
        annual_return = (1 + total_return) ** (annual_factor / max(n_periods, 1)) - 1

        # Sharpe Ratio (assuming hourly returns)
        returns = np.diff(equity) / equity[:-1]
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(annual_factor)
        else:
            sharpe = 0.0

        # Max Drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0

        # Trade metrics
        winning = [t for t in trades if t.pnl > 0]
        losing = [t for t in trades if t.pnl <= 0]

        win_rate = len(winning) / max(len(trades), 1)
        avg_win = np.mean([t.return_pct for t in winning]) if winning else 0.0
        avg_loss = np.mean([t.return_pct for t in losing]) if losing else 0.0

        gross_profit = sum(t.pnl for t in winning)
        gross_loss = abs(sum(t.pnl for t in losing))
        profit_factor = gross_profit / max(gross_loss, 1e-10)

        result = BacktestResult(
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            avg_win=avg_win,
            avg_loss=avg_loss,
            equity_curve=equity_curve,
            trades=trades,
        )

        logger.info(result.summary())
        return result
