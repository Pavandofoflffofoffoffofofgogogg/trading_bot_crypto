"""
Execution Engine - Places and manages orders on the exchange via ccxt.
Handles order lifecycle: create, monitor, cancel.
"""
import logging
import time
from datetime import datetime, timezone
from typing import Optional

import ccxt

from config import settings
from engines.signal_engine import TradeSignal, SignalType

logger = logging.getLogger(__name__)


class ExecutionEngine:
    """Executes trades on the exchange."""

    def __init__(self, exchange: ccxt.Exchange):
        self.exchange = exchange
        self.order_history: list[dict] = []

    def execute_market_order(
        self,
        signal: TradeSignal,
        quantity: float,
    ) -> Optional[dict]:
        """
        Execute a market order based on a trade signal.

        Args:
            signal: TradeSignal with BUY or SELL type
            quantity: Amount in base currency

        Returns:
            Order response dict or None on failure
        """
        if signal.signal_type == SignalType.HOLD:
            logger.info("HOLD signal — no order placed")
            return None

        side = "buy" if signal.signal_type == SignalType.BUY else "sell"

        try:
            logger.info(
                "Placing %s MARKET order: %s qty=%.6f",
                side.upper(), signal.symbol, quantity,
            )
            order = self.exchange.create_order(
                symbol=signal.symbol,
                type="market",
                side=side,
                amount=quantity,
            )

            order_record = {
                "order_id": order.get("id"),
                "symbol": signal.symbol,
                "side": side,
                "type": "market",
                "quantity": quantity,
                "price": order.get("average") or order.get("price") or signal.price,
                "status": order.get("status"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "raw": order,
            }
            self.order_history.append(order_record)

            logger.info(
                "Order executed: id=%s status=%s price=%s",
                order_record["order_id"], order_record["status"], order_record["price"],
            )
            return order_record

        except ccxt.InsufficientFunds as e:
            logger.error("Insufficient funds for %s %s: %s", side, signal.symbol, e)
        except ccxt.InvalidOrder as e:
            logger.error("Invalid order for %s %s: %s", side, signal.symbol, e)
        except ccxt.NetworkError as e:
            logger.error("Network error placing order: %s", e)
        except ccxt.ExchangeError as e:
            logger.error("Exchange error placing order: %s", e)

        return None

    def execute_limit_order(
        self,
        signal: TradeSignal,
        quantity: float,
        price: float,
    ) -> Optional[dict]:
        """Execute a limit order at a specified price."""
        if signal.signal_type == SignalType.HOLD:
            return None

        side = "buy" if signal.signal_type == SignalType.BUY else "sell"

        try:
            logger.info(
                "Placing %s LIMIT order: %s qty=%.6f @ %.2f",
                side.upper(), signal.symbol, quantity, price,
            )
            order = self.exchange.create_order(
                symbol=signal.symbol,
                type="limit",
                side=side,
                amount=quantity,
                price=price,
            )

            order_record = {
                "order_id": order.get("id"),
                "symbol": signal.symbol,
                "side": side,
                "type": "limit",
                "quantity": quantity,
                "price": price,
                "status": order.get("status"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "raw": order,
            }
            self.order_history.append(order_record)
            return order_record

        except ccxt.BaseError as e:
            logger.error("Error placing limit order: %s", e)
            return None

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an open order."""
        try:
            self.exchange.cancel_order(order_id, symbol)
            logger.info("Cancelled order %s for %s", order_id, symbol)
            return True
        except ccxt.BaseError as e:
            logger.error("Error cancelling order %s: %s", order_id, e)
            return False

    def get_open_orders(self, symbol: str = None) -> list[dict]:
        """Fetch open orders from the exchange."""
        try:
            return self.exchange.fetch_open_orders(symbol)
        except ccxt.BaseError as e:
            logger.error("Error fetching open orders: %s", e)
            return []

    def get_order_status(self, order_id: str, symbol: str) -> Optional[dict]:
        """Check the status of a specific order."""
        try:
            return self.exchange.fetch_order(order_id, symbol)
        except ccxt.BaseError as e:
            logger.error("Error fetching order %s: %s", order_id, e)
            return None
