"""
Telegram Engine - Sends trade alerts and status updates via Telegram bot.
"""
import logging
from typing import Optional
from urllib.parse import quote
from urllib.request import urlopen, Request
from urllib.error import URLError
import json

from config import settings

logger = logging.getLogger(__name__)


class TelegramEngine:
    """Sends notifications to Telegram."""

    def __init__(
        self,
        bot_token: str = settings.TELEGRAM_BOT_TOKEN,
        chat_id: str = settings.TELEGRAM_CHAT_ID,
    ):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = bool(bot_token and chat_id)

        if not self.enabled:
            logger.warning(
                "Telegram alerts disabled — set TELEGRAM_BOT_TOKEN and "
                "TELEGRAM_CHAT_ID environment variables to enable"
            )

    def _send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """Send a message via Telegram Bot API."""
        if not self.enabled:
            return False

        url = (
            f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            f"?chat_id={self.chat_id}"
            f"&parse_mode={parse_mode}"
            f"&text={quote(text)}"
        )

        try:
            req = Request(url, method="GET")
            with urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read().decode())
                if result.get("ok"):
                    logger.debug("Telegram message sent successfully")
                    return True
                else:
                    logger.error("Telegram API error: %s", result)
                    return False
        except URLError as e:
            logger.error("Failed to send Telegram message: %s", e)
            return False

    def send_signal_alert(
        self,
        signal_type: str,
        symbol: str,
        price: float,
        confidence: float,
        reason: str = "",
    ) -> bool:
        """Send a trade signal alert."""
        emoji = {"BUY": "\U0001f7e2", "SELL": "\U0001f534", "HOLD": "\U0001f7e1"}.get(signal_type, "\u2753")

        text = (
            f"{emoji} <b>{signal_type} Signal</b>\n"
            f"\n"
            f"<b>Symbol:</b> {symbol}\n"
            f"<b>Price:</b> ${price:,.2f}\n"
            f"<b>Confidence:</b> {confidence:.1%}\n"
        )
        if reason:
            text += f"<b>Reason:</b> {reason}\n"

        return self._send_message(text)

    def send_trade_executed(
        self,
        side: str,
        symbol: str,
        quantity: float,
        price: float,
        order_id: Optional[str] = None,
    ) -> bool:
        """Send notification when a trade is executed."""
        text = (
            f"\U0001f4b0 <b>Trade Executed</b>\n"
            f"\n"
            f"<b>Side:</b> {side.upper()}\n"
            f"<b>Symbol:</b> {symbol}\n"
            f"<b>Quantity:</b> {quantity:.6f}\n"
            f"<b>Price:</b> ${price:,.2f}\n"
            f"<b>Value:</b> ${quantity * price:,.2f}\n"
        )
        if order_id:
            text += f"<b>Order ID:</b> {order_id}\n"

        return self._send_message(text)

    def send_position_closed(
        self,
        symbol: str,
        pnl: float,
        exit_reason: str,
        exit_price: float,
    ) -> bool:
        """Send notification when a position is closed."""
        emoji = "\U0001f4c8" if pnl > 0 else "\U0001f4c9"
        text = (
            f"{emoji} <b>Position Closed</b>\n"
            f"\n"
            f"<b>Symbol:</b> {symbol}\n"
            f"<b>PnL:</b> ${pnl:,.2f}\n"
            f"<b>Exit Price:</b> ${exit_price:,.2f}\n"
            f"<b>Reason:</b> {exit_reason}\n"
        )
        return self._send_message(text)

    def send_daily_summary(
        self,
        portfolio_value: float,
        daily_pnl: float,
        total_pnl: float,
        open_positions: int,
        trades_today: int,
    ) -> bool:
        """Send end-of-day performance summary."""
        emoji = "\U0001f4c8" if daily_pnl >= 0 else "\U0001f4c9"
        text = (
            f"\U0001f4ca <b>Daily Summary</b>\n"
            f"\n"
            f"<b>Portfolio:</b> ${portfolio_value:,.2f}\n"
            f"<b>Daily PnL:</b> {emoji} ${daily_pnl:,.2f}\n"
            f"<b>Total PnL:</b> ${total_pnl:,.2f}\n"
            f"<b>Open Positions:</b> {open_positions}\n"
            f"<b>Trades Today:</b> {trades_today}\n"
        )
        return self._send_message(text)

    def send_error_alert(self, component: str, error: str) -> bool:
        """Send critical error alert."""
        text = (
            f"\u26a0\ufe0f <b>ERROR</b>\n"
            f"\n"
            f"<b>Component:</b> {component}\n"
            f"<b>Error:</b> {error}\n"
        )
        return self._send_message(text)

    def send_risk_alert(self, message: str) -> bool:
        """Send risk management alert (drawdown, daily loss limit, etc.)."""
        text = f"\U0001f6a8 <b>Risk Alert</b>\n\n{message}"
        return self._send_message(text)

    def send_model_retrained(self, metrics: dict) -> bool:
        """Send notification when model is retrained."""
        text = (
            f"\U0001f9e0 <b>Model Retrained</b>\n"
            f"\n"
            f"<b>Accuracy:</b> {metrics.get('accuracy', 0):.2%}\n"
            f"<b>F1 Score:</b> {metrics.get('f1', 0):.2%}\n"
            f"<b>Sharpe:</b> {metrics.get('strategy_sharpe', 0):.2f}\n"
        )
        return self._send_message(text)
