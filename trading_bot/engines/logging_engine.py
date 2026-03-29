"""
Logging Engine - Persists trades, signals, and errors to SQLite.
Also configures Python logging for the entire system.
"""
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config import settings

logger = logging.getLogger(__name__)


def setup_logging(level: str = settings.LOG_LEVEL) -> None:
    """Configure logging for the entire trading system."""
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))

    # File handler
    log_file = settings.LOGS_DIR / "trading.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


class LoggingEngine:
    """Persists trade logs, signals, and errors to SQLite."""

    def __init__(self, db_path: Path = settings.DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Create tables if they don't exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    pnl REAL,
                    status TEXT DEFAULT 'open',
                    order_id TEXT,
                    notes TEXT
                );

                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    price REAL NOT NULL,
                    confirmed INTEGER DEFAULT 0,
                    reason TEXT
                );

                CREATE TABLE IF NOT EXISTS errors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    component TEXT NOT NULL,
                    error_type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    details TEXT
                );

                CREATE TABLE IF NOT EXISTS performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    portfolio_value REAL NOT NULL,
                    daily_pnl REAL,
                    total_pnl REAL,
                    open_positions INTEGER,
                    drawdown REAL
                );
            """)
        logger.info("Database initialized at %s", self.db_path)

    def log_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        exit_price: Optional[float] = None,
        pnl: Optional[float] = None,
        status: str = "open",
        order_id: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> int:
        """Log a trade to the database. Returns the trade ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """INSERT INTO trades
                   (timestamp, symbol, side, quantity, entry_price, exit_price, pnl, status, order_id, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now(timezone.utc).isoformat(),
                    symbol, side, quantity, entry_price,
                    exit_price, pnl, status, order_id, notes,
                ),
            )
            trade_id = cursor.lastrowid
        logger.info("Logged trade #%d: %s %s %.6f @ %.2f", trade_id, side, symbol, quantity, entry_price)
        return trade_id

    def update_trade(
        self,
        trade_id: int,
        exit_price: float,
        pnl: float,
        status: str = "closed",
    ) -> None:
        """Update a trade with exit info."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE trades SET exit_price=?, pnl=?, status=? WHERE id=?",
                (exit_price, pnl, status, trade_id),
            )
        logger.info("Updated trade #%d: exit=%.2f pnl=%.2f", trade_id, exit_price, pnl)

    def log_signal(
        self,
        symbol: str,
        signal_type: str,
        confidence: float,
        price: float,
        confirmed: bool = False,
        reason: str = "",
    ) -> None:
        """Log a signal to the database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO signals
                   (timestamp, symbol, signal_type, confidence, price, confirmed, reason)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now(timezone.utc).isoformat(),
                    symbol, signal_type, confidence, price,
                    int(confirmed), reason,
                ),
            )

    def log_error(
        self,
        component: str,
        error_type: str,
        message: str,
        details: Optional[str] = None,
    ) -> None:
        """Log an error to the database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO errors
                   (timestamp, component, error_type, message, details)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    datetime.now(timezone.utc).isoformat(),
                    component, error_type, message, details,
                ),
            )
        logger.error("[%s] %s: %s", component, error_type, message)

    def log_performance(
        self,
        portfolio_value: float,
        daily_pnl: float,
        total_pnl: float,
        open_positions: int,
        drawdown: float,
    ) -> None:
        """Log a performance snapshot."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO performance
                   (timestamp, portfolio_value, daily_pnl, total_pnl, open_positions, drawdown)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now(timezone.utc).isoformat(),
                    portfolio_value, daily_pnl, total_pnl,
                    open_positions, drawdown,
                ),
            )

    def get_trades(self, symbol: Optional[str] = None, status: Optional[str] = None) -> list[dict]:
        """Query trades from the database."""
        query = "SELECT * FROM trades WHERE 1=1"
        params = []
        if symbol:
            query += " AND symbol=?"
            params.append(symbol)
        if status:
            query += " AND status=?"
            params.append(status)
        query += " ORDER BY timestamp DESC"

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def get_performance_history(self, limit: int = 100) -> list[dict]:
        """Get recent performance snapshots."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM performance ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]
