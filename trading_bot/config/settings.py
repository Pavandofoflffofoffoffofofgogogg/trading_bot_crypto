"""
Central configuration for the trading system.
All parameters are centralized here for easy tuning.
"""
import os
from pathlib import Path

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# --- Exchange ---
EXCHANGE_ID = "binance"
API_KEY = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")
SANDBOX_MODE = True  # Use testnet by default

# --- Trading ---
SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"
LOOKBACK_BARS = 500  # Number of historical bars to fetch
TRADE_PAIRS = ["BTC/USDT", "ETH/USDT"]

# --- Feature Engineering ---
FEATURE_WINDOWS = [7, 14, 21, 50]
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2
ATR_PERIOD = 14

# --- Labeling ---
FORWARD_PERIOD = 5  # Bars to look ahead for return calculation
PROFIT_THRESHOLD = 0.005  # 0.5% min profit to label as BUY
LOSS_THRESHOLD = -0.005  # -0.5% min loss to label as SELL

# --- Model ---
MODEL_TYPE = "xgboost"  # "xgboost", "random_forest", "lightgbm"
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 200
MAX_DEPTH = 6
LEARNING_RATE = 0.05
PROBABILITY_THRESHOLD = 0.6  # Min probability to trigger a trade

# --- Risk Management ---
MAX_POSITION_SIZE_PCT = 0.02  # Max 2% of portfolio per trade
MAX_DRAWDOWN_PCT = 0.10  # 10% max drawdown before halt
STOP_LOSS_PCT = 0.02  # 2% stop loss
TAKE_PROFIT_PCT = 0.04  # 4% take profit (2:1 reward/risk)
MAX_OPEN_POSITIONS = 3
DAILY_LOSS_LIMIT_PCT = 0.05  # 5% daily loss limit

# --- Backtest ---
INITIAL_CAPITAL = 10000.0
COMMISSION_PCT = 0.001  # 0.1% per trade (Binance default)

# --- Logging ---
LOG_LEVEL = "INFO"
DB_PATH = BASE_DIR / "data" / "trading.db"

# --- Cooldown ---
COOLDOWN_BARS = 3  # Minimum bars between trades to avoid overtrading

# --- Slippage ---
SLIPPAGE_PCT = 0.0005  # 0.05% slippage per trade

# --- LSTM ---
LSTM_SEQUENCE_LENGTH = 30  # Lookback window for LSTM input
LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 32
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.2

# --- Telegram ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# --- Retraining ---
RETRAIN_INTERVAL_HOURS = 24  # Retrain model every N hours
MIN_RETRAIN_SAMPLES = 200  # Minimum new samples before retraining

# --- Dashboard ---
DASHBOARD_PORT = 8501

# --- Scheduling ---
LIVE_LOOP_INTERVAL_SECONDS = 60  # How often to check for signals in live mode
