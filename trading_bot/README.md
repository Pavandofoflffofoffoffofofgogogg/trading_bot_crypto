# ML Crypto Trading Bot

A production-ready machine learning-based crypto trading system built with Python.

## Features

- **ML Models**: XGBoost, Random Forest, LSTM for trade prediction
- **80+ Features**: RSI, MACD, Bollinger Bands, volume signals, lag features, market regime detection
- **Risk Management**: Position sizing, stop-loss/take-profit, max drawdown halt, daily loss limits
- **Backtesting**: Full simulation with slippage, commissions, cooldown, Sharpe ratio
- **Live Trading**: Binance integration via ccxt (sandbox mode by default)
- **Dashboard**: Real-time Streamlit dashboard for monitoring
- **Alerts**: Telegram notifications for trades, signals, and errors
- **Auto-Retrain**: Scheduled model retraining with performance comparison

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train a model
python main.py train --symbol BTC/USDT --timeframe 1h --bars 1000

# Backtest
python main.py backtest --symbol BTC/USDT --bars 1000

# Launch dashboard
streamlit run dashboard.py

# Paper trade
python main.py paper --symbol BTC/USDT
```

## Environment Variables

```bash
export BINANCE_API_KEY="your_key"
export BINANCE_API_SECRET="your_secret"
export TELEGRAM_BOT_TOKEN="your_bot_token"
export TELEGRAM_CHAT_ID="your_chat_id"
```

## Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set **Main file path** to `dashboard.py`
5. Add your secrets in **App settings > Secrets**

## Architecture

```
trading_bot/
├── main.py                    # CLI entry point
├── dashboard.py               # Streamlit dashboard
├── config/settings.py         # All tunable parameters
├── engines/
│   ├── data_engine.py         # OHLCV data fetching (ccxt)
│   ├── feature_engine.py      # 80+ technical indicators
│   ├── labeling_engine.py     # Target variable creation
│   ├── model_engine.py        # ML model training & inference
│   ├── signal_engine.py       # Prediction → BUY/SELL/HOLD
│   ├── risk_engine.py         # Position sizing & risk limits
│   ├── execution_engine.py    # Order execution (Binance)
│   ├── backtest_engine.py     # Historical simulation
│   ├── logging_engine.py      # SQLite persistence
│   ├── telegram_engine.py     # Telegram alerts
│   ├── retrain_engine.py      # Auto-retraining pipeline
│   └── visualization_engine.py # Charts & plots
├── .streamlit/config.toml     # Streamlit configuration
└── requirements.txt
```
