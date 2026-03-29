"""
Streamlit Dashboard - Real-time monitoring of the ML trading system.

Deploy: Connect this repo to Streamlit Cloud with dashboard.py as the entry point.
Local:  streamlit run dashboard.py
"""
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# --- Paths (work both locally and on Streamlit Cloud) ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
DB_PATH = DATA_DIR / "trading.db"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# --- Page Config ---
st.set_page_config(
    page_title="ML Trading Bot Dashboard",
    page_icon="\U0001f4c8",
    layout="wide",
)


def get_db_connection():
    """Get a SQLite connection, initializing the DB if it doesn't exist."""
    db_exists = DB_PATH.exists()
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    if not db_exists:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT, symbol TEXT, side TEXT,
                quantity REAL, entry_price REAL, exit_price REAL,
                pnl REAL, status TEXT, order_id TEXT, notes TEXT
            );
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT, symbol TEXT, signal_type TEXT,
                confidence REAL, price REAL, confirmed INTEGER, reason TEXT
            );
            CREATE TABLE IF NOT EXISTS errors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT, component TEXT, error_type TEXT,
                message TEXT, details TEXT
            );
            CREATE TABLE IF NOT EXISTS performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT, portfolio_value REAL, daily_pnl REAL,
                total_pnl REAL, open_positions INTEGER, drawdown REAL
            );
        """)
    return conn


@st.cache_resource
def init_db():
    return get_db_connection()


def load_table(table: str, limit: int = 500) -> pd.DataFrame:
    """Load data from SQLite."""
    try:
        conn = init_db()
        df = pd.read_sql_query(
            f"SELECT * FROM {table} ORDER BY timestamp DESC LIMIT {limit}",
            conn,
        )
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    except Exception:
        return pd.DataFrame()


def load_model_metrics() -> dict:
    """Load the latest model metrics from the saved model file."""
    import pickle
    model_files = list(MODELS_DIR.glob("*.pkl"))
    if not model_files:
        return {}
    latest = max(model_files, key=lambda p: p.stat().st_mtime)
    with open(latest, "rb") as f:
        payload = pickle.load(f)
    return payload.get("metrics", {})


# --- Sidebar ---
st.sidebar.title("\U0001f916 ML Trading Bot")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Trades", "Signals", "Model Performance", "Risk Monitor", "Charts"],
)
st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit + XGBoost + ccxt")

# ================================================================== #
#  OVERVIEW                                                           #
# ================================================================== #
if page == "Overview":
    st.title("\U0001f4ca Trading Dashboard")

    perf = load_table("performance")
    trades = load_table("trades")

    col1, col2, col3, col4 = st.columns(4)
    if not perf.empty:
        latest = perf.iloc[0]
        col1.metric("Portfolio Value", f"${latest['portfolio_value']:,.2f}")
        col2.metric("Daily PnL", f"${latest['daily_pnl']:,.2f}",
                     delta=f"{latest['daily_pnl']:+,.2f}")
        col3.metric("Total PnL", f"${latest['total_pnl']:,.2f}",
                     delta=f"{latest['total_pnl']:+,.2f}")
        col4.metric("Open Positions", int(latest["open_positions"]))
    else:
        col1.metric("Portfolio Value", "$10,000.00")
        col2.metric("Daily PnL", "$0.00")
        col3.metric("Total PnL", "$0.00")
        col4.metric("Open Positions", 0)

    # Equity curve
    if not perf.empty and len(perf) > 1:
        st.subheader("Equity Curve")
        chart_data = perf.sort_values("timestamp")[["timestamp", "portfolio_value"]].set_index("timestamp")
        st.line_chart(chart_data)
    else:
        st.info("Start trading to see the equity curve here.")

    # Recent trades
    if not trades.empty:
        st.subheader("Recent Trades")
        st.dataframe(trades.head(20), use_container_width=True)

# ================================================================== #
#  TRADES                                                             #
# ================================================================== #
elif page == "Trades":
    st.title("\U0001f4b0 Trade History")

    trades = load_table("trades", limit=1000)
    if trades.empty:
        st.info("No trades recorded yet. Run the bot in paper or live mode first.")
    else:
        closed = trades[trades["status"] == "closed"]
        if not closed.empty:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Trades", len(closed))
            col2.metric("Win Rate", f"{(closed['pnl'] > 0).mean():.1%}")
            col3.metric("Total PnL", f"${closed['pnl'].sum():,.2f}")
            col4.metric("Avg PnL", f"${closed['pnl'].mean():,.2f}")

            st.subheader("PnL per Trade")
            st.bar_chart(closed.set_index("timestamp")["pnl"])

        st.subheader("All Trades")
        st.dataframe(trades, use_container_width=True)

# ================================================================== #
#  SIGNALS                                                            #
# ================================================================== #
elif page == "Signals":
    st.title("\U0001f4e1 Signal History")

    signals = load_table("signals", limit=500)
    if signals.empty:
        st.info("No signals recorded yet.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Signal Type Distribution")
            signal_counts = signals["signal_type"].value_counts()
            st.bar_chart(signal_counts)
        with col2:
            st.subheader("Confirmation Rate")
            confirmed = signals["confirmed"].sum()
            total = len(signals)
            st.metric("Confirmed Signals", f"{confirmed}/{total} ({confirmed/max(total,1):.1%})")

        st.subheader("Confidence Over Time")
        st.bar_chart(signals.set_index("timestamp")["confidence"])

        st.subheader("Recent Signals")
        st.dataframe(signals.head(50), use_container_width=True)

# ================================================================== #
#  MODEL PERFORMANCE                                                  #
# ================================================================== #
elif page == "Model Performance":
    st.title("\U0001f9e0 Model Performance")

    metrics = load_model_metrics()
    if not metrics:
        st.info("No trained model found. Run `python main.py train` first.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
        col2.metric("Precision", f"{metrics.get('precision', 0):.2%}")
        col3.metric("Recall", f"{metrics.get('recall', 0):.2%}")
        col4.metric("F1 Score", f"{metrics.get('f1', 0):.2%}")

        col5, col6 = st.columns(2)
        if "strategy_sharpe" in metrics:
            col5.metric("Strategy Sharpe Ratio", f"{metrics['strategy_sharpe']:.2f}")
        if "auc_roc" in metrics:
            col6.metric("AUC-ROC", f"{metrics['auc_roc']:.4f}")

        if "classification_report" in metrics:
            st.subheader("Classification Report")
            st.code(metrics["classification_report"], language="text")

        if "confusion_matrix" in metrics:
            st.subheader("Confusion Matrix")
            cm = np.array(metrics["confusion_matrix"])
            st.dataframe(pd.DataFrame(cm), use_container_width=True)

    # Feature importance chart
    fi_path = LOGS_DIR / "charts" / "feature_importance.png"
    if fi_path.exists():
        st.subheader("Feature Importance")
        st.image(str(fi_path))

# ================================================================== #
#  RISK MONITOR                                                       #
# ================================================================== #
elif page == "Risk Monitor":
    st.title("\U0001f6e1 Risk Monitor")

    perf = load_table("performance")
    errors = load_table("errors", limit=50)

    if not perf.empty:
        latest = perf.iloc[0]
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Drawdown", f"{latest.get('drawdown', 0):.2%}")
        col2.metric("Daily PnL", f"${latest['daily_pnl']:,.2f}")
        col3.metric("Open Positions", int(latest["open_positions"]))

        if "drawdown" in perf.columns and len(perf) > 1:
            st.subheader("Drawdown Over Time")
            dd_data = perf.sort_values("timestamp")[["timestamp", "drawdown"]].set_index("timestamp")
            st.area_chart(dd_data)
    else:
        st.info("No performance data yet. Start trading to monitor risk.")

    if not errors.empty:
        st.subheader("\u26a0\ufe0f Recent Errors")
        st.dataframe(errors, use_container_width=True)
    else:
        st.success("No errors recorded.")

# ================================================================== #
#  CHARTS                                                             #
# ================================================================== #
elif page == "Charts":
    st.title("\U0001f4c8 Analysis Charts")

    charts_dir = LOGS_DIR / "charts"
    if charts_dir.exists():
        chart_files = sorted(charts_dir.glob("*.png"))
        if chart_files:
            for chart_path in chart_files:
                st.subheader(chart_path.stem.replace("_", " ").title())
                st.image(str(chart_path))
        else:
            st.info("No charts generated yet. Run a backtest to generate charts.")
    else:
        st.info("Run `python main.py backtest` to generate analysis charts.")
