"""
Visualization Engine - Feature importance plots, equity curves, and analytics.
Generates static charts saved to the logs directory.
"""
import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from config import settings

logger = logging.getLogger(__name__)

OUTPUT_DIR = settings.LOGS_DIR / "charts"


def _ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    filename: str = "feature_importance.png",
) -> Path:
    """
    Plot horizontal bar chart of top-N feature importances.

    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        top_n: Number of top features to show
        filename: Output filename

    Returns:
        Path to saved chart
    """
    _ensure_output_dir()

    top = importance_df.head(top_n).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    bars = ax.barh(top["feature"], top["importance"], color="#2196F3", edgecolor="white")

    # Highlight top 5
    for bar in bars[-5:]:
        bar.set_color("#FF5722")

    ax.set_xlabel("Importance Score")
    ax.set_title(f"Top {top_n} Feature Importances")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Feature importance chart saved to %s", path)
    return path


def plot_equity_curve(
    equity_curve: list[float],
    filename: str = "equity_curve.png",
    title: str = "Backtest Equity Curve",
) -> Path:
    """Plot equity curve with drawdown overlay."""
    _ensure_output_dir()

    equity = np.array(equity_curve)
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak * 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1], sharex=True)

    # Equity curve
    ax1.plot(equity, color="#2196F3", linewidth=1.5, label="Portfolio Value")
    ax1.fill_between(range(len(equity)), equity[0], equity, alpha=0.1, color="#2196F3")
    ax1.axhline(y=equity[0], color="gray", linestyle="--", alpha=0.5, label="Initial Capital")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.set_title(title)
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    # Drawdown
    ax2.fill_between(range(len(drawdown)), 0, -drawdown, color="#FF5722", alpha=0.4)
    ax2.plot(-drawdown, color="#FF5722", linewidth=0.8)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Bar Index")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Equity curve chart saved to %s", path)
    return path


def plot_trade_analysis(
    trades: list,
    filename: str = "trade_analysis.png",
) -> Path:
    """Plot trade PnL distribution and cumulative returns."""
    _ensure_output_dir()

    pnls = [t.pnl for t in trades]
    returns = [t.return_pct for t in trades]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # PnL distribution
    ax = axes[0, 0]
    colors = ["#4CAF50" if p > 0 else "#FF5722" for p in pnls]
    ax.bar(range(len(pnls)), pnls, color=colors, alpha=0.7)
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.5)
    ax.set_title("Trade PnL ($)")
    ax.set_xlabel("Trade #")
    ax.set_ylabel("PnL ($)")
    ax.grid(alpha=0.3)

    # Return distribution
    ax = axes[0, 1]
    ax.hist(returns, bins=30, color="#2196F3", alpha=0.7, edgecolor="white")
    ax.axvline(x=0, color="red", linestyle="--", alpha=0.7)
    ax.axvline(x=np.mean(returns), color="green", linestyle="--", alpha=0.7, label=f"Mean: {np.mean(returns):.2%}")
    ax.set_title("Return Distribution")
    ax.set_xlabel("Return (%)")
    ax.legend()
    ax.grid(alpha=0.3)

    # Cumulative PnL
    ax = axes[1, 0]
    cumulative = np.cumsum(pnls)
    ax.plot(cumulative, color="#2196F3", linewidth=1.5)
    ax.fill_between(range(len(cumulative)), 0, cumulative, alpha=0.1, color="#2196F3")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("Cumulative PnL ($)")
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Cumulative PnL ($)")
    ax.grid(alpha=0.3)

    # Win/Loss by exit reason
    ax = axes[1, 1]
    reasons = {}
    for t in trades:
        r = t.exit_reason
        if r not in reasons:
            reasons[r] = {"wins": 0, "losses": 0}
        if t.pnl > 0:
            reasons[r]["wins"] += 1
        else:
            reasons[r]["losses"] += 1

    reason_names = list(reasons.keys())
    wins = [reasons[r]["wins"] for r in reason_names]
    losses = [reasons[r]["losses"] for r in reason_names]
    x = np.arange(len(reason_names))
    width = 0.35
    ax.bar(x - width/2, wins, width, label="Wins", color="#4CAF50", alpha=0.7)
    ax.bar(x + width/2, losses, width, label="Losses", color="#FF5722", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(reason_names, rotation=45, ha="right")
    ax.set_title("Wins vs Losses by Exit Reason")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Trade analysis chart saved to %s", path)
    return path


def plot_predictions_vs_actual(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    filename: str = "predictions_vs_actual.png",
) -> Path:
    """Confusion-style scatter of predictions vs actual labels."""
    _ensure_output_dir()

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=sorted(set(y_test)))
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    ax.set_title("Prediction Confusion Matrix")

    plt.tight_layout()
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Predictions chart saved to %s", path)
    return path
