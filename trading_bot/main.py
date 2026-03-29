"""
Main Orchestrator - Ties all engines together.
Supports: train, backtest, live, paper, retrain, and dashboard modes.

Usage:
    python main.py train --symbol BTC/USDT --timeframe 1h
    python main.py backtest --symbol BTC/USDT
    python main.py live --symbol BTC/USDT
    python main.py paper --symbol BTC/USDT
    python main.py retrain --symbol BTC/USDT
    streamlit run dashboard.py
"""
import argparse
import logging
import time
from datetime import datetime, timezone

import numpy as np

from config import settings
from engines.backtest_engine import BacktestEngine
from engines.data_engine import DataEngine
from engines.execution_engine import ExecutionEngine
from engines.feature_engine import FeatureEngine
from engines.labeling_engine import LabelingEngine
from engines.logging_engine import LoggingEngine, setup_logging
from engines.model_engine import ModelEngine
from engines.retrain_engine import RetrainEngine
from engines.risk_engine import RiskEngine
from engines.signal_engine import SignalEngine
from engines.telegram_engine import TelegramEngine
from engines import visualization_engine as viz

logger = logging.getLogger(__name__)


class TradingBot:
    """Main orchestrator that coordinates all engines."""

    def __init__(self, symbol: str = settings.SYMBOL, timeframe: str = settings.TIMEFRAME):
        self.symbol = symbol
        self.timeframe = timeframe

        # Initialize engines
        self.data_engine = DataEngine()
        self.feature_engine = FeatureEngine()
        self.labeling_engine = LabelingEngine()
        self.model_engine = ModelEngine()
        self.signal_engine = SignalEngine()
        self.risk_engine = RiskEngine()
        self.execution_engine = ExecutionEngine(self.data_engine.exchange)
        self.backtest_engine = BacktestEngine()
        self.logging_engine = LoggingEngine()
        self.telegram = TelegramEngine()
        self.retrain_engine = RetrainEngine(
            self.data_engine, self.feature_engine, self.labeling_engine,
        )

        logger.info("TradingBot initialized for %s %s", symbol, timeframe)

    # ------------------------------------------------------------------ #
    #  TRAIN MODE                                                         #
    # ------------------------------------------------------------------ #
    def train(
        self,
        bars: int = settings.LOOKBACK_BARS,
        label_method: str = "fixed",
        model_type: str = settings.MODEL_TYPE,
    ) -> dict:
        """
        Full training pipeline:
        1. Fetch data
        2. Compute features
        3. Create labels
        4. Train model
        5. Evaluate, visualize, and save
        """
        logger.info("=== TRAINING PIPELINE START ===")

        # 1. Fetch data
        df = self.data_engine.fetch_ohlcv(self.symbol, self.timeframe, limit=bars)
        self.data_engine.save_to_csv(df, f"{self.symbol.replace('/', '_')}_{self.timeframe}.csv")

        # 2. Compute features
        df = self.feature_engine.compute_all_features(df)
        feature_names = self.feature_engine.get_feature_names(df)

        # 3. Create labels
        if label_method == "triple_barrier":
            df = self.labeling_engine.triple_barrier_label(df)
        elif label_method == "binary":
            df = self.labeling_engine.binary_label(df)
        else:
            df = self.labeling_engine.fixed_horizon_label(df)

        # 4. Prepare data and train
        self.model_engine = ModelEngine(model_type=model_type)
        X_train, X_test, y_train, y_test = self.model_engine.prepare_data(df, feature_names)
        self.model_engine.train(X_train, y_train)

        # 5. Evaluate
        metrics = self.model_engine.evaluate(X_test, y_test)

        # Feature importance + visualization
        importance = self.model_engine.get_feature_importance()
        if not importance.empty:
            logger.info("Top 10 features:\n%s", importance.head(10).to_string())
            viz.plot_feature_importance(importance)

        # Predictions vs actual visualization
        y_pred = self.model_engine.model.predict(X_test)
        viz.plot_predictions_vs_actual(y_test, y_pred)

        # Save model
        model_filename = f"model_{self.symbol.replace('/', '_')}_{self.timeframe}.pkl"
        self.model_engine.save_model(model_filename)

        # Telegram notification
        self.telegram.send_model_retrained(metrics)

        logger.info("=== TRAINING PIPELINE COMPLETE ===")
        return metrics

    # ------------------------------------------------------------------ #
    #  BACKTEST MODE                                                      #
    # ------------------------------------------------------------------ #
    def backtest(
        self,
        bars: int = settings.LOOKBACK_BARS,
        label_method: str = "fixed",
        model_type: str = settings.MODEL_TYPE,
    ):
        """
        Backtest pipeline:
        1. Fetch data + features + labels
        2. Train on first portion, predict on rest
        3. Run backtest simulation with slippage and cooldown
        4. Generate visualizations
        """
        logger.info("=== BACKTEST PIPELINE START ===")

        # Fetch and prepare
        df = self.data_engine.fetch_ohlcv(self.symbol, self.timeframe, limit=bars)
        df = self.feature_engine.compute_all_features(df)

        if label_method == "triple_barrier":
            df = self.labeling_engine.triple_barrier_label(df)
        elif label_method == "binary":
            df = self.labeling_engine.binary_label(df)
        else:
            df = self.labeling_engine.fixed_horizon_label(df)

        feature_names = self.feature_engine.get_feature_names(df)

        # Train/test split (chronological)
        self.model_engine = ModelEngine(model_type=model_type)
        X_train, X_test, y_train, y_test = self.model_engine.prepare_data(df, feature_names)
        self.model_engine.train(X_train, y_train)
        metrics = self.model_engine.evaluate(X_test, y_test)

        # Generate predictions for test set
        predictions = self.model_engine.model.predict(X_test)
        probabilities = self.model_engine.model.predict_proba(X_test)

        # Run backtest on test portion
        split_idx = int(len(df) * (1 - settings.TEST_SIZE))
        test_df = df.iloc[split_idx:].reset_index(drop=True)

        result = self.backtest_engine.run(test_df, predictions, probabilities)
        print(result.summary())

        # Generate visualizations
        viz.plot_equity_curve(result.equity_curve)
        if result.trades:
            viz.plot_trade_analysis(result.trades)

        importance = self.model_engine.get_feature_importance()
        if not importance.empty:
            viz.plot_feature_importance(importance)

        logger.info("=== BACKTEST PIPELINE COMPLETE ===")
        return result

    # ------------------------------------------------------------------ #
    #  RETRAIN MODE                                                       #
    # ------------------------------------------------------------------ #
    def retrain(
        self,
        bars: int = settings.LOOKBACK_BARS,
        label_method: str = "fixed",
        model_type: str = settings.MODEL_TYPE,
    ) -> dict:
        """Run a single retraining cycle, replacing model only if improved."""
        result = self.retrain_engine.retrain(
            symbol=self.symbol,
            timeframe=self.timeframe,
            bars=bars,
            label_method=label_method,
            model_type=model_type,
        )
        self.telegram.send_model_retrained(result.get("metrics", {}))
        return result

    # ------------------------------------------------------------------ #
    #  LIVE / PAPER TRADING MODE                                          #
    # ------------------------------------------------------------------ #
    def run_live(self, paper: bool = True):
        """
        Live trading loop:
        1. Load trained model
        2. Every interval: fetch latest data, compute features, predict
        3. Generate signal (with cooldown), check risk, execute if confirmed
        4. Periodically retrain model
        """
        mode = "PAPER" if paper else "LIVE"
        logger.info("=== %s TRADING MODE START ===", mode)

        # Load model
        model_filename = f"model_{self.symbol.replace('/', '_')}_{self.timeframe}.pkl"
        self.model_engine.load_model(model_filename)
        feature_names = self.model_engine.feature_names

        # Track portfolio
        if paper:
            portfolio_value = settings.INITIAL_CAPITAL
        else:
            balance = self.data_engine.get_balance()
            portfolio_value = sum(balance["total"].values())

        self.risk_engine.update_portfolio_peak(portfolio_value)

        logger.info("Starting %s trading loop (interval=%ds)", mode, settings.LIVE_LOOP_INTERVAL_SECONDS)

        while True:
            try:
                # Advance cooldown counter each iteration
                self.signal_engine.update_bar()

                self._trading_iteration(feature_names, portfolio_value, paper)

                # Check if retraining is due
                if self.retrain_engine.should_retrain():
                    logger.info("Scheduled retraining triggered...")
                    result = self.retrain_engine.retrain(
                        symbol=self.symbol, timeframe=self.timeframe,
                    )
                    if result["status"] == "replaced":
                        self.model_engine.load_model(model_filename)
                        feature_names = self.model_engine.feature_names
                        self.telegram.send_model_retrained(result.get("metrics", {}))

                time.sleep(settings.LIVE_LOOP_INTERVAL_SECONDS)
            except KeyboardInterrupt:
                logger.info("Trading loop stopped by user")
                break
            except Exception as e:
                logger.exception("Error in trading loop: %s", e)
                self.logging_engine.log_error("main", type(e).__name__, str(e))
                self.telegram.send_error_alert("main_loop", str(e))
                time.sleep(60)

    def _trading_iteration(
        self,
        feature_names: list[str],
        portfolio_value: float,
        paper: bool,
    ) -> None:
        """Single iteration of the trading loop."""
        # Fetch latest data
        df = self.data_engine.fetch_ohlcv(self.symbol, self.timeframe, limit=100)
        df = self.feature_engine.compute_all_features(df)

        if df.empty:
            logger.warning("No data available")
            return

        # Check existing positions for SL/TP
        ticker = self.data_engine.fetch_ticker(self.symbol)
        current_price = ticker["last"]

        trigger = self.risk_engine.check_stop_loss_take_profit(self.symbol, current_price)
        if trigger:
            pnl = self.risk_engine.close_position(self.symbol, current_price)
            if not paper:
                self.execution_engine.execute_market_order(
                    _make_exit_signal(self.symbol, current_price),
                    self.risk_engine.open_positions.get(self.symbol, None),
                )
            self.logging_engine.log_trade(
                self.symbol, "close", 0, current_price,
                exit_price=current_price, pnl=pnl, status="closed",
                notes=f"Closed by {trigger}",
            )
            self.telegram.send_position_closed(self.symbol, pnl, trigger, current_price)
            logger.info("Position closed by %s: PnL=$%.2f", trigger, pnl)
            return

        # Predict
        latest_features = df[feature_names].iloc[-1:].values
        probabilities = self.model_engine.predict_proba(latest_features)[0]

        # Generate signal (cooldown is enforced inside)
        signal = self.signal_engine.generate_signal(
            probabilities, self.symbol, current_price
        )

        # Log signal
        self.logging_engine.log_signal(
            signal.symbol, signal.signal_type.value,
            signal.confidence, signal.price,
            reason=signal.reason,
        )

        # Check confirmation
        if not self.signal_engine.confirm_signal(signal, df):
            return

        # Send signal alert
        self.telegram.send_signal_alert(
            signal.signal_type.value, signal.symbol,
            signal.price, signal.confidence, signal.reason,
        )

        # Check risk limits
        allowed, reason = self.risk_engine.check_risk_limits(signal, portfolio_value)
        if not allowed:
            logger.info("Trade blocked by risk engine: %s", reason)
            self.telegram.send_risk_alert(f"Trade blocked: {reason}")
            return

        # Calculate position size
        quantity = self.risk_engine.calculate_position_size(signal, portfolio_value, current_price)

        # Execute
        if paper:
            logger.info(
                "[PAPER] Would execute: %s %s qty=%.6f @ %.2f",
                signal.signal_type.value, self.symbol, quantity, current_price,
            )
            self.risk_engine.open_position(signal, quantity)
            self.telegram.send_trade_executed(
                signal.signal_type.value, self.symbol, quantity, current_price,
            )
        else:
            order = self.execution_engine.execute_market_order(signal, quantity)
            if order:
                self.risk_engine.open_position(signal, quantity)
                self.logging_engine.log_trade(
                    self.symbol, signal.signal_type.value.lower(),
                    quantity, current_price,
                    order_id=order.get("order_id"),
                )
                self.telegram.send_trade_executed(
                    signal.signal_type.value, self.symbol,
                    quantity, current_price, order.get("order_id"),
                )

        # Log performance
        self.logging_engine.log_performance(
            portfolio_value=portfolio_value,
            daily_pnl=self.risk_engine.daily_pnl,
            total_pnl=self.risk_engine.total_pnl,
            open_positions=len(self.risk_engine.open_positions),
            drawdown=0.0,
        )


def _make_exit_signal(symbol, price):
    """Helper to create a SELL signal for closing positions."""
    from engines.signal_engine import TradeSignal, SignalType
    return TradeSignal(
        signal_type=SignalType.SELL,
        symbol=symbol,
        confidence=1.0,
        price=price,
        timestamp=datetime.now(timezone.utc),
        reason="Position exit (SL/TP)",
    )


def main():
    parser = argparse.ArgumentParser(description="ML Crypto Trading Bot")
    parser.add_argument(
        "mode",
        choices=["train", "backtest", "live", "paper", "retrain"],
        help="Operating mode",
    )
    parser.add_argument("--symbol", default=settings.SYMBOL, help="Trading pair (e.g. BTC/USDT)")
    parser.add_argument("--timeframe", default=settings.TIMEFRAME, help="Candle timeframe (e.g. 1h)")
    parser.add_argument("--bars", type=int, default=settings.LOOKBACK_BARS, help="Number of historical bars")
    parser.add_argument("--label-method", default="fixed", choices=["fixed", "triple_barrier", "binary"])
    parser.add_argument("--model-type", default=settings.MODEL_TYPE,
                        choices=["xgboost", "random_forest", "lstm"])

    args = parser.parse_args()

    # Setup
    setup_logging()
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    bot = TradingBot(symbol=args.symbol, timeframe=args.timeframe)

    if args.mode == "train":
        bot.train(bars=args.bars, label_method=args.label_method, model_type=args.model_type)
    elif args.mode == "backtest":
        bot.backtest(bars=args.bars, label_method=args.label_method, model_type=args.model_type)
    elif args.mode == "retrain":
        bot.retrain(bars=args.bars, label_method=args.label_method, model_type=args.model_type)
    elif args.mode == "live":
        bot.run_live(paper=False)
    elif args.mode == "paper":
        bot.run_live(paper=True)


if __name__ == "__main__":
    main()
