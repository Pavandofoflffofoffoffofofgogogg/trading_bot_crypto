"""
Retrain Engine - Automated model retraining pipeline.
Handles scheduled retraining, data refresh, and model comparison.
"""
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import settings
from engines.data_engine import DataEngine
from engines.feature_engine import FeatureEngine
from engines.labeling_engine import LabelingEngine
from engines.model_engine import ModelEngine

logger = logging.getLogger(__name__)


class RetrainEngine:
    """Manages automated model retraining."""

    def __init__(
        self,
        data_engine: DataEngine,
        feature_engine: FeatureEngine,
        labeling_engine: LabelingEngine,
        retrain_interval_hours: int = settings.RETRAIN_INTERVAL_HOURS,
        min_samples: int = settings.MIN_RETRAIN_SAMPLES,
    ):
        self.data_engine = data_engine
        self.feature_engine = feature_engine
        self.labeling_engine = labeling_engine
        self.retrain_interval_hours = retrain_interval_hours
        self.min_samples = min_samples
        self.last_retrain_time: Optional[datetime] = None
        self.retrain_history: list[dict] = []

    def should_retrain(self) -> bool:
        """Check if it's time to retrain based on the configured interval."""
        if self.last_retrain_time is None:
            return True

        hours_elapsed = (
            datetime.now(timezone.utc) - self.last_retrain_time
        ).total_seconds() / 3600

        return hours_elapsed >= self.retrain_interval_hours

    def retrain(
        self,
        symbol: str = settings.SYMBOL,
        timeframe: str = settings.TIMEFRAME,
        bars: int = settings.LOOKBACK_BARS,
        label_method: str = "fixed",
        model_type: str = settings.MODEL_TYPE,
    ) -> dict:
        """
        Full retraining pipeline:
        1. Fetch fresh data
        2. Compute features and labels
        3. Train new model
        4. Compare with existing model
        5. Replace if new model is better

        Returns:
            Dict with retraining results and metrics
        """
        logger.info("=== RETRAINING PIPELINE START ===")
        start_time = time.time()

        # 1. Fetch fresh data
        df = self.data_engine.fetch_ohlcv(symbol, timeframe, limit=bars)
        if len(df) < self.min_samples:
            logger.warning(
                "Insufficient data for retraining: %d < %d",
                len(df), self.min_samples,
            )
            return {"status": "skipped", "reason": "insufficient_data"}

        # 2. Features and labels
        df = self.feature_engine.compute_all_features(df)

        if label_method == "triple_barrier":
            df = self.labeling_engine.triple_barrier_label(df)
        elif label_method == "binary":
            df = self.labeling_engine.binary_label(df)
        else:
            df = self.labeling_engine.fixed_horizon_label(df)

        feature_names = self.feature_engine.get_feature_names(df)

        # 3. Train new model
        new_model = ModelEngine(model_type=model_type)
        X_train, X_test, y_train, y_test = new_model.prepare_data(df, feature_names)
        new_model.train(X_train, y_train)
        new_metrics = new_model.evaluate(X_test, y_test)

        # 4. Compare with existing model
        model_filename = f"model_{symbol.replace('/', '_')}_{timeframe}.pkl"
        model_path = settings.MODELS_DIR / model_filename
        should_replace = True

        if model_path.exists():
            old_model = ModelEngine(model_type=model_type)
            old_model.load_model(model_filename)
            old_metrics = old_model.metrics

            # Compare F1 scores — only replace if new model is better
            old_f1 = old_metrics.get("f1", 0)
            new_f1 = new_metrics.get("f1", 0)

            if new_f1 <= old_f1:
                should_replace = False
                logger.info(
                    "New model (F1=%.4f) not better than existing (F1=%.4f) — keeping old model",
                    new_f1, old_f1,
                )
            else:
                logger.info(
                    "New model (F1=%.4f) outperforms existing (F1=%.4f) — replacing",
                    new_f1, old_f1,
                )

        # 5. Save if better
        if should_replace:
            new_model.save_model(model_filename)
            logger.info("New model saved as %s", model_filename)

        elapsed = time.time() - start_time
        self.last_retrain_time = datetime.now(timezone.utc)

        result = {
            "status": "replaced" if should_replace else "kept_existing",
            "metrics": new_metrics,
            "training_time_seconds": round(elapsed, 2),
            "data_samples": len(df),
            "timestamp": self.last_retrain_time.isoformat(),
        }
        self.retrain_history.append(result)

        logger.info("=== RETRAINING COMPLETE in %.1fs ===", elapsed)
        return result

    def get_retrain_history(self) -> list[dict]:
        """Return history of retraining runs."""
        return self.retrain_history
