"""
Model Engine - Trains, evaluates, and manages ML models.
Supports XGBoost, Random Forest, and LSTM.
"""
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config import settings

logger = logging.getLogger(__name__)


class LSTMWrapper:
    """
    Sklearn-compatible wrapper around a PyTorch LSTM for classification.
    Handles sequence creation, training, and prediction.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = settings.LSTM_HIDDEN_SIZE,
        num_layers: int = settings.LSTM_NUM_LAYERS,
        n_classes: int = 3,
        sequence_length: int = settings.LSTM_SEQUENCE_LENGTH,
        epochs: int = settings.LSTM_EPOCHS,
        batch_size: int = settings.LSTM_BATCH_SIZE,
        dropout: float = settings.LSTM_DROPOUT,
        learning_rate: float = 0.001,
    ):
        import torch
        import torch.nn as nn

        self.sequence_length = sequence_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = _LSTMNet(
            input_size, hidden_size, num_layers, n_classes, dropout
        ).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.classes_ = None
        self.feature_importances_ = None

    def _make_sequences(self, X: np.ndarray) -> np.ndarray:
        """Convert flat feature matrix to overlapping sequences."""
        if len(X) <= self.sequence_length:
            # Pad if not enough data
            pad = np.zeros((self.sequence_length - len(X) + 1, X.shape[1]))
            X = np.vstack([pad, X])

        sequences = []
        for i in range(self.sequence_length, len(X) + 1):
            sequences.append(X[i - self.sequence_length : i])
        return np.array(sequences)

    def fit(self, X: np.ndarray, y: np.ndarray):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        self.classes_ = np.unique(y)
        # Map labels to 0..n_classes-1
        label_map = {label: i for i, label in enumerate(sorted(self.classes_))}
        y_mapped = np.array([label_map[label] for label in y])

        # Create sequences — align labels with end of each sequence
        X_seq = self._make_sequences(X)
        y_seq = y_mapped[self.sequence_length - 1 :] if len(y_mapped) > self.sequence_length else y_mapped

        # Ensure alignment
        min_len = min(len(X_seq), len(y_seq))
        X_seq = X_seq[:min_len]
        y_seq = y_seq[:min_len]

        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.LongTensor(y_seq).to(self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for X_batch, y_batch in loader:
                self.optimizer.zero_grad()
                output = self.model(X_batch)
                loss = self.criterion(output, y_batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                logger.info("LSTM Epoch %d/%d Loss: %.4f", epoch + 1, self.epochs, total_loss / len(loader))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        sorted_classes = sorted(self.classes_)
        return np.array([sorted_classes[i] for i in indices])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch

        X_seq = self._make_sequences(X)
        X_tensor = torch.FloatTensor(X_seq).to(self.device)

        self.model.eval()
        with torch.no_grad():
            output = self.model(X_tensor)
            proba = torch.softmax(output, dim=1).cpu().numpy()
        return proba


def _create_lstm_net():
    """Lazy import to avoid requiring torch when not using LSTM."""
    import torch.nn as nn

    class _LSTMNet(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, n_classes, dropout):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size, hidden_size, num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0,
            )
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size, n_classes)

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            last_hidden = lstm_out[:, -1, :]
            out = self.dropout(last_hidden)
            return self.fc(out)

    return _LSTMNet


# Lazy-load to avoid import error when torch not installed
try:
    _LSTMNet = _create_lstm_net()
except ImportError:
    _LSTMNet = None


class ModelEngine:
    """Trains and manages ML models for trade prediction."""

    def __init__(self, model_type: str = settings.MODEL_TYPE):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names: list[str] = []
        self.metrics: dict = {}

    def _create_model(self, n_classes: int = 3, n_features: int = 0):
        """Instantiate the configured model."""
        if self.model_type == "xgboost":
            params = {
                "n_estimators": settings.N_ESTIMATORS,
                "max_depth": settings.MAX_DEPTH,
                "learning_rate": settings.LEARNING_RATE,
                "random_state": settings.RANDOM_STATE,
                "use_label_encoder": False,
                "eval_metric": "mlogloss" if n_classes > 2 else "logloss",
                "tree_method": "hist",
            }
            if n_classes > 2:
                params["objective"] = "multi:softprob"
                params["num_class"] = n_classes
            else:
                params["objective"] = "binary:logistic"
            self.model = XGBClassifier(**params)

        elif self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=settings.N_ESTIMATORS,
                max_depth=settings.MAX_DEPTH,
                random_state=settings.RANDOM_STATE,
                n_jobs=-1,
            )

        elif self.model_type == "lstm":
            if _LSTMNet is None:
                raise ImportError("PyTorch is required for LSTM. Install with: pip install torch")
            self.model = LSTMWrapper(
                input_size=n_features,
                n_classes=n_classes,
            )

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        logger.info("Created %s model", self.model_type)

    def prepare_data(
        self,
        df: pd.DataFrame,
        feature_names: list[str],
        label_col: str = "label",
        test_size: float = settings.TEST_SIZE,
    ) -> tuple:
        """
        Split data chronologically (no shuffle for time series).

        Returns:
            (X_train, X_test, y_train, y_test)
        """
        self.feature_names = feature_names

        X = df[feature_names].values
        y = df[label_col].values

        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Fit scaler on train only
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        logger.info(
            "Data split: train=%d test=%d features=%d",
            len(X_train), len(X_test), len(feature_names),
        )
        return X_train, X_test, y_train, y_test

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the model."""
        n_classes = len(np.unique(y_train))
        self._create_model(n_classes, n_features=X_train.shape[1])

        logger.info("Training %s model on %d samples...", self.model_type, len(X_train))
        self.model.fit(X_train, y_train)
        logger.info("Training complete.")

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate model performance and return metrics."""
        y_pred = self.model.predict(X_test)

        self.metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "classification_report": classification_report(y_test, y_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }

        # AUC for binary classification
        if len(np.unique(y_test)) == 2:
            y_proba = self.model.predict_proba(X_test)[:, 1]
            self.metrics["auc_roc"] = roc_auc_score(y_test, y_proba)

        # Strategy Sharpe ratio (using predictions as signals)
        self.metrics["strategy_sharpe"] = self._compute_strategy_sharpe(X_test, y_test)

        logger.info("Model Performance:")
        logger.info("  Accuracy:       %.4f", self.metrics["accuracy"])
        logger.info("  Precision:      %.4f", self.metrics["precision"])
        logger.info("  Recall:         %.4f", self.metrics["recall"])
        logger.info("  F1 Score:       %.4f", self.metrics["f1"])
        logger.info("  Strategy Sharpe:%.4f", self.metrics["strategy_sharpe"])
        if "auc_roc" in self.metrics:
            logger.info("  AUC-ROC:        %.4f", self.metrics["auc_roc"])

        return self.metrics

    def _compute_strategy_sharpe(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        annualization_factor: float = np.sqrt(365 * 24),
    ) -> float:
        """
        Compute Sharpe ratio of a strategy that goes long on BUY predictions
        and short on SELL predictions, using actual future returns as proxy.

        This gives a model-quality metric that correlates with trading performance.
        """
        y_pred = self.model.predict(X_test)

        # Simulate: +1 return when correctly predicting up, -1 when wrong
        # Use sign agreement as proxy returns
        strategy_returns = np.where(y_pred == y_test, abs(y_pred) * 0.005, -abs(y_pred) * 0.005)
        strategy_returns = np.where(y_pred == 0, 0, strategy_returns)  # No return on HOLD

        if len(strategy_returns) < 2 or np.std(strategy_returns) == 0:
            return 0.0

        return float((np.mean(strategy_returns) / np.std(strategy_returns)) * annualization_factor)

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
    ) -> dict:
        """Time-series cross-validation."""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = cross_val_score(self.model, X, y, cv=tscv, scoring="f1_weighted")

        result = {
            "cv_scores": scores.tolist(),
            "cv_mean": scores.mean(),
            "cv_std": scores.std(),
        }
        logger.info("CV F1: %.4f +/- %.4f", result["cv_mean"], result["cv_std"])
        return result

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict classes."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importance as a sorted DataFrame."""
        if hasattr(self.model, "feature_importances_"):
            importance = pd.DataFrame({
                "feature": self.feature_names,
                "importance": self.model.feature_importances_,
            }).sort_values("importance", ascending=False).reset_index(drop=True)
            return importance
        return pd.DataFrame()

    def save_model(self, filename: str = "model.pkl") -> Path:
        """Save model, scaler, and metadata to disk."""
        path = settings.MODELS_DIR / filename
        payload = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "model_type": self.model_type,
            "metrics": self.metrics,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        logger.info("Model saved to %s", path)
        return path

    def load_model(self, filename: str = "model.pkl") -> None:
        """Load model, scaler, and metadata from disk."""
        path = settings.MODELS_DIR / filename
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self.model = payload["model"]
        self.scaler = payload["scaler"]
        self.feature_names = payload["feature_names"]
        self.model_type = payload["model_type"]
        self.metrics = payload.get("metrics", {})
        logger.info("Model loaded from %s", path)
