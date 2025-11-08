"""
XGBoost-based time series model for conformal prediction framework.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from xgboost import XGBRegressor
from .base import TimeSeriesModel


class XGBoostTimeSeriesModel(TimeSeriesModel):
    """XGBoost model for univariate time series forecasting."""

    def __init__(
        self,
        n_lags: int = 7,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 5,
        subsample: float = 0.9,
        colsample_bytree: float = 0.9,
        random_state: int = 42,
    ):
        super().__init__()
        self.n_lags = n_lags
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=1.0,
            random_state=random_state,
        )

        # bookkeeping
        self.train_series = None        # numpy array of training series used for fit
        self.history_series = None      # list used for iterative forecasting
        self.residuals_ = None

    def _create_lag_features(self, series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create lag features for supervised learning from a 1D series."""
        X, y = [], []
        for i in range(self.n_lags, len(series)):
            X.append(series[i - self.n_lags : i])
            y.append(series[i])
        return np.array(X), np.array(y)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "XGBoostTimeSeriesModel":
        """
        Fit the XGBoost model.

        Expected usage in conformal CV:
          - X : array-like indices (ignored internally)
          - y : 1D array of target values corresponding to the indices
        """
        # If y provided, use it as the underlying series
        if y is None:
            # if user passed a raw series to fit (non-CV mode)
            series = np.asarray(X).flatten()
        else:
            series = np.asarray(y).flatten()

        # store training series for use at prediction time (iterative forecasting)
        self.train_series = series.copy()
        self.history_series = list(series.copy())

        X_train, y_train = self._create_lag_features(series)
        if len(X_train) == 0:
            raise ValueError("Series too short for the configured n_lags")

        self.model.fit(X_train, y_train)
        self.is_fitted = True

        # residuals computed on training feature matrix
        self.residuals_ = y_train - self.model.predict(X_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict next values.

        - If X is a 2D array with shape (n_samples, n_lags) assume it's already lag-feature matrix.
        - If X is a 2D/1D array of time indices (e.g. shape (m,1) or (m,)), produce m iterative predictions
          using history stored during fit (autoregressive iterative forecasting).
        - If X is a full raw series (1D) we create lag-features and predict for all available rows.
        """
        self.check_is_fitted()

        X_arr = np.asarray(X)

        # Case A: user passed a proper lag-feature matrix
        if X_arr.ndim == 2 and X_arr.shape[1] == self.n_lags:
            return self.model.predict(X_arr)

        # Case B: user passed a 1D raw series (we create lag features)
        if X_arr.ndim == 1 and X_arr.size >= self.n_lags:
            X_feat, _ = self._create_lag_features(X_arr)
            return self.model.predict(X_feat)

        # Case C: user passed indices (common in CV+). We assume X_arr is shape (m,1) or (m,)
        # We'll produce m iterative one-step-ahead predictions using stored history.
        flat = X_arr.flatten()

        # Must maintain a temporary copy of history for iterative forecasting
        history = list(self.history_series) if self.history_series is not None else []
        preds = []

        for _ in range(len(flat)):
            if len(history) < self.n_lags:
                # not enough history to predict
                raise ValueError("Not enough history for iterative prediction with n_lags=%d" % self.n_lags)

            x_row = np.array(history[-self.n_lags :]).reshape(1, -1)
            p = float(self.model.predict(x_row)[0])
            preds.append(p)
            history.append(p)  # iterative: use own prediction for next step

        return np.array(preds)

    def predict_interval(
        self, X: np.ndarray, confidence: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict intervals using residual quantiles."""
        self.check_is_fitted()
        preds = self.predict(X)

        # Compute residual quantiles from training residuals
        alpha = 1 - confidence
        lower_q = np.quantile(self.residuals_, alpha / 2)
        upper_q = np.quantile(self.residuals_, 1 - alpha / 2)

        lower = preds + lower_q
        upper = preds + upper_q
        return preds, lower, upper
