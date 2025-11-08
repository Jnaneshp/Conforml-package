"""Linear Regression model for time series forecasting with lag features."""

import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Optional, Tuple
from .base import TimeSeriesModel


class LinearTimeSeriesModel(TimeSeriesModel):
    """Linear Regression with lag features for time series forecasting."""

    def __init__(self, n_lags: int = 5):
        super().__init__()
        self.model = LinearRegression()
        self.n_lags = n_lags
        self.residuals_ = None

    def _create_lag_features(self, series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate lag features for univariate time series.
        Example: if n_lags=3, X_t = [y_{t-3}, y_{t-2}, y_{t-1}], target = y_t
        """
        X, y = [], []
        for i in range(self.n_lags, len(series)):
            X.append(series[i - self.n_lags:i])
            y.append(series[i])
        return np.array(X), np.array(y)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'LinearTimeSeriesModel':
        """
        Fit model.
        If y is None, treat X as a univariate time series and build lag features.
        """
        if y is None:
            X, y = self._create_lag_features(X)

        self.model.fit(X, y)
        preds = self.model.predict(X)
        self.residuals_ = y - preds  # store residuals for interval estimation
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict next values given lagged features.
        NOTE: User must provide lagged features of shape (n_samples, n_lags).
        """
        self.check_is_fitted()
        return self.model.predict(X)

    def predict_interval(self, X: np.ndarray,
                         confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prediction intervals using residual standard deviation.
        Assumes homoscedastic errors.
        """
        self.check_is_fitted()
        preds = self.model.predict(X)

        if self.residuals_ is None:
            raise ValueError("Residuals not available. Fit the model first.")

        sigma = np.std(self.residuals_)  # standard deviation of residuals
        z = 1.96 if confidence == 0.95 else 2.58  # 95% or 99%

        lower = preds - z * sigma
        upper = preds + z * sigma
        return preds, lower, upper
