"""Prophet model for time series forecasting."""

import numpy as np
import pandas as pd
from prophet import Prophet
from typing import Optional, Tuple
from .base import TimeSeriesModel


class ProphetModel(TimeSeriesModel):
    """Facebook Prophet wrapper for conformal time series forecasting."""

    def __init__(self, yearly_seasonality: bool = True,
                 weekly_seasonality: bool = True,
                 daily_seasonality: bool = False,
                 seasonality_mode: str = "additive"):
        super().__init__()
        self.model = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            seasonality_mode=seasonality_mode
        )
        self.fitted_model = None
        self.train_data = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'ProphetModel':
        """
        Fit Prophet model.
        Args:
            X: Either pandas.DatetimeIndex or np.ndarray of shape (n_samples,)
            y: Target values (np.ndarray). If None, assumes X is a DataFrame with ['ds','y'].
        """
        if y is None:
            # Assume X is a DataFrame with ['ds', 'y']
            df = pd.DataFrame(X, columns=["ds", "y"])
        else:
            # Assume X is datetime-like, y is numeric
            df = pd.DataFrame({"ds": pd.to_datetime(X), "y": y})

        self.model = Prophet(
            yearly_seasonality=self.model.yearly_seasonality,
            weekly_seasonality=self.model.weekly_seasonality,
            daily_seasonality=self.model.daily_seasonality,
            seasonality_mode=self.model.seasonality_mode
        )
        self.model.fit(df)
        self.fitted_model = self.model
        self.train_data = df
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate point forecasts for given dates.
        Args:
            X: Future dates (array-like, datetime or string format).
        """
        self.check_is_fitted()
        future = pd.DataFrame({"ds": pd.to_datetime(X)})
        forecast = self.fitted_model.predict(future)
        return forecast["yhat"].values

    def predict_interval(self, X: np.ndarray,
                         confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate forecast intervals from Prophet.
        Args:
            X: Future dates
            confidence: Interval coverage (default 95%)
        """
        self.check_is_fitted()
        future = pd.DataFrame({"ds": pd.to_datetime(X)})
        forecast = self.fitted_model.predict(future)

        preds = forecast["yhat"].values
        lower = forecast[f"yhat_lower"].values
        upper = forecast[f"yhat_upper"].values

        return preds, lower, upper
