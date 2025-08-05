import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from .base import TimeSeriesModel

class ARIMAModel(TimeSeriesModel):
    def __init__(self, order=(1,1,1), seasonal_order=(0,0,0,0)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.result = None
        self.fitted = False

    def fit(self, X, y):
        if y is None or len(y) == 0:
            raise ValueError("Target y cannot be empty")
        self.model = ARIMA(y, order=self.order, seasonal_order=self.seasonal_order)
        self.result = self.model.fit()
        self.fitted = True

    def predict(self, X):
        if not self.fitted:
            raise RuntimeError("Model is not fitted yet!")
        forecast = self.result.get_forecast(steps=len(X))
        mean = forecast.predicted_mean
        return mean

    def predict_interval(self, X, confidence=0.95):
        if not self.fitted:
            raise RuntimeError("Model is not fitted yet!")
        forecast = self.result.get_forecast(steps=len(X))
        mean = forecast.predicted_mean
        conf_int = forecast.conf_int(alpha=1-confidence)
        # Handle both DataFrame and numpy array for conf_int
        if hasattr(conf_int, 'iloc'):
            lower = conf_int.iloc[:, 0].values
            upper = conf_int.iloc[:, 1].values
        else:
            lower = conf_int[:, 0]
            upper = conf_int[:, 1]
        return mean, lower, upper
