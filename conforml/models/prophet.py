import numpy as np
from prophet import Prophet
from .base import TimeSeriesModel
import pandas as pd

class ProphetModel(TimeSeriesModel):
    def __init__(self):
        super().__init__()
        self.model = Prophet()
        self.is_fitted = False
        self.train_df = None

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        # X is expected to be timestamps, y is values
        df = pd.DataFrame({'ds': X.flatten(), 'y': y})
        self.model = Prophet()  # Always instantiate a new Prophet object
        self.model.fit(df)
        self.is_fitted = True
        self.train_df = df
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.check_is_fitted()
        future = pd.DataFrame({'ds': X.flatten()})
        forecast = self.model.predict(future)
        return forecast['yhat'].values

    def predict_interval(self, X: np.ndarray, confidence: float = 0.95):
        self.check_is_fitted()
        future = pd.DataFrame({'ds': X.flatten()})
        forecast = self.model.predict(future)
        preds = forecast['yhat'].values
        lower = forecast['yhat_lower'].values
        upper = forecast['yhat_upper'].values
        return preds, lower, upper