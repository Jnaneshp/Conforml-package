import numpy as np
from sklearn.linear_model import LinearRegression
from .base import TimeSeriesModel

class LinearRegressionModel(TimeSeriesModel):
    def __init__(self):
        super().__init__()
        self.model = LinearRegression()

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.check_is_fitted()
        return self.model.predict(X)

    def predict_interval(self, X: np.ndarray, confidence: float = 0.95):
        self.check_is_fitted()
        preds = self.model.predict(X)
        # Calculate residuals using training data
        train_preds = self.model.predict(self.model._X)
        residuals = self.model._y - train_preds
        std = np.std(residuals)
        from scipy.stats import norm
        z = norm.ppf(0.5 + confidence / 2)
        lower = preds - z * std
        upper = preds + z * std
        return preds, lower, upper