import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from .base import TimeSeriesModel

class SARIMAModel(TimeSeriesModel):
    """
    A wrapper for the statsmodels SARIMAX model.
    """
    def __init__(self, order=(1,0,0), seasonal_order=(0,0,0,0)):
        super().__init__()
        self.order = order
        self.seasonal_order = seasonal_order
        self.model_fit = None

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """
        Fits the SARIMAX model to the training data.
        
        Args:
            X (np.ndarray): The input features (unused by SARIMAX).
            y (np.ndarray): The time series values to fit the model on.
        """
        self.model = SARIMAX(y, order=self.order, seasonal_order=self.seasonal_order)
        self.model_fit = self.model.fit(disp=False)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Makes point predictions.
        
        Args:
            X (np.ndarray): An array of time indices for which to make predictions.
        
        Returns:
            np.ndarray: The point predictions.
        """
        self.check_is_fitted()
        start = int(X[0])
        end = int(X[-1])
        preds = self.model_fit.predict(start=start, end=end)
        return preds

    def predict_interval(self, X: np.ndarray, confidence: float = 0.95):
        """
        Makes predictions with confidence intervals.
        
        Args:
            X (np.ndarray): An array of time indices for which to make predictions.
            confidence (float): The desired confidence level for the interval.
        
        Returns:
            tuple: A tuple containing (predictions, lower_bound, upper_bound).
        """
        self.check_is_fitted()
        start = int(X[0])
        end = int(X[-1])
        pred_res = self.model_fit.get_prediction(start=start, end=end)
        
        preds = pred_res.predicted_mean
        conf_int = pred_res.conf_int(alpha=1-confidence)
        
        # FIX: Changed from .iloc to NumPy slicing to handle array format.
        lower = conf_int[:, 0]
        upper = conf_int[:, 1]
        
        return preds, lower, upper