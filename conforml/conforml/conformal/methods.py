from .base import ConformalPredictor
from ..models.base import TimeSeriesModel
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

class CVPlusConformal(ConformalPredictor):
    """Cross-validation+ conformal prediction for time series.

    Suitable for stationary time series with local validity guarantees.
    """

    def __init__(self, model: TimeSeriesModel, alpha: float = 0.1, n_folds: int = 5):
        super().__init__(model, alpha)
        self.n_folds = n_folds

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'CVPlusConformal':
        scores = []
        splitter = TimeSeriesSplit(n_splits=self.n_folds)
        for train_idx, cal_idx in splitter.split(X):
            X_train, X_cal = X[train_idx], X[cal_idx]
            y_train, y_cal = y[train_idx], y[cal_idx]
            self.model.fit(X_train, y_train)
            preds = self.model.predict(X_cal)
            scores.extend(self._compute_conformity_scores(y_cal, preds))
        self.calibration_scores = np.sort(scores)
        self.quantile = self._get_quantile(self.calibration_scores)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray):
        if not self.is_fitted:
            raise RuntimeError("CVPlusConformal must be fitted before prediction.")
        preds = self.model.predict(X)
        lower = preds - self.quantile
        upper = preds + self.quantile
        return preds, lower, upper

    def save(self, path: str) -> None:
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'alpha': self.alpha,
                'n_folds': self.n_folds,
                'calibration_scores': self.calibration_scores,
                'quantile': self.quantile,
                'is_fitted': self.is_fitted
            }, f)

    def load(self, path: str) -> None:
        import pickle
        with open(path, 'rb') as f:
            state = pickle.load(f)
            self.model = state['model']
            self.alpha = state['alpha']
            self.n_folds = state['n_folds']
            self.calibration_scores = state['calibration_scores']
            self.quantile = state['quantile']
            self.is_fitted = state['is_fitted']


class AdaptiveConformal(ConformalPredictor):
    """Adaptive conformal prediction for non-stationary time series.

    Implements the adaptive conformal framework from Gibbs & Candès (2021).
    """

    def __init__(self, model: TimeSeriesModel, alpha: float = 0.1, 
                 threshold: float = 0.05):
        super().__init__(model, alpha)
        self.threshold = threshold
        self.weights = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'AdaptiveConformal':
        self.model.fit(X, y)
        preds = self.model.predict(X)
        residuals = np.abs(y - preds)
        
        # Calculate exponentially decaying weights
        t = len(residuals)
        decay = (1 - self.threshold) ** np.arange(t)
        self.weights = decay[::-1] / decay.sum()  # Reverse for time-ordered weighting
        
        sorted_residuals = np.sort(residuals)
        cum_weights = np.cumsum(self.weights)
        effective_quantile = 1 - self.alpha * cum_weights[-1]
        idx = np.searchsorted(cum_weights, effective_quantile)
        self.quantile = sorted_residuals[min(idx, len(sorted_residuals)-1)]
        self.is_fitted = True
        return self