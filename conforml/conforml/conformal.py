from abc import ABC, abstractmethod
import numpy as np

class ConformityScore(ABC):
    """Base class for modular conformity score implementations"""
    
    @abstractmethod
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute nonconformity scores"""
        pass

class AbsoluteErrorScore(ConformityScore):
    def compute(self, y_true, y_pred):
        return np.abs(y_true - y_pred)

class SignedErrorScore(ConformityScore):
    def compute(self, y_true, y_pred):
        return y_pred - y_true

class ConformalPredictor(ABC):
    """Base class for conformal prediction wrappers"""
    
    def __init__(self, model, alpha: float = 0.1, conformity_score: ConformityScore = None):
        self.model = model
        self.alpha = alpha
        self.calibration_scores = None
        self.conformity_score = conformity_score or AbsoluteErrorScore()

    def _get_scores(self, y_true, y_pred):
        return self.conformity_score.compute(y_true, y_pred)

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the conformal predictor"""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray):
        """Predict with conformal intervals"""
        pass

class SplitConformal(ConformalPredictor):
    """Split conformal prediction implementation for base time series models"""
    
    def __init__(self, model: TimeSeriesModel, alpha: float = 0.1):
        if not isinstance(model, TimeSeriesModel):
            raise ValueError("Model must implement TimeSeriesModel interface")
        super().__init__(model, alpha)
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        # Split data into proper training and calibration sets
        split_idx = int(len(X) * 0.8)
        X_train, X_cal = X[:split_idx], X[split_idx:]
        y_train, y_cal = y[:split_idx], y[split_idx:]
        
        # Train base model using standardized interface
        self.model.fit(X_train, y_train)
        
        # Calculate calibration scores using model-agnostic prediction
        cal_preds = self.model.predict(X_cal)
        self.calibration_scores = self._get_scores(y_cal, cal_preds)
        
    def predict(self, X: np.ndarray):
        # Generate predictions through base model interface
        point_preds = self.model.predict(X)
        
        # Calculate conformal intervals
        quantile = np.quantile(self.calibration_scores, 1 - self.alpha)
        lower = point_preds - quantile
        upper = point_preds + quantile
        
        return point_preds, lower, upper