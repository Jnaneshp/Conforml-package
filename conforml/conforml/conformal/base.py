"""Base class for conformal prediction methods.

Defines the interface and common functionality for all conformal prediction
implementations.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Tuple, Union
from ..models.base import TimeSeriesModel

class ConformalPredictor(ABC):
    """Abstract base class for conformal prediction methods."""
    
    def __init__(self, model: TimeSeriesModel, alpha: float = 0.1):
        """Initialize conformal predictor.
        
        Args:
            model: The underlying time series model
            alpha: Significance level (default: 0.1)
        """
        self.model = model
        self.alpha = alpha
        self.calibration_scores = None
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ConformalPredictor':
        """Fit the conformal predictor.
        
        Args:
            X: Training data
            y: Target values
            
        Returns:
            self
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate predictions with conformal prediction intervals.
        
        Args:
            X: Input data
            
        Returns:
            Tuple of (predictions, lower_bounds, upper_bounds)
        """
        pass
    
    def _compute_conformity_scores(self, y_true: np.ndarray,
                                  y_pred: np.ndarray) -> np.ndarray:
        """Compute nonconformity scores.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Array of nonconformity scores
        """
        return np.abs(y_true - y_pred)
    
    def _get_quantile(self, scores: np.ndarray) -> float:
        """Compute the (1-alpha) quantile of the calibration scores.
        
        Args:
            scores: Array of calibration scores
            
        Returns:
            Quantile value
        """
        return np.quantile(scores, 1 - self.alpha)
    
    def check_is_fitted(self) -> None:
        """Check if the predictor has been fitted.
        
        Raises:
            ValueError: If the predictor has not been fitted
        """
        if not self.is_fitted:
            raise ValueError("Predictor must be fitted before making predictions.")
    
    def get_coverage_stats(self, y_true: np.ndarray,
                          lower: np.ndarray,
                          upper: np.ndarray) -> dict:
        """Compute coverage statistics.
        
        Args:
            y_true: True values
            lower: Lower prediction bounds
            upper: Upper prediction bounds
            
        Returns:
            Dictionary with coverage statistics
        """
        in_interval = (y_true >= lower) & (y_true <= upper)
        coverage = np.mean(in_interval)
        avg_width = np.mean(upper - lower)
        
        return {
            'empirical_coverage': coverage,
            'target_coverage': 1 - self.alpha,
            'average_interval_width': avg_width,
            'number_of_predictions': len(y_true)
        }

    @abstractmethod
    def save(self, path: str) -> None:
        """Save the conformal predictor to disk.
        
        Args:
            path: File path to save the predictor
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load a conformal predictor from disk.
        
        Args:
            path: File path to load the predictor from
        """
        pass

    def _get_serializable_params(self) -> dict:
        """Get parameters needed for serialization.
        
        Returns:
            Dictionary of serializable parameters
        """
        return {
            'alpha': self.alpha,
            'calibration_scores': self.calibration_scores.tolist() if self.calibration_scores is not None else None,
            'model_params': self.model.get_params() if hasattr(self.model, 'get_params') else {}
        }