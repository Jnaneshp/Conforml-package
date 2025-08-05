"""Base class for time series models.

Defines the interface that all time series models should implement.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Union, Tuple

class TimeSeriesModel(ABC):
    """Abstract base class for time series models."""
    
    def __init__(self):
        self.model = None
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'TimeSeriesModel':
        """Fit the model to the data.
        
        Args:
            X: Training data
            y: Target values (if applicable)
            
        Returns:
            self
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate point predictions.
        
        Args:
            X: Input data
            
        Returns:
            Array of predictions
        """
        pass
    
    @abstractmethod
    def predict_interval(self, X: np.ndarray, 
                        confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate prediction intervals.
        
        Args:
            X: Input data
            confidence: Confidence level for intervals (default: 0.95)
            
        Returns:
            Tuple of (predictions, lower_bounds, upper_bounds)
        """
        pass
    
    def check_is_fitted(self) -> None:
        """Check if the model has been fitted.
        
        Raises:
            ValueError: If the model has not been fitted
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions.")
    
    def get_params(self) -> dict:
        """Get model parameters.
        
        Returns:
            Dictionary of model parameters
        """
        return self.__dict__
    
    def set_params(self, **params) -> 'TimeSeriesModel':
        """Set model parameters.
        
        Args:
            **params: Parameters to set
            
        Returns:
            self
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self