"""Time series data preprocessing functionality.

Provides classes and functions for preprocessing time series data including
scaling, normalization, and feature engineering.
"""

import numpy as np
from typing import Optional, Tuple, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class TimeSeriesPreprocessor:
    """Class for preprocessing time series data."""
    
    def __init__(self):
        self.scaler = None
        self.window_size = None
    
    def fit_scaler(self, data: np.ndarray, method: str = 'standard') -> None:
        """Fit a scaler to the data.
        
        Args:
            data: Input time series data
            method: Scaling method ('standard' or 'minmax')
        """
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
            
        # Reshape for sklearn compatibility
        reshaped_data = data.reshape(-1, 1)
        self.scaler.fit(reshaped_data)
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform the data using the fitted scaler.
        
        Args:
            data: Input time series data
            
        Returns:
            Scaled data
        """
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit_scaler first.")
            
        reshaped_data = data.reshape(-1, 1)
        return self.scaler.transform(reshaped_data).flatten()
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform scaled data back to original scale.
        
        Args:
            data: Scaled time series data
            
        Returns:
            Data in original scale
        """
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit_scaler first.")
            
        reshaped_data = data.reshape(-1, 1)
        return self.scaler.inverse_transform(reshaped_data).flatten()
    
    def create_sequences(self, data: np.ndarray, 
                        window_size: int,
                        horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series prediction.
        
        Args:
            data: Input time series data
            window_size: Number of time steps to use as input
            horizon: Number of time steps to predict
            
        Returns:
            Tuple of (X, y) where X contains input sequences and y contains target values
        """
        self.window_size = window_size
        X, y = [], []
        
        for i in range(len(data) - window_size - horizon + 1):
            X.append(data[i:(i + window_size)])
            y.append(data[(i + window_size):(i + window_size + horizon)])
        
        return np.array(X), np.array(y)
    
    def add_time_features(self, timestamps: np.ndarray) -> np.ndarray:
        """Extract time-based features from timestamps.
        
        Args:
            timestamps: Array of timestamps
            
        Returns:
            Array of time features (hour, day, month, etc.)
        """
        # Convert to pandas datetime if not already
        dates = pd.to_datetime(timestamps)
        
        # Extract various time features
        features = np.column_stack([
            dates.hour.values,
            dates.day.values,
            dates.month.values,
            dates.dayofweek.values,
            dates.quarter.values
        ])
        
        return features