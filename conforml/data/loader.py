"""Time series data loading functionality.

Provides classes and functions for loading time series data from various sources
including CSV files, pandas DataFrames, and numpy arrays.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, Tuple

class TimeSeriesLoader:
    """Class for loading and basic handling of time series data."""
    
    def __init__(self):
        self.data = None
        self.timestamps = None
    
    def load_from_csv(self, filepath: str, 
                      timestamp_col: str,
                      value_col: str,
                      date_format: Optional[str] = None) -> None:
        """Load time series data from a CSV file.
        
        Args:
            filepath: Path to the CSV file
            timestamp_col: Name of the timestamp column
            value_col: Name of the value column
            date_format: Optional format string for parsing dates
        """
        df = pd.read_csv(filepath)
        if date_format:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], format=date_format)
        else:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            
        self.data = df[value_col].values
        self.timestamps = df[timestamp_col].values
    
    def load_from_dataframe(self, df: pd.DataFrame,
                           timestamp_col: str,
                           value_col: str) -> None:
        """Load time series data from a pandas DataFrame.
        
        Args:
            df: Input DataFrame
            timestamp_col: Name of the timestamp column
            value_col: Name of the value column
        """
        self.data = df[value_col].values
        self.timestamps = df[timestamp_col].values
    
    def load_from_array(self, data: np.ndarray,
                       timestamps: Optional[np.ndarray] = None) -> None:
        """Load time series data from numpy arrays.
        
        Args:
            data: Array of time series values
            timestamps: Optional array of timestamps
        """
        self.data = data
        self.timestamps = timestamps if timestamps is not None else np.arange(len(data))
    
    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieve the loaded time series data.
        
        Returns:
            Tuple of (timestamps, values)
        """
        return self.timestamps, self.data
    
    def split_data(self, train_ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split the data into training and testing sets.
        
        Args:
            train_ratio: Ratio of data to use for training (default: 0.8)
            
        Returns:
            Tuple of (train_timestamps, train_values, test_timestamps, test_values)
        """
        split_idx = int(len(self.data) * train_ratio)
        return (self.timestamps[:split_idx], self.data[:split_idx],
                self.timestamps[split_idx:], self.data[split_idx:])