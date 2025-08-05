import numpy as np

class CoverageCalculator:
    """Calculate coverage rates for prediction intervals"""
    
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.total_samples = 0
        self.covered_samples = 0

    def update(self, y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray):
        """Update coverage statistics with new predictions"""
        if y_true.shape != lower.shape or y_true.shape != upper.shape:
            raise ValueError("Shapes of true values and intervals must match")
            
        covered = np.logical_and(y_true >= lower, y_true <= upper)
        self.covered_samples += np.sum(covered)
        self.total_samples += y_true.size

    def get_coverage_rate(self) -> float:
        """Return empirical coverage rate"""
        if self.total_samples == 0:
            return 0.0
        return self.covered_samples / self.total_samples


class IntervalWidthCalculator:
    """Track statistics about prediction interval widths"""
    
    def __init__(self):
        self.widths = []
        
    def update(self, lower: np.ndarray, upper: np.ndarray):
        """Accumulate interval width measurements"""
        self.widths.extend((upper - lower).flatten().tolist())

    def get_mean_width(self) -> float:
        """Calculate mean interval width"""
        return np.mean(self.widths) if self.widths else 0.0

    def get_quantile_width(self, q: float) -> float:
        """Get width at specific quantile"""
        return np.quantile(self.widths, q) if self.widths else 0.0


class RMSECalculator:
    """Calculate Root Mean Squared Error for point predictions"""
    
    def __init__(self):
        self.squared_errors = []
        self.total_samples = 0

    def update(self, y_true: np.ndarray, y_pred: np.ndarray):
        if y_true.shape != y_pred.shape:
            raise ValueError("True values and predictions must have same shape")
        self.squared_errors.extend(((y_true - y_pred) ** 2).tolist())
        self.total_samples += y_true.size

    def get_rmse(self) -> float:
        return np.sqrt(np.mean(self.squared_errors)) if self.squared_errors else 0.0

class MAPECalculator:
    """Calculate Mean Absolute Percentage Error for point predictions"""
    
    def __init__(self, epsilon: float = 1e-8):
        self.absolute_percentage_errors = []
        self.epsilon = epsilon
        self.total_samples = 0

    def update(self, y_true: np.ndarray, y_pred: np.ndarray):
        if y_true.shape != y_pred.shape:
            raise ValueError("True values and predictions must have same shape")
        ape = np.abs((y_true - y_pred) / (np.abs(y_true) + self.epsilon)) * 100
        self.absolute_percentage_errors.extend(ape.tolist())
        self.total_samples += y_true.size

    def get_mape(self) -> float:
        return np.mean(self.absolute_percentage_errors) if self.absolute_percentage_errors else 0.0