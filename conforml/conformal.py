from abc import ABC, abstractmethod
import numpy as np

class SplitConformal:
    def __init__(self, model, alpha=0.1):
        self.model = model
        self.alpha = alpha
        self.calibration_scores = None

    def fit(self, X, y):
        self.model.fit(X, y)
        predictions = self.model.predict(X)
        residuals = np.abs(y - predictions)
        self.calibration_scores = np.sort(residuals)
        self.quantile = np.quantile(self.calibration_scores, 1 - self.alpha, interpolation='higher')

    def predict(self, X):
        predictions = self.model.predict(X)
        lower = predictions - self.quantile
        upper = predictions + self.quantile
        return predictions, lower, upper

class AdaptiveConformal(SplitConformal):
    def __init__(self, model, alpha=0.1, threshold=0.01):
        super().__init__(model, alpha)
        self.threshold = threshold
        self.adaptive_quantile = None

    def fit(self, X, y):
        super().fit(X, y)
        self.adaptive_quantile = self.quantile * (1 + self.threshold)

    def predict(self, X):
        predictions = self.model.predict(X)
        lower = predictions - self.adaptive_quantile
        upper = predictions + self.adaptive_quantile
        return predictions, lower, upper