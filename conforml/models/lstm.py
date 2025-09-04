import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
from .base import TimeSeriesModel

class LSTMModel(TimeSeriesModel):
    def __init__(self, input_shape=(1,1), units=50, epochs=10, batch_size=32):
        super().__init__()
        self.input_shape = input_shape
        self.units = units
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        # Reshape X for LSTM: (samples, timesteps, features)
        X = X.reshape((X.shape[0], self.input_shape[0], self.input_shape[1]))
        self.model = Sequential()
        self.model.add(Input(shape=self.input_shape))
        self.model.add(LSTM(self.units))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mse')
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.check_is_fitted()
        X = X.reshape((X.shape[0], self.input_shape[0], self.input_shape[1]))
        preds = self.model.predict(X, verbose=0)
        return preds.flatten()

    def predict_interval(self, X: np.ndarray, confidence: float = 0.95):
        preds = self.predict(X)
        # Dummy intervals: mean +/- std
        std = np.std(preds)
        lower = preds - std
        upper = preds + std
        return preds, lower, upper