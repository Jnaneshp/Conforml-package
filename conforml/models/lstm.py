import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # use last time step
        return out


class LSTMForecaster:
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1, dropout=0.2, lr=0.001, epochs=50, batch_size=32, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout).to(self.device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def _create_sequences(self, data, seq_length):
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            x = data[i:(i + seq_length)]
            y = data[i + seq_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    def fit(self, series, seq_length=10):
        series = np.array(series, dtype=np.float32).reshape(-1, 1)
        X, y = self._create_sequences(series, seq_length)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for xb, yb in loader:
                xb = xb.unsqueeze(-1)  # add feature dim
                self.optimizer.zero_grad()
                preds = self.model(xb)
                loss = self.criterion(preds, yb)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {total_loss/len(loader):.6f}")

    def predict(self, series, seq_length=10, steps=1):
        self.model.eval()
        series = np.array(series, dtype=np.float32).reshape(-1, 1).tolist()
        predictions = []
        for _ in range(steps):
            seq = torch.tensor(series[-seq_length:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(self.device)
            with torch.no_grad():
                pred = self.model(seq).cpu().numpy().flatten()[0]
            predictions.append(pred)
            series.append([pred])
        return np.array(predictions)
