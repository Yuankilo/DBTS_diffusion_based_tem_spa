"""Recurrent baseline models (LSTM and GRU) for multi-step metro ridership forecasting."""

import torch
from torch import nn


class RecurrentPointForecast(nn.Module):
    def __init__(self, nodes, history, horizon, kind="lstm", hidden=64, layers=2):
        super().__init__()
        if kind not in ("lstm", "gru"):
            raise ValueError("kind must be 'lstm' or 'gru'.")
        if min(nodes, history, horizon, hidden, layers) < 1:
            raise ValueError("Model dimensions must be positive.")
        self.nodes, self.history, self.horizon = nodes, history, horizon
        self.recurrent = (nn.LSTM if kind == "lstm" else nn.GRU)(nodes, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, nodes * horizon)

    def forward(self, history):
        if history.ndim != 3 or history.shape[1:] != (self.history, self.nodes):
            raise ValueError("history must have shape [B, history, N].")
        encoded, _ = self.recurrent(history)
        return torch.relu(self.head(encoded[:, -1])).reshape(history.shape[0], self.horizon, self.nodes)

    def loss(self, history, target):
        prediction = self(history)
        if prediction.shape != target.shape:
            raise ValueError("target shape mismatch.")
        loss = (prediction - target).square().mean()
        return loss, {"total": loss.detach(), "mse": loss.detach()}
