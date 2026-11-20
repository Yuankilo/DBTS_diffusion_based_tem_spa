"""Transformer encoder baseline for multi-step metro ridership forecasting."""

from torch import nn


class TransformerPointForecast(nn.Module):
    def __init__(self, nodes, history, horizon, heads=3, feedforward=1024, layers=4):
        super().__init__()
        if min(nodes, history, horizon, heads, feedforward, layers) < 1 or nodes % heads:
            raise ValueError("nodes must be positive and divisible by heads.")
        self.nodes, self.history, self.horizon = nodes, history, horizon
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=nodes,
            nhead=heads,
            dim_feedforward=feedforward,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.fc = nn.Linear(nodes, nodes * horizon)

    def forward(self, history):
        if history.ndim != 3 or history.shape[1:] != (self.history, self.nodes):
            raise ValueError("history shape mismatch.")
        encoded = self.transformer_encoder(history)
        return self.fc(encoded[:, -1]).reshape(len(history), self.horizon, self.nodes)

    def loss(self, history, target):
        prediction = self(history)
        if target.shape != prediction.shape:
            raise ValueError("target shape mismatch.")
        loss = (prediction - target).square().mean()
        return loss, {"total": loss.detach(), "mse": loss.detach()}
