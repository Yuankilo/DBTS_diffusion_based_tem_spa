"""Reference ridership mapping and scenario-consistency log-likelihood module (Eq. 6)."""

from math import log, pi

import torch
from torch import nn
from torch.nn import functional as F


class ScenarioGuidance(nn.Module):
    """Estimate reference ridership and condition log-likelihood for reverse-diffusion guidance."""

    def __init__(self, nodes, conditions, width):
        super().__init__()
        if min(nodes, conditions, width) < 1:
            raise ValueError("Scenario guidance requires positive node, condition, and hidden widths.")
        self.reference_head = nn.Sequential(
            nn.Linear(conditions, width),
            nn.GELU(),
            nn.Linear(width, nodes),
        )
        self.condition_head = nn.Sequential(
            nn.Linear(nodes, width),
            nn.GELU(),
            nn.Linear(width, conditions),
        )

    def reference(self, condition):
        """Predict expected station-wise reference ridership x_0^{rf} from dynamic conditions."""
        return self.reference_head(condition)

    def log_probability(self, flow, condition):
        """Evaluate Gaussian log-likelihood log p(c | x) over normalized dynamic condition channels."""
        squared = (self.condition_head(flow) - condition).square()
        return -0.5 * (squared + log(2 * pi)).sum(dim=(1, 2))

    def training_loss(self, flow, condition):
        """Jointly fit reference regression and condition-consistency log-likelihood on training pairs."""
        ref_loss = F.mse_loss(self.reference(condition), flow)
        nll = -self.log_probability(flow, condition).mean() / (condition.shape[1] * condition.shape[2])
        return ref_loss + nll
