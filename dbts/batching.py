"""Minibatch sampling utilities for temporal windows and contrastive negative pairs."""

import numpy as np


def sample_indices(origins, size, rng, window=None, max_attempts=1000):
    origins = np.asarray(origins)
    if not 1 <= size <= len(origins):
        raise ValueError("Invalid batch size.")
    if window is not None and (size < 2 or window < 1):
        raise ValueError("Contrastive batching requires at least two windows.")
    for attempt in range(max_attempts):
        ids = rng.choice(len(origins), size=size, replace=False)
        if window is None:
            return ids, 0
        chosen = origins[ids]
        valid = np.abs(chosen[:, None] - chosen[None, :]) >= window
        if valid.any(axis=1).all():
            return ids, attempt
    raise ValueError("Unable to construct a batch with nonoverlapping negative pairs.")
