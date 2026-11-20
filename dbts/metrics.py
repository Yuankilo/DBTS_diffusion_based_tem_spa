"""Point forecasting (RMSE, MAE, R2, WMAPE) and probabilistic (CRPS, PICP, MPIW) metrics."""

import numpy as np


def point(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if y.shape != p.shape or y.size == 0 or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Metric inputs must be finite and shape-matched.")
    e = p - y
    denom = np.sum(np.abs(y))
    sst = np.sum((y - y.mean()) ** 2)
    return {
        "rmse": float(np.sqrt(np.mean(e**2))),
        "mae": float(np.mean(np.abs(e))),
        "r2": float(1.0 - np.sum(e**2) / sst) if sst > 0 else None,
        "wmape": float(np.sum(np.abs(e)) / denom) if denom > 0 else None,
        "n": int(y.size),
    }


def probabilistic(y, samples, train_range, levels=(0.5, 0.8, 0.95)):
    y = np.asarray(y)
    s = np.asarray(samples)
    if s.shape[1:] != y.shape or s.shape[0] < 2 or not np.isfinite(s).all() or not np.isfinite(y).all():
        raise ValueError("Expected >=2 finite forecast trajectories [M, B, P, N] and targets [B, P, N].")
    ordered = np.sort(s, axis=0)
    m = len(s)
    weights = (2 * np.arange(1, m + 1) - m - 1).reshape((m,) + (1,) * y.ndim)
    crps = np.mean(np.abs(s - y), axis=0) - np.sum(weights * ordered, axis=0) / (m**2)
    result = {"crps": float(crps.mean()), "intervals": {}, "pinball": {}}
    tr = np.asarray(train_range)
    if (tr < 0).any() or not np.isfinite(tr).all():
        raise ValueError("Invalid training range array.")
    for q in (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95):
        err = y - np.quantile(s, q, axis=0)
        result["pinball"][str(q)] = float(np.maximum(q * err, (q - 1.0) * err).mean())
    for level in levels:
        alpha = 1.0 - level
        lo, hi = np.quantile(s, [alpha / 2.0, 1.0 - alpha / 2.0], axis=0)
        width = hi - lo
        score = width + 2.0 / alpha * (lo - y) * (y < lo) + 2.0 / alpha * (y - hi) * (y > hi)
        valid = tr > 0
        result["intervals"][str(level)] = {
            "picp": float(((lo <= y) & (y <= hi)).mean()),
            "mpiw": float(width.mean()),
            "pinaw": float((width[..., valid] / tr[valid]).mean()) if valid.any() else None,
            "interval_score": float(score.mean()),
        }
    return result


def evaluate(y, samples, stations, train_range):
    p = samples.mean(axis=0)
    return {
        "point": point(y, p),
        "probabilistic": probabilistic(y, samples, train_range),
        "station": {s: point(y[..., i], p[..., i]) for i, s in enumerate(stations)},
        "horizon": {str(i + 1): point(y[:, i], p[:, i]) for i in range(y.shape[1])},
    }
