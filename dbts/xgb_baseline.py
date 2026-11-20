"""Station-wise recursive XGBoost baseline for multi-step metro ridership forecasting."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import xgboost as xgb

from dbts.metrics import point
from dbts.runner import dump, prepare


def supervised(series, origins, history):
    origins = np.asarray(origins)
    if series.ndim != 1 or origins.ndim != 1 or not len(origins) or origins.min() < history or origins.max() >= len(series):
        raise ValueError("Invalid one-step training origins.")
    return np.stack([series[o - history : o] for o in origins]), series[origins].copy()


def recursive_forecast(model, history, horizon):
    if history.ndim != 2 or history.shape[1] < 1 or horizon < 1 or not np.isfinite(history).all():
        raise ValueError("Expected finite history array [origins, history].")
    window = history.copy()
    forecasts = []
    for _ in range(horizon):
        prediction = np.asarray(model.predict(window), dtype=np.float32)
        forecasts.append(prediction)
        window = np.concatenate([window[:, 1:], prediction[:, None]], axis=1)
    return np.stack(forecasts, axis=1)


def run(config, output):
    c = json.loads(Path(config).read_text(encoding="utf-8"))
    out = Path(output)
    if out.exists():
        raise ValueError(f"Output directory already exists: {out}")
    times, stations, raw, norm, _, origins, scaler, dynamic, static, metadata = prepare(c)
    out.mkdir(parents=True)
    (out / "models").mkdir()
    dump(out / "config.json", c)
    dump(out / "metadata.json", metadata)

    h, p = c["history"], c["horizon"]
    outputs = {s: [] for s in ("val", "test")}
    start = time.perf_counter()
    for i, station in enumerate(stations):
        x, y = supervised(norm[:, i], origins["train"], h)
        model = xgb.XGBRegressor(random_state=c["seed"], **c["xgb"])
        model.fit(x, y)
        model.save_model(out / "models" / f"{i:03d}.ubj")
        for split in outputs:
            history = np.stack([norm[o - h : o, i] for o in origins[split]])
            outputs[split].append(recursive_forecast(model, history, p))

    summary = {"seconds": time.perf_counter() - start, "stations": len(stations)}
    observed = [stations.index(s) for s in metadata.get("observed_station_ids", stations)]
    for split, predictions in outputs.items():
        prediction = scaler.inverse(np.stack(predictions, axis=-1))
        truth = np.stack([raw[o : o + p] for o in origins[split]])
        np.savez_compressed(out / f"{split}_point_predictions.npz", point=prediction, truth=truth, origins=origins[split], stations=stations)
        metrics = {
            "point": point(truth, prediction),
            "observed_nodes": point(truth[..., observed], prediction[..., observed]),
            "station": {s: point(truth[..., i], prediction[..., i]) for i, s in enumerate(stations)},
            "horizon": {str(i + 1): point(truth[:, i], prediction[:, i]) for i in range(p)},
        }
        dump(out / f"{split}_metrics.json", metrics)
        summary[split] = metrics["point"]
    dump(out / "xgb_run_summary.json", summary)
    print(json.dumps(summary), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run XGBoost baseline.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run(args.config, args.out)
