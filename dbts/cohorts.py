"""Period and station-subset evaluation utilities for disruption analysis."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dbts.metrics import evaluate
from dbts.runner import dump


def cohort_mask(target_times, start, end):
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    if start >= end:
        raise ValueError("Cohort start time must precede end time.")
    return (target_times >= start.to_datetime64()) & (target_times < end.to_datetime64())


def score_cohorts(y, samples, target_times, stations, train_range, cohorts):
    if target_times.shape != y.shape[:2]:
        raise ValueError("Target timestamp shape mismatch.")
    if cohorts.cohort_id.duplicated().any():
        raise ValueError("Duplicate cohort_id found.")
    results = {}
    for row in cohorts.itertuples():
        mask = cohort_mask(target_times, row.start, row.end)
        spec = getattr(row, "station_ids", None)
        selected = stations if spec is None or pd.isna(spec) else json.loads(spec)
        nodes = [stations.index(name) for name in selected]
        if not mask.any():
            results[row.cohort_id] = {"status": "NO_SAMPLES"}
            continue
        ys = y[..., nodes]
        ss = samples[..., nodes]
        result = evaluate(ys[mask][:, None, :], ss[:, mask][:, :, None, :], selected, np.asarray(train_range)[nodes])
        result.update(
            status="SCORED",
            target_exposures=int(mask.sum()),
            distinct_timestamps=int(np.unique(target_times[mask]).size),
            station_ids=selected,
            start=row.start,
            end=row.end,
        )
        results[row.cohort_id] = result
    return results


def run_cohorts(run_dir, cohort_csv, out_dir):
    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise ValueError(f"Output directory already exists: {out_dir}")
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    data = np.load(run_dir / "test_predictions.npz")
    split = np.load(run_dir / "split.npz")
    times = pd.to_datetime(split["timestamps"]).to_numpy()
    idx = data["origins"][:, None] + np.arange(data["truth"].shape[1])[None]
    results = score_cohorts(
        data["truth"],
        data["samples"],
        times[idx],
        data["stations"].tolist(),
        np.array(metadata["normalization"]["observed_train_range"]),
        pd.read_csv(cohort_csv),
    )
    out_dir.mkdir(parents=True)
    dump(out_dir / "cohort_metrics.json", {"run": str(run_dir), "cohorts": results})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate forecasts on dated station cohorts.")
    parser.add_argument("--run", required=True)
    parser.add_argument("--cohorts", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run_cohorts(args.run, args.cohorts, args.out)
