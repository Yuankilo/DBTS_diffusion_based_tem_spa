"""Training and evaluation entry point for DBTS experiments."""

import argparse
import copy
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dbts.batching import sample_indices
from dbts.data import (
    Scaler,
    align_fixed_static,
    align_graph,
    align_static,
    asof_features,
    delayed_daily_observations,
    periodic_features,
    read_flow,
    split_origins,
)
from dbts.diffusion import Diffusion
from dbts.metrics import evaluate
from dbts.model import DBTS, SequenceDiffusionBaseline
from dbts.training import Trainer, seed_all

ROOT = Path(__file__).resolve().parents[1]


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def prepare(c):
    dc = c["data"]
    h, p = c["history"], c["horizon"]
    metadata = {}
    interval = dc.get("interval_minutes", 30)
    if isinstance(interval, bool) or interval not in (30, 60):
        raise ValueError("Supported interval_minutes are 30 and 60.")

    if dc["kind"] == "synthetic":
        rng = np.random.default_rng(dc.get("seed", 9001))
        n = dc.get("nodes", 8)
        length = dc.get("length", 360)
        times = pd.date_range("2022-01-01", periods=length, freq=f"{interval}min")
        values = (
            20.0
            + 8.0 * np.sin(np.arange(length)[:, None] * 2.0 * np.pi / (1440 // interval) + np.arange(n)[None] * 0.3)
            + rng.normal(size=(length, n))
        ).astype(np.float32)
        stations = [f"s{i}" for i in range(n)]
        adj = (np.eye(n, k=1) + np.eye(n, k=-1)).astype(np.float32)
        metadata = {"kind": "synthetic", "seed": dc.get("seed", 9001)}
    else:
        fp = (ROOT / dc["flow"]).resolve()
        gp = (ROOT / dc["adjacency"]).resolve()
        times, stations, values, quality = read_flow(fp, dc.get("trailing_partial", "error"), interval_minutes=interval)
        adj = align_graph(gp, stations)
        metadata = {"kind": "real", "flow": str(dc["flow"]), "adjacency": str(dc["adjacency"]), **quality}

    metadata["interval_minutes"] = interval
    if "observed_station_list" in dc:
        path = (ROOT / dc["observed_station_list"]).resolve()
        observed = json.loads(path.read_text(encoding="utf-8"))
        if not observed or len(observed) != len(set(observed)) or not set(observed) <= set(stations):
            raise ValueError("Invalid observed_station_list.")
        metadata["observed_station_ids"] = observed

    if "train_end" in dc:
        train_end = int(times.searchsorted(pd.Timestamp(dc["train_end"])))
        val_end = int(times.searchsorted(pd.Timestamp(dc["val_end"])))
    else:
        train_end = int(len(values) * 0.6)
        val_end = int(len(values) * 0.8)

    origins = split_origins(len(values), h, p, train_end, val_end, c.get("stride", p))
    if "train_stride" in dc:
        origins["train"] = split_origins(len(values), h, p, train_end, val_end, dc["train_stride"])["train"]
    if any(len(x) == 0 for x in origins.values()):
        raise ValueError("One or more dataset splits contain zero windows.")

    scaler = Scaler.fit(values[:train_end], dc.get("scaling", "neg1_pos1"))
    normalized = scaler.transform(values).astype(np.float32)
    metadata.update(
        rows=len(values),
        stations=stations,
        time_start=str(times[0]),
        time_end=str(times[-1]),
        train_end_exclusive=str(times[train_end]),
        val_end_exclusive=str(times[val_end]),
    )

    dyn_tables, fixed_tables, observation_tables = [], [], []
    static = None

    for name, block in dc.get("dynamic_blocks", {}).items():
        if not block.get("enabled", True):
            continue
        path = (ROOT / block["path"]).resolve()
        table = pd.read_csv(path)
        cutoff = times[train_end - 1]
        fit = table[(pd.to_datetime(table.valid_time) < times[train_end]) & (pd.to_datetime(table.available_at) <= cutoff)]
        fit = fit.sort_values("available_at").drop_duplicates("valid_time", keep="last")
        fs = Scaler.fit(fit[block["columns"]].to_numpy(dtype=np.float32))
        dyn_tables.append((table, block["columns"], fs))

    for name, block in dc.get("static_blocks", {}).items():
        if not block.get("enabled", True):
            continue
        path = (ROOT / block["path"]).resolve()
        table = pd.read_csv(path)
        if block.get("availability") == "fixed":
            s = align_fixed_static(table[["station_id"] + block["columns"]], stations)
        else:
            s = align_static(table[["station_id", "available_at"] + block["columns"]], stations, times[origins["train"][0] - 1])
        fs = Scaler.fit(s)
        s = fs.transform(s).astype(np.float32)
        static = s if static is None else np.concatenate([static, s], axis=1)

    for name, block in dc.get("fixed_periodic_blocks", {}).items():
        if not block.get("enabled", True):
            continue
        path = (ROOT / block["path"]).resolve()
        table = pd.read_csv(path)
        schedule_interval = block.get("interval_minutes", 30)
        values_fixed = periodic_features(table, times, block["columns"], block["anchor_minutes"], schedule_interval)
        fs = Scaler.fit(values_fixed[:train_end])
        fixed_tables.append(fs.transform(values_fixed).astype(np.float32))

    for name, block in dc.get("daily_observation_blocks", {}).items():
        if not block.get("enabled", True):
            continue
        path = (ROOT / block["path"]).resolve()
        table = pd.read_csv(path)
        delay = block["delay_hours"]
        available = pd.to_datetime(table.observed_day) + pd.Timedelta(days=1) + pd.Timedelta(hours=delay)
        fit = table.loc[available <= times[train_end - 1], block["columns"]].to_numpy(dtype=np.float32)
        fs = Scaler.fit(fit)
        observation_tables.append((table, block["columns"], delay, fs))

    def dynamic_for(os_):
        if not dyn_tables and not fixed_tables and not observation_tables:
            return None
        return np.stack(
            [
                np.concatenate(
                    [fs.transform(asof_features(table, times[o - h : o + p], times[o - 1], cols)) for table, cols, fs in dyn_tables]
                    + [v[o - h : o + p] for v in fixed_tables]
                    + [
                        fs.transform(delayed_daily_observations(table, times[o - h : o + p], times[o - 1], cols, delay))
                        for table, cols, delay, fs in observation_tables
                    ],
                    axis=-1,
                )
                for o in os_
            ]
        ).astype(np.float32)

    dynamic = {key: dynamic_for(os_) for key, os_ in origins.items()}
    metadata["normalization"] = {
        "kind": "train_minmax_" + scaler.mode,
        "observed_train_range": np.ptp(values[:train_end], axis=0).tolist(),
    }
    return times, stations, values, normalized, adj, origins, scaler, dynamic, static, metadata


def build(c, adj, dynamic, static):
    mc = copy.deepcopy(c["model"])
    name = mc.pop("name", "dbts")
    length = c["history"] + c["horizon"]
    kwargs = dict(nodes=len(adj), length=length, **mc)
    if name == "sequence_baseline":
        m = SequenceDiffusionBaseline(**kwargs)
    elif name == "dbts":
        m = DBTS(
            adj=torch.as_tensor(adj, dtype=torch.float32),
            dynamic_features=dynamic.shape[-1] if dynamic is not None else 0,
            static_features=static.shape[-1] if static is not None else 0,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown model name: {name}")
    return Diffusion(m, **c["diffusion"])


def sample_split(model, normalized, values, os_, scaler, dynamic, static, c, device):
    h, p = c["history"], c["horizon"]
    ec = c["evaluation"]
    draws = []
    batch_size = ec.get("batch_size", 4)
    start = time.perf_counter()
    for seed in ec["sample_seeds"]:
        batches = []
        for start_i in range(0, len(os_), batch_size):
            oo = os_[start_i : start_i + batch_size]
            hist = torch.tensor(np.stack([normalized[o - h : o] for o in oo]), device=device)
            dyn = torch.tensor(dynamic[start_i : start_i + len(oo)], device=device) if dynamic is not None else None
            full, _ = model.forecast(
                hist, p, dynamic=dyn, static=static, seed=seed + start_i * 100003, return_trace=True, **ec["sampler"]
            )
            batches.append(full[:, h:].cpu().numpy())
        draws.append(np.concatenate(batches))
    elapsed = time.perf_counter() - start
    samples = scaler.inverse(np.stack(draws))
    y = np.stack([values[o : o + p] for o in os_])
    return y, samples, {"seconds": elapsed, "seconds_per_origin_draw": elapsed / (len(os_) * len(draws))}


def run(config, out, resume=False):
    c = json.loads(Path(config).read_text(encoding="utf-8"))
    out = Path(out)
    evaluation_splits = c.get("evaluation", {}).get("splits", ["val", "test"])
    if out.exists() and not resume:
        raise ValueError(f"Output directory already exists: {out}")

    seed_all(c["seed"])
    device = torch.device(c.get("device", "cpu"))
    times, stations, values, norm, adj, origins, scaler, dynamic, static, metadata = prepare(c)
    model = build(c, adj, dynamic["train"], static).to(device)
    denoiser = getattr(model, "model", None)
    if getattr(denoiser, "gate_modules_by_external_features", False):
        metadata["module_activation"] = denoiser.module_activation()

    meta = {"config": c, "data": metadata}
    trainer = Trainer(model, **c["training"]["optimizer"])
    stat = torch.tensor(static, device=device) if static is not None else None
    out.mkdir(parents=True, exist_ok=True)

    if resume:
        trainer.load(out / "last.pt", meta)
    else:
        dump(out / "config.json", c)
        dump(out / "metadata.json", metadata)
        dump(
            out / "environment.json",
            {
                "python": sys.version,
                "torch": torch.__version__,
                "numpy": np.__version__,
                "platform": platform.platform(),
                "device": str(device),
            },
        )
        np.savez_compressed(out / "split.npz", **origins, timestamps=times.astype(str).to_numpy(dtype=str), stations=np.array(stations))

    h, p = c["history"], c["horizon"]
    tc = c["training"]
    start = time.perf_counter()
    with open(out / "training.jsonl", "a", encoding="utf-8") as log:
        while trainer.step < tc["steps"]:
            batches = []
            for micro in range(tc.get("gradient_accumulation", 1)):
                rng = np.random.default_rng(c["seed"] + trainer.step * 997 + micro * 10000019)
                window = (h + p) if tc.get("exclude_overlapping_dcl", False) and c["diffusion"]["lambda_dcl"] > 0 else None
                ids, _ = sample_indices(origins["train"], min(tc["batch_size"], len(origins["train"])), rng, window)
                oo = origins["train"][ids]
                x = torch.tensor(np.stack([norm[o - h : o + p] for o in oo]), device=device)
                dyn = torch.tensor(dynamic["train"][ids], device=device) if dynamic["train"] is not None else None
                batches.append(
                    (
                        x,
                        dict(
                            dynamic=dyn,
                            static=stat,
                            origins=torch.tensor(oo, device=device) if tc.get("exclude_overlapping_dcl", False) else None,
                        ),
                    )
                )
            terms = trainer.update_many(batches)
            log.write(json.dumps(terms) + "\n")
            log.flush()
            if trainer.step % tc.get("checkpoint_every", 500) == 0:
                trainer.save(out / "last.pt", meta)
        trainer.save(out / "last.pt", meta)

    training_seconds = time.perf_counter() - start
    train_range = np.ptp(values[: int(times.searchsorted(pd.Timestamp(metadata["train_end_exclusive"])))], axis=0)
    results = {}
    for split in evaluation_splits:
        count = c["evaluation"].get("max_origins")
        os_ = origins[split][:count] if count else origins[split]
        dyn = dynamic[split][: len(os_)] if dynamic[split] is not None else None
        y, samples, cost = sample_split(trainer.ema, norm, values, os_, scaler, dyn, stat, c, device)
        metrics = evaluate(y, samples, stations, train_range)
        if "observed_station_ids" in metadata:
            selected = metadata["observed_station_ids"]
            ids = [stations.index(n) for n in selected]
            dump(out / f"{split}_observed_metrics.json", evaluate(y[..., ids], samples[..., ids], selected, train_range[ids]))
        np.savez_compressed(
            out / f"{split}_predictions.npz",
            truth=y,
            samples=samples,
            point=samples.mean(0),
            origins=os_,
            stations=np.array(stations),
        )
        dump(out / f"{split}_metrics.json", metrics)
        dump(out / f"{split}_cost.json", cost)
        pd.DataFrame(metrics["station"]).T.to_csv(out / f"{split}_station_metrics.csv", encoding="utf-8-sig")
        results[split] = metrics["point"]

    dump(
        out / "run_summary.json",
        {
            "training_steps": trainer.step,
            "training_seconds": training_seconds,
            "parameter_count": sum(param.numel() for param in model.parameters()),
            "metrics": results,
        },
    )
    print(json.dumps({"out": str(out), "results": results}, ensure_ascii=True))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate DBTS.")
    parser.add_argument("--config", required=True, help="Path to JSON configuration file.")
    parser.add_argument("--out", required=True, help="Output directory for checkpoints and metrics.")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint.")
    args = parser.parse_args()
    run(args.config, args.out, args.resume)
