"""Data loading, station-graph alignment, feature extraction, scaling, and window splitting."""

from dataclasses import dataclass
import csv
import numpy as np
import pandas as pd
import torch


def read_flow(path, trailing_partial="error", interval_minutes=30):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    if len(rows[0]) != len(set(rows[0])):
        raise ValueError("Duplicate station column labels.")
    removed = []
    if len(rows[-1]) != len(rows[0]) and trailing_partial == "drop_documented":
        removed = [len(rows) - 1]
        rows = rows[:-1]
    if any(len(r) != len(rows[0]) for r in rows):
        raise ValueError("Malformed CSV row length.")
    df = pd.DataFrame(rows[1:], columns=rows[0])
    times = pd.DatetimeIndex(pd.to_datetime(df.iloc[:, 0], errors="raise"))
    validate_times(times, interval_minutes)
    values = df.iloc[:, 1:].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Passenger flow must be finite and nonnegative.")
    return times, df.columns[1:].tolist(), values, {"dropped_partial_csv_rows": removed}


def validate_times(times, interval_minutes=30):
    if isinstance(interval_minutes, bool) or interval_minutes not in (30, 60):
        raise ValueError("Supported interval_minutes are 30 and 60.")
    if times.hasnans or times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError("Timestamps must be unique and strictly increasing.")
    if len(times) > 1 and not np.all(np.diff(times.asi8) == interval_minutes * 60 * 10**9):
        raise ValueError("Missing or irregular timestamp spacing.")


def align_graph(path, stations):
    df = pd.read_csv(path, index_col=0)
    if df.index.has_duplicates or df.columns.has_duplicates or len(set(stations)) != len(stations):
        raise ValueError("Duplicate station identifiers.")
    if set(df.index) != set(stations) or set(df.columns) != set(stations):
        raise ValueError("Graph and ridership station sets do not match.")
    a = df.loc[stations, stations].to_numpy(dtype=np.float32)
    if not np.isfinite(a).all() or (a < 0).any() or not np.allclose(a, a.T):
        raise ValueError("Expected finite nonnegative undirected adjacency matrix.")
    return a


def align_static(df, stations, origin):
    if df.station_id.duplicated().any() or set(df.station_id) != set(stations):
        raise ValueError("Static feature station keys differ from ridership stations.")
    available = pd.to_datetime(df.available_at, errors="raise")
    if available.isna().any() or (available > pd.Timestamp(origin)).any():
        raise ValueError("Static features unavailable at forecast origin.")
    out = df.set_index("station_id").loc[stations].drop(columns="available_at").to_numpy(dtype=np.float32)
    if not np.isfinite(out).all():
        raise ValueError("Static features contain nonfinite values.")
    return out


def align_fixed_static(df, stations):
    if df.station_id.duplicated().any() or set(df.station_id) != set(stations):
        raise ValueError("Static feature station keys differ from ridership stations.")
    out = df.set_index("station_id").loc[stations].to_numpy(dtype=np.float32)
    if not np.isfinite(out).all():
        raise ValueError("Static features contain nonfinite values.")
    return out


def periodic_features(table, times, columns, anchor_minutes, interval_minutes=30):
    if isinstance(interval_minutes, bool) or interval_minutes not in (30, 60):
        raise ValueError("Supported schedule intervals are 30 and 60 minutes.")
    slots_per_day = 1440 // interval_minutes
    if sorted(table.slot.tolist()) != list(range(slots_per_day)):
        raise ValueError("Timetable slot count does not match declared interval.")
    if anchor_minutes not in range(0, 1440, interval_minutes):
        raise ValueError("Invalid timetable anchor_minutes.")
    times = pd.DatetimeIndex(times)
    if times.hasnans or not (times == times.floor(f"{interval_minutes}min")).all():
        raise ValueError("Off-grid timestamp requested for periodic timetable.")
    values = table.set_index("slot").loc[range(slots_per_day), columns].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Timetable contains nonfinite values.")
    slots = ((times.hour * 60 + times.minute - anchor_minutes) % 1440) // interval_minutes
    return values[slots]


def delayed_daily_observations(table, times, origin, columns, delay_hours):
    if not np.isfinite(delay_hours) or delay_hours < 0:
        raise ValueError("delay_hours must be finite and nonnegative.")
    days = pd.DatetimeIndex(pd.to_datetime(table.observed_day, errors="raise"))
    if days.hasnans or days.has_duplicates or not days.is_monotonic_increasing or not (days == days.normalize()).all():
        raise ValueError("Observation dates must be unique ordered dates.")
    if len(days) > 1 and not np.all(np.diff(days.asi8) == pd.Timedelta(days=1).value):
        raise ValueError("Missing daily observation rows.")
    times = pd.DatetimeIndex(times)
    origin = pd.Timestamp(origin)
    if times.hasnans or pd.isna(origin):
        raise ValueError("Invalid timestamp requested.")
    availability = days + pd.Timedelta(days=1) + pd.Timedelta(hours=delay_hours)
    cutoff = np.minimum(times.asi8, origin.value)
    indices = np.searchsorted(availability.asi8, cutoff, side="right") - 1
    if (indices < 0).any():
        raise ValueError("No completed daily observation available before forecast origin.")
    result = table[columns].to_numpy(dtype=np.float32)[indices]
    if not np.isfinite(result).all():
        raise ValueError("Daily observations contain nonfinite values.")
    return result


def asof_features(df, requested_times, origin, feature_columns):
    """Select dynamic external features available up to the forecast origin."""
    df = df.copy()
    for col in ("valid_time", "available_at"):
        df[col] = pd.to_datetime(df[col], errors="raise")
        if df[col].isna().any():
            raise ValueError("Feature timestamps cannot be missing.")
    if df.duplicated(["valid_time", "available_at"]).any():
        raise ValueError("Duplicate (valid_time, available_at) rows.")
    df = df[df.available_at <= pd.Timestamp(origin)].sort_values("available_at")
    df = df.drop_duplicates("valid_time", keep="last").set_index("valid_time")
    out = df.reindex(pd.DatetimeIndex(requested_times))[feature_columns].to_numpy(dtype=np.float32)
    if not np.isfinite(out).all():
        raise ValueError("Missing external feature values for requested timestamps.")
    return out


@dataclass
class Scaler:
    minimum: np.ndarray
    span: np.ndarray
    mode: str = "neg1_pos1"

    @classmethod
    def fit(cls, train, mode="neg1_pos1"):
        if len(train) == 0 or not np.isfinite(train).all():
            raise ValueError("Invalid training array for scaling.")
        if mode not in ("neg1_pos1", "zero_one"):
            raise ValueError("mode must be 'neg1_pos1' or 'zero_one'.")
        return cls(train.min(axis=0), np.maximum(train.max(axis=0) - train.min(axis=0), 1.0), mode)

    def transform(self, x):
        y = (x - self.minimum) / self.span
        return y if self.mode == "zero_one" else y * 2.0 - 1.0

    def inverse(self, x):
        y = x if self.mode == "zero_one" else (x + 1.0) / 2.0
        return y * self.span + self.minimum


def split_origins(length, history, horizon, train_end, val_end, stride=1):
    if not 0 < train_end < val_end < length or min(history, horizon, stride) < 1:
        raise ValueError("Invalid split boundaries.")
    return {
        name: np.arange(max(history, start), end - horizon + 1, stride, dtype=int)
        for name, start, end in (("train", 0, train_end), ("val", train_end, val_end), ("test", val_end, length))
    }


class Windows(torch.utils.data.Dataset):
    def __init__(self, values, origins, history, horizon):
        self.values = torch.as_tensor(values, dtype=torch.float32)
        self.origins = np.asarray(origins)
        self.history = history
        self.horizon = horizon
        if len(self.origins) == 0:
            raise ValueError("Empty window split.")
        if self.origins.min() < history or self.origins.max() + horizon > len(values):
            raise ValueError("Window origins out of bounds.")

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, index):
        o = int(self.origins[index])
        x = self.values[o - self.history : o + self.horizon]
        return {"history": x[: self.history].clone(), "target": x[self.history :].clone(), "joint": x.clone(), "origin": o}
