import argparse
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

from engine.solver3_1 import Trainer
from Utils.io_utils import instantiate_from_config, load_yaml_config, seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Run the reviewer-ready forecasting pipeline.")
    parser.add_argument("--config", default="Config/reviewer_forecast.yaml")
    parser.add_argument("--series-csv", required=True, help="Private multivariate time-series CSV.")
    parser.add_argument("--adj-csv", required=True, help="Private adjacency matrix CSV.")
    parser.add_argument("--external-csv", default=None, help="Optional private exogenous feature CSV.")
    parser.add_argument("--adj-new-csv", default=None, help="Optional private auxiliary node matrix CSV.")
    parser.add_argument("--output-dir", default="review_outputs/reviewer_run")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--pred-len", type=int, default=None)
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint file. If provided, skip training and only run inference.")
    return parser.parse_args()


def load_numeric_table(path, index_col=None):
    df = pd.read_csv(path, index_col=index_col)
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] == 0:
        raise ValueError(f"No numeric columns found in {path}.")
    return numeric_df.to_numpy(dtype=np.float32)


def load_adjacency(path):
    try:
        matrix = load_numeric_table(path, index_col=0)
        if matrix.shape[0] == matrix.shape[1]:
            return matrix
    except Exception:
        pass
    matrix = load_numeric_table(path, index_col=None)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Adjacency matrix must be square, got {matrix.shape} from {path}.")
    return matrix


def reshape_to_sequences(matrix, seq_length):
    usable_steps = (len(matrix) // seq_length) * seq_length
    if usable_steps == 0:
        raise ValueError(f"Not enough rows to form one sequence of length {seq_length}.")
    trimmed = matrix[:usable_steps]
    return trimmed.reshape(-1, seq_length, matrix.shape[1])


def weighted_mape(y_true, y_pred):
    mask = y_true != 0
    if not np.any(mask):
        return float("nan")
    return np.sum(np.abs(y_true[mask] - y_pred[mask])) / np.sum(np.abs(y_true[mask]))


class TrainDataset(Dataset):
    def __init__(self, series, external_features, adj, adj_new):
        self.series = series
        self.external_features = external_features
        self.adj = adj
        self.adj_new = adj_new

    def __len__(self):
        return len(self.series)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.series[idx]).float(),
            torch.from_numpy(self.external_features[idx]).float(),
            torch.from_numpy(self.adj).float(),
            torch.from_numpy(self.adj_new).float(),
        )


class EvalDataset(Dataset):
    def __init__(self, series, external_features, adj, adj_new, pred_len):
        self.series = series
        self.external_features = external_features
        self.adj = adj
        self.adj_new = adj_new
        self.mask = np.ones_like(series, dtype=bool)
        self.mask[:, -pred_len:, :] = False

    def __len__(self):
        return len(self.series)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.series[idx]).float(),
            torch.from_numpy(self.mask[idx]).bool(),
            torch.from_numpy(self.external_features[idx]).float(),
            torch.from_numpy(self.adj).float(),
            torch.from_numpy(self.adj_new).float(),
        )


def main():
    args = parse_args()
    config = load_yaml_config(args.config)

    data_cfg = config["data"]
    seq_length = data_cfg["seq_length"]
    pred_len = args.pred_len or data_cfg["pred_len"]
    feature_size = data_cfg["feature_size"]
    external_dim = data_cfg["external_dim"]

    if args.seed is None:
        args.seed = data_cfg["seed"]
    seed_everything(args.seed)

    if args.max_epochs is not None:
        config["solver"]["max_epochs"] = args.max_epochs
    if args.batch_size is not None:
        config["data"]["batch_size"] = args.batch_size
    if args.eval_batch_size is not None:
        config["data"]["eval_batch_size"] = args.eval_batch_size

    os.makedirs(args.output_dir, exist_ok=True)
    config["solver"]["results_folder"] = os.path.join(args.output_dir, "checkpoints")

    series_raw = load_numeric_table(args.series_csv)
    adj = load_adjacency(args.adj_csv)

    if series_raw.shape[1] != feature_size:
        raise ValueError(
            f"Expected {feature_size} series features, got {series_raw.shape[1]}. "
            "The current model code is dimension-specific."
        )
    if adj.shape != (feature_size, feature_size):
        raise ValueError(
            f"Expected adjacency shape {(feature_size, feature_size)}, got {adj.shape}."
        )

    if args.external_csv is not None:
        external_raw = load_numeric_table(args.external_csv)
        if external_raw.shape[1] != external_dim:
            raise ValueError(
                f"Expected {external_dim} external features, got {external_raw.shape[1]}. "
                "The current model code is dimension-specific."
            )
        if len(external_raw) < len(series_raw):
            raise ValueError("External feature rows must be at least as many as the series rows.")
        external_raw = external_raw[: len(series_raw)]
    else:
        external_raw = np.zeros((len(series_raw), external_dim), dtype=np.float32)

    if args.adj_new_csv is not None:
        adj_new = load_numeric_table(args.adj_new_csv)
        if adj_new.shape[0] != feature_size:
            raise ValueError(
                f"Expected adj_new to have {feature_size} rows, got {adj_new.shape[0]}."
            )
    else:
        adj_new = np.zeros((feature_size, 1), dtype=np.float32)

    series_scaler = MinMaxScaler()
    series_scaled = series_scaler.fit_transform(series_raw)

    if args.external_csv is not None:
        external_scaler = MinMaxScaler()
        external_scaled = external_scaler.fit_transform(external_raw)
    else:
        external_scaled = external_raw

    series_sequences = reshape_to_sequences(series_scaled, seq_length)
    external_sequences = reshape_to_sequences(external_scaled, seq_length)

    train_dataset = TrainDataset(series_sequences, external_sequences, adj, adj_new)
    eval_dataset = EvalDataset(series_sequences, external_sequences, adj, adj_new, pred_len)

    train_batch_size = min(config["data"]["batch_size"], len(train_dataset))
    eval_batch_size = min(config["data"]["eval_batch_size"], len(eval_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
        pin_memory=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=True,
    )

    if args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = instantiate_from_config(config["model"]).to(device)

    trainer_args = SimpleNamespace(
        name="reviewer_forecast",
        save_dir=args.output_dir,
        mode="predict",
        pred_len=pred_len,
        gpu=args.gpu,
    )
    trainer = Trainer(
        config=config,
        args=trainer_args,
        model=model,
        dataloader={"dataloader": train_loader},
    )

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        data = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(data['model'])
        trainer.ema.load_state_dict(data['ema'])
        trainer.step = data.get('step', 0)
    else:
        trainer.train()

    sampling_cfg = config["sampling"]
    predictions, reals, masks = trainer.restore(
        eval_loader,
        shape=[seq_length, feature_size],
        coef=sampling_cfg["coefficient"],
        stepsize=sampling_cfg["step_size"],
        sampling_steps=sampling_cfg["sampling_steps"],
    )

    predictions = series_scaler.inverse_transform(
        predictions.reshape(-1, feature_size)
    ).reshape(predictions.shape)
    reals = series_scaler.inverse_transform(
        reals.reshape(-1, feature_size)
    ).reshape(reals.shape)

    forecast_mask = ~masks
    y_true = reals[forecast_mask]
    y_pred = predictions[forecast_mask]

    metrics = pd.DataFrame(
        {
            "RMSE": [np.sqrt(mean_squared_error(y_true, y_pred))],
            "MAE": [mean_absolute_error(y_true, y_pred)],
            "R2": [r2_score(y_true, y_pred)],
            "WMAPE": [weighted_mape(y_true, y_pred)],
        }
    )
    metrics.to_csv(os.path.join(args.output_dir, "metrics.csv"), index=False)

    np.save(os.path.join(args.output_dir, "predictions.npy"), predictions)
    np.save(os.path.join(args.output_dir, "ground_truth.npy"), reals)
    np.save(os.path.join(args.output_dir, "forecast_mask.npy"), forecast_mask)

    print(f"Run complete. Results saved to: {args.output_dir}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
