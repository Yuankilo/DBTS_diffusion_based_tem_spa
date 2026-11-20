# DBTS: A Robust Diffusion-Based Generative Framework for Spatiotemporal Metro Passenger Flow Forecasting

Official PyTorch implementation for *A Robust Diffusion-Based Generative Framework for Spatiotemporal Metro Passenger Flow Forecasting*.

## Overview

DBTS is a conditional denoising diffusion framework designed for network-wide urban rail transit passenger flow forecasting under both regular operations and non-recurring disruptions. Its main components include:

- **Temporal-Spatial Dual-Branch Denoiser (`dbts/model.py`)**: Combines a Transformer-based Temporal Dependency Block (TDB) with a symmetric degree-normalized Graph Convolutional Network (GCN) Spatial Dependency Block (SDB) using parallel additive fusion (`decoder_fusion="paper_add"`).
- **Reconstruction Block (`dbts/model.py`, `dbts/layers/transformer.py`)**: Decomposes denoised station ridership trajectories into trend, periodic (Fourier), and residual components.
- **Dynamic Contrastive Learning (`dbts/model.py`, `dbts/diffusion.py`)**: Training-only representation regularization over paired original and randomly masked flow views through the shared encoder and projection head.
- **Dual-Term Conditional Reverse-Diffusion Guidance (`dbts/scenario.py`, `dbts/diffusion.py`)**: Incorporates reference ridership consistency and condition log-likelihood gradients into reverse diffusion sampling.
- **Time-Frequency Joint Objective (`dbts/diffusion.py`)**: Combines timestep-weighted time-domain and FFT frequency-domain reconstruction losses with the contrastive loss.

## Repository Structure

```text
DBTS_diffusion_based_tem_spa/
├── README.md
├── requirements.txt
├── configs/
│   ├── paper_guidance_s2026.json    # Full DBTS model with dual-term scenario guidance
│   ├── E1_full_inputs.json          # Full multi-source exogenous input configuration
│   ├── E2_without_sdb.json          # Module ablation: disable SDB (GCN)
│   └── E2_without_weather.json      # Feature ablation: remove weather features
└── dbts/
    ├── __init__.py
    ├── model.py                     # Symmetric normalized GCN, TDB/SDB/RB denoiser, DCL, ablation switches
    ├── diffusion.py                 # Forward/reverse diffusion, time-frequency + DCL loss, scenario guidance
    ├── scenario.py                  # Reference ridership head and condition log-likelihood (ScenarioGuidance)
    ├── layers/
    │   ├── __init__.py
    │   ├── transformer.py           # Transformer encoder/decoder, TrendBlock, FourierLayer
    │   ├── model_utils.py           # AdaLayerNorm, positional encoding, Conv_MLP
    │   └── lr_sch.py                # Warmup learning-rate schedulers
    ├── data.py                      # Ridership/graph loading, exogenous feature alignment, scaling, splitting
    ├── batching.py                  # Training batch sampler with non-overlapping DCL window support
    ├── training.py                  # Adam optimization, EMA updates, checkpoint save/load
    ├── runner.py                    # End-to-end training and evaluation CLI
    ├── metrics.py                   # Point (RMSE, MAE, R2) and probabilistic (CRPS) evaluation metrics
    ├── cohorts.py                   # Temporal regime slicing (peak/off-peak, weekday/weekend)
    ├── point_models.py              # LSTM and GRU point forecasting baselines
    ├── transformer_baseline.py      # Transformer point forecasting baseline
    └── xgb_baseline.py              # XGBoost station-level baseline
```

## Installation

Python 3.10+ and PyTorch 2.1+ are recommended:

```bash
pip install -r requirements.txt
```

## Data Preparation

Place preprocessed station ridership, undirected physical adjacency matrix, and exogenous feature tables under `data/` (paths are configurable in `configs/*.json`):

- `data/flow.csv`: Timestamp column followed by station ridership columns at fixed 30-minute intervals.
- `data/adjacency.csv`: Square symmetric nonnegative station adjacency matrix indexed by station IDs.
- `data/static.csv` (or `poi_snapshot.csv`, `bus_snapshot.csv`): Station-level POI and bus connectivity attributes.
- `data/timetable_period.csv` (or `timetable_vintages.csv`): Periodic train service schedule features.
- `data/weather_daily.csv` (or `weather_vintages.csv`): Daily or vintage meteorological observations.

## Training and Evaluation

Run training and evaluation with any JSON configuration file:

```bash
python -m dbts.runner --config configs/paper_guidance_s2026.json --out runs/dbts_main
```

To resume from the latest saved checkpoint:

```bash
python -m dbts.runner --config configs/paper_guidance_s2026.json --out runs/dbts_main --resume
```

Outputs saved under `--out` include `config.json`, `metadata.json`, `training.jsonl`, `last.pt`, `{val,test}_metrics.json`, `{val,test}_station_metrics.csv`, `{val,test}_predictions.npz`, and `run_summary.json`.

## Module-to-Paper Correspondence

| Component | Paper Reference | Implementation |
| --- | --- | --- |
| Symmetric normalized GCN | Section III-E, Eq. (9) | `dbts/model.py` (`normalize_graph`, `GCN`) |
| Parallel TDB + SDB fusion & RB decomposition | Section III-E, Eqs. (10), (19), Fig. 5 | `dbts/model.py` (`DBTS.forward`) |
| Dynamic Contrastive Learning (DCL) | Section III-F, Eqs. (12)–(13), Fig. 6 | `dbts/model.py` (`contrastive`, `DBTS.dcl_loss`), `dbts/diffusion.py` (`Diffusion.loss`) |
| Dual-term conditional reverse guidance | Section III-C, Eq. (6) | `dbts/scenario.py` (`ScenarioGuidance`), `dbts/diffusion.py` (`Diffusion.guide_scenario`, `Diffusion.forecast`) |
| Time-frequency joint objective | Section III-G, Eqs. (24)–(26) | `dbts/diffusion.py` (`Diffusion.__init__`, `Diffusion.loss`) |
| Linked / independent ablation control | Section V-C, Tables VI–VII | `dbts/model.py` (`DBTS.module_activation`) |
