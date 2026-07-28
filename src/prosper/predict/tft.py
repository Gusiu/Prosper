"""Temporal Fusion Transformer predictions using pytorch-forecasting."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from prosper.config import Settings, get_settings
from prosper.domain import (
    DEFAULT_HORIZONS,
    DEPTH_BIN_LABELS,
    IDX_TO_DIR,
    assign_depth_bin,
    direction_from_return,
    parse_depth_bins,
)
from prosper.predict.window import (
    days_to_steps,
    training_bounds,
    untrained_horizon_payload,
    validate_train_window,
)
from prosper.storage.layout import get_features_parquet_path
from prosper.storage.predictions import write_versioned_predictions
from prosper.utils.time import parse_date


def _robust_normalize(X: np.ndarray, med: np.ndarray, iqr: np.ndarray) -> np.ndarray:
    X_safe = np.where(np.isfinite(X), X, 0.0)
    X_norm = (X_safe - med) / iqr
    return np.clip(np.nan_to_num(X_norm, nan=0.0, posinf=3.0, neginf=-3.0), -5.0, 5.0)


def predict_tft(
    symbol: str,
    start: str,
    end: str,
    settings: Settings | None = None,
    interval: str = "1d",
    seq_len: int = 60,
    train_window_days: int = 730,
    max_epochs: int = 10,
    hidden_size: int = 32,
    attention_head_size: int = 2,
    flat_threshold: float = 0.01,
    learning_rate: float = 1e-3,
) -> dict[str, Any]:
    """
    Train a Temporal Fusion Transformer every calendar month (rolling window)
    and produce per-bar JSONL predictions identical in format to ML/GRU models.
    """
    if settings is None:
        settings = get_settings()

    horizon_specs = list(DEFAULT_HORIZONS)
    steps_by_horizon = validate_train_window(train_window_days, interval, horizon_specs)
    train_window_steps = days_to_steps(train_window_days, interval)

    # ── Seed & deterministic mode (research) ─────────────────────────────────
    if settings.deterministic:
        seed = settings.seed if settings.seed is not None else 42
        np.random.seed(seed)
        import random

        random.seed(seed)
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)
        except ImportError:
            pass
        print(f"[research] Deterministic mode ON (seed={seed})")
    elif settings.seed is not None:
        np.random.seed(settings.seed)
        import random

        random.seed(settings.seed)
        print(f"[research] Seed set to {settings.seed} (non-deterministic CUDA)")

    # Suppress noisy upstream warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # ── 1. Load features ──────────────────────────────────────────────────────
    feat_path = get_features_parquet_path(symbol, interval, settings=settings)
    if not feat_path.exists():
        return {"error": f"No features parquet for {symbol}. Run: features build", "symbol": symbol}

    df_pl = pl.read_parquet(feat_path, hive_partitioning=False).sort("open_time")
    if df_pl.is_empty():
        return {"error": "Empty features DataFrame", "symbol": symbol}

    df_pl = df_pl.with_columns(pl.col("open_time").dt.date().alias("_date"))
    exclude_cols = {"open_time", "close_time", "_date", "symbol"}
    feature_cols = [c for c in df_pl.columns if c not in exclude_cols and c != "close"]

    open_times = df_pl["open_time"].to_list()
    dates = df_pl["_date"].to_list()
    closes = df_pl["close"].to_list()
    X_raw = df_pl.select(feature_cols).to_numpy().astype(np.float32)
    n = len(df_pl)

    start_dt = parse_date(start).date()
    end_dt = parse_date(end).date()

    # ── 2. Pre-compute forward labels ─────────────────────────────────────────
    dir_map = {"short": 0, "flat": 1, "long": 2}
    depth_bins, depth_labels = parse_depth_bins(",".join(DEPTH_BIN_LABELS))
    y_dir_all: dict[str, np.ndarray] = {}
    y_depth_all: dict[str, np.ndarray] = {}
    for h in horizon_specs:
        forward_steps = steps_by_horizon[h.name]
        dir_arr = np.full(n, -1, dtype=np.int64)
        depth_arr = np.full(n, -1, dtype=np.int64)
        for i in range(n - forward_steps):
            j = i + forward_steps
            if closes[i] and closes[j]:
                r = closes[j] / closes[i] - 1.0
                direction = direction_from_return(r, flat_threshold)
                dir_arr[i] = dir_map[direction]
                if direction in ("long", "short"):
                    bin_idx = assign_depth_bin(abs(r) * 100.0, depth_bins)
                    if bin_idx is not None:
                        depth_arr[i] = bin_idx
        y_dir_all[h.name] = dir_arr
        y_depth_all[h.name] = depth_arr

    # ── 3. Rolling-month train + inference ───────────────────────────────────
    try:
        import lightning as L
        from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
        from pytorch_forecasting.data.encoders import NaNLabelEncoder
        from pytorch_forecasting.metrics import CrossEntropy
    except ImportError:
        return {
            "error": "pytorch-forecasting not installed. Run: poetry add pytorch-forecasting lightning",
            "symbol": symbol,
        }

    predictions: list[dict[str, Any]] = []
    untrained_counts: dict[str, int] = {h.name: 0 for h in horizon_specs}
    current_train_month = None
    tft_model: dict[str, Any] = {}
    dataset_cache: dict[str, Any] = {}
    depth_cache: dict[str, dict[str, dict[str, float]]] = {}
    X_norm_all: np.ndarray | None = None

    for i in range(n):
        row_date = dates[i]
        if row_date < start_dt or row_date > end_dt:
            continue

        month_key = (row_date.year, row_date.month)

        # ── Train once per calendar month ─────────────────────────────────
        if month_key != current_train_month:
            current_train_month = month_key
            tft_model = {}
            dataset_cache = {}
            depth_cache = {}

            # Normalise on past rows only; stats never see future bars.
            X_tr = X_raw[max(0, i - train_window_steps) : i]
            X_safe = np.where(np.isfinite(X_tr), X_tr, 0.0)
            med = np.median(X_safe, axis=0)
            iqr_arr = np.percentile(X_safe, 75, axis=0) - np.percentile(X_safe, 25, axis=0)
            iqr_arr[iqr_arr == 0] = 1.0
            X_norm_all = _robust_normalize(X_raw, med, iqr_arr)

            for h in horizon_specs:
                h_name = h.name
                forward_steps = steps_by_horizon[h_name]
                y_dir = y_dir_all[h_name]
                train_start, train_end = training_bounds(i, train_window_steps, forward_steps)
                # Build pandas dataframe for TimeSeriesDataSet
                valid_idx = [k for k in range(train_start, train_end) if y_dir[k] >= 0]
                if len(valid_idx) < seq_len + 10 or len(set(y_dir[valid_idx])) < 2:
                    tft_model[h_name] = None
                    continue

                rows = []
                for k in valid_idx:
                    row = {"time_idx": k, "group": symbol, "target": IDX_TO_DIR[int(y_dir[k])]}
                    for fi, fc in enumerate(feature_cols):
                        row[fc] = float(X_norm_all[k, fi])
                    rows.append(row)

                df_pd = pd.DataFrame(rows)

                max_enc = min(seq_len, len(df_pd) - seq_len)
                if max_enc < 5:
                    tft_model[h_name] = None
                    continue

                try:
                    ds = TimeSeriesDataSet(
                        df_pd,
                        time_idx="time_idx",
                        target="target",
                        group_ids=["group"],
                        min_encoder_length=max_enc // 2,
                        max_encoder_length=max_enc,
                        min_prediction_length=1,
                        max_prediction_length=1,
                        time_varying_unknown_reals=feature_cols,
                        target_normalizer=NaNLabelEncoder(add_nan=True),
                        add_relative_time_idx=True,
                        add_target_scales=False,
                        add_encoder_length=True,
                    )

                    loader = ds.to_dataloader(train=True, batch_size=32, num_workers=0)

                    tft = TemporalFusionTransformer.from_dataset(
                        ds,
                        learning_rate=learning_rate,
                        hidden_size=hidden_size,
                        attention_head_size=attention_head_size,
                        dropout=0.1,
                        hidden_continuous_size=16,
                        loss=CrossEntropy(),
                        log_interval=-1,
                        reduce_on_plateau_patience=2,
                    )

                    # The model is used immediately and discarded, so neither a
                    # logger nor a checkpoint is wanted. Left on, Lightning
                    # writes one lightning_logs/version_N/ and one .ckpt per
                    # Trainer — a full run once left ~18k directories and
                    # 260 MB of dead artifacts in the repo root.
                    trainer_kwargs: dict[str, Any] = dict(
                        max_epochs=max_epochs,
                        enable_progress_bar=False,
                        enable_model_summary=False,
                        enable_checkpointing=False,
                        logger=False,
                        accelerator="cpu",
                        default_root_dir=str(settings.meta_dir / "lightning"),
                    )
                    if settings.deterministic:
                        trainer_kwargs["deterministic"] = True
                    trainer = L.Trainer(**trainer_kwargs)
                    trainer.fit(tft, train_dataloaders=loader)
                    tft_model[h_name] = tft
                    dataset_cache[h_name] = ds
                    depth_cache[h_name] = _empirical_depth(
                        y_dir_all[h_name][train_start:train_end],
                        y_depth_all[h_name][train_start:train_end],
                        depth_labels,
                    )
                except Exception as e:
                    print(f"Training exception for month {current_train_month}: {e}")
                    tft_model[h_name] = None

        # ── Inference ─────────────────────────────────────────────────────
        date_str = row_date.isoformat()
        out: dict[str, Any] = {
            "open_time": open_times[i].isoformat(),
            "date": date_str,
            "symbol": symbol,
        }

        for h in horizon_specs:
            h_name = h.name
            forward_steps = steps_by_horizon[h_name]
            tft_h = tft_model.get(h_name)
            if not tft_h or X_norm_all is None or i < seq_len:
                out[h_name] = untrained_horizon_payload(depth_labels)
                untrained_counts[h_name] += 1
            else:
                try:
                    ds_ref = dataset_cache.get(h_name)
                    if not ds_ref:
                        raise ValueError("No ds_ref")
                    enc_len = min(seq_len, i)
                    rows_inf = []
                    y_dir = y_dir_all[h_name]
                    for k in range(i - enc_len, i + 1):
                        # The encoder consumes target history. Only labels
                        # realised before bar i are known then; anything newer
                        # must be masked or the model reads its own answer.
                        known = k + forward_steps < i and y_dir[k] >= 0
                        row = {
                            "time_idx": k,
                            "group": symbol,
                            "target": IDX_TO_DIR[int(y_dir[k])] if known else np.nan,
                        }
                        for fi, fc in enumerate(feature_cols):
                            row[fc] = float(X_norm_all[k, fi])
                        rows_inf.append(row)

                    df_inf = pd.DataFrame(rows_inf)
                    ds_inf = TimeSeriesDataSet.from_dataset(
                        ds_ref, df_inf, predict=True, stop_randomization=True
                    )
                    inf_loader = ds_inf.to_dataloader(train=False, batch_size=1, num_workers=0)

                    raw_preds = tft_h.predict(inf_loader, mode="raw", return_x=False)
                    import torch

                    # Handle various output formats from pytorch-forecasting
                    if isinstance(raw_preds, dict):
                        logits = raw_preds["prediction"][0, 0].cpu().numpy()
                    elif isinstance(raw_preds, tuple):
                        if isinstance(raw_preds[0], dict):
                            logits = raw_preds[0]["prediction"][0, 0].cpu().numpy()
                        else:
                            logits = raw_preds[0][0, 0].cpu().numpy() if isinstance(raw_preds[0], torch.Tensor) else raw_preds[0].prediction[0, 0].cpu().numpy()
                    elif hasattr(raw_preds, "output"):
                        logits = raw_preds.output.prediction[0, 0].cpu().numpy()
                    elif hasattr(raw_preds, "prediction"):
                        logits = raw_preds.prediction[0, 0].cpu().numpy()
                    elif isinstance(raw_preds, torch.Tensor):
                        # Handle plain tensor - ensure correct shape
                        if raw_preds.ndim == 3:
                            logits = raw_preds[0, 0].cpu().numpy()
                        elif raw_preds.ndim == 2:
                            logits = raw_preds[0].cpu().numpy()
                        else:
                            logits = raw_preds.cpu().numpy()
                    else:
                        # Last resort fallback
                        logits = np.array(raw_preds)
                        if logits.ndim == 3:
                            logits = logits[0, 0]
                        elif logits.ndim == 2:
                            logits = logits[0]

                    # Ensure logits is 1D array of class probabilities
                    if logits.ndim > 1:
                        logits = logits.flatten()

                    # NaNLabelEncoder(add_nan=True) prepends an "unknown"
                    # class; drop it so the 3 direction classes line up.
                    if logits.shape[0] == len(IDX_TO_DIR) + 1:
                        logits = logits[1:]

                    shifted = logits - np.max(logits)
                    probs = np.exp(shifted) / np.exp(shifted).sum()
                    p_short, p_flat, p_long = float(probs[0]), float(probs[1]), float(probs[2])
                except Exception as e:
                    print(f"Inference exception at {date_str}: {e}")
                    out[h_name] = untrained_horizon_payload(depth_labels)
                    untrained_counts[h_name] += 1
                    continue

                depths = depth_cache.get(h_name, {})
                out[h_name] = {
                    "P_long": p_long,
                    "P_flat": p_flat,
                    "P_short": p_short,
                    "depth_long_bins": depths.get("long", _uniform_depth(depth_labels)),
                    "depth_short_bins": depths.get("short", _uniform_depth(depth_labels)),
                    "trained": True,
                }

        predictions.append(out)

    if not predictions:
        return {"error": "No predictions generated for the requested date range", "symbol": symbol}


    run_dir = write_versioned_predictions(
        predictions, symbol, "tft", settings, interval=interval
    )

    return {
        "symbol": symbol,
        "start": start,
        "end": end,
        "interval": interval,
        "predictions": len(predictions),
        "untrained_horizons": untrained_counts,
        "run_dir": str(run_dir),
    }


def _uniform_depth(labels: list[str]) -> dict[str, float]:
    return {label: 1.0 / len(labels) for label in labels}


def _empirical_depth(
    directions: np.ndarray,
    depths: np.ndarray,
    labels: list[str],
) -> dict[str, dict[str, float]]:
    """Realised depth-bin distribution per direction over a training window."""
    counts = {"long": [0] * len(labels), "short": [0] * len(labels)}
    for direction_idx, depth_idx in zip(directions, depths, strict=True):
        if depth_idx < 0 or direction_idx < 0:
            continue
        direction = IDX_TO_DIR.get(int(direction_idx))
        if direction in counts:
            counts[direction][int(depth_idx)] += 1

    out: dict[str, dict[str, float]] = {}
    for direction, bucket in counts.items():
        total = sum(bucket)
        out[direction] = (
            _uniform_depth(labels)
            if total == 0
            else {label: c / total for label, c in zip(labels, bucket, strict=True)}
        )
    return out
