"""Temporal Fusion Transformer predictions using pytorch-forecasting."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from prosper.config import Settings, get_settings
from prosper.labels.depth import (
    DEPTH_BIN_LABELS,
    N_DEPTH_BINS,
    direction_from_return,
)
from prosper.storage.layout import get_features_parquet_path
from prosper.storage.predictions import write_predictions_jsonl, write_versioned_predictions
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
    train_window_days: int = 365,
    max_epochs: int = 10,
    hidden_size: int = 32,
    attention_head_size: int = 2,
    flat_threshold: float = 0.01,
    short_forward_days: int = 28,
    medium_forward_days: int = 182,
    long_forward_days: int = 365,
    learning_rate: float = 1e-3,
) -> dict[str, Any]:
    """
    Train a Temporal Fusion Transformer every calendar month (rolling window)
    and produce daily JSONL predictions identical in format to ML/GRU models.
    """
    if settings is None:
        settings = get_settings()

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

    dates = df_pl["_date"].to_list()
    closes = df_pl["close"].to_list()
    X_raw = df_pl.select(feature_cols).to_numpy().astype(np.float32)
    n = len(df_pl)

    start_dt = parse_date(start).date()
    end_dt = parse_date(end).date()

    # ── 2. Pre-compute forward labels ─────────────────────────────────────────
    dir_map = {"short": 0, "flat": 1, "long": 2}
    horizons = {
        "short": short_forward_days,
        "medium": medium_forward_days,
        "long": long_forward_days,
    }
    y_dir_all = {}
    for h_name, f_days in horizons.items():
        arr = np.full(n, -1, dtype=np.int64)
        for i in range(n - f_days):
            j = i + f_days
            if closes[i] and closes[j]:
                r = closes[j] / closes[i] - 1.0
                arr[i] = dir_map[direction_from_return(r, flat_threshold)]
        y_dir_all[h_name] = arr

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
    current_train_month = None
    tft_model = {}
    norm_params: tuple[np.ndarray, np.ndarray] | None = None  # (med, iqr)

    uniform_depth = {lbl: 1.0 / N_DEPTH_BINS for lbl in DEPTH_BIN_LABELS}

    for i in range(n):
        row_date = dates[i]
        if row_date < start_dt or row_date > end_dt:
            continue

        train_end = i - 1
        train_start = max(0, i - train_window_days)
        month_key = (row_date.year, row_date.month)

        # ── Train once per calendar month ─────────────────────────────────
        if month_key != current_train_month and (train_end - train_start) >= seq_len + 20:
            current_train_month = month_key
            tft_model = {}

            # Normalise
            X_tr = X_raw[train_start : train_end + 1]
            X_safe = np.where(np.isfinite(X_tr), X_tr, 0.0)
            med = np.median(X_safe, axis=0)
            iqr_arr = np.percentile(X_safe, 75, axis=0) - np.percentile(X_safe, 25, axis=0)
            iqr_arr[iqr_arr == 0] = 1.0
            norm_params = (med, iqr_arr)
            X_norm_all = _robust_normalize(X_raw, med, iqr_arr)

            norm_params_cache = {}

            for h_name, f_days in horizons.items():
                y_dir = y_dir_all[h_name]
                # Prevent look-ahead leakage: label at k needs close[k + f_days]
                safe_train_end = train_end - f_days
                if safe_train_end <= train_start:
                    tft_model[h_name] = None
                    continue
                # Build pandas dataframe for TimeSeriesDataSet
                valid_idx = [k for k in range(train_start, safe_train_end + 1) if y_dir[k] >= 0]
                if len(valid_idx) < seq_len + 10 or len(set(y_dir[valid_idx])) < 2:
                    tft_model[h_name] = None
                    continue

                dir_map_rev = {0: "short", 1: "flat", 2: "long"}
                rows = []
                for k in valid_idx:
                    row = {"time_idx": k, "group": symbol, "target": dir_map_rev[y_dir[k]]}
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

                    trainer_kwargs = dict(
                        max_epochs=max_epochs,
                        enable_progress_bar=False,
                        enable_model_summary=False,
                        logger=False,
                        accelerator="cpu",
                    )
                    if settings.deterministic:
                        trainer_kwargs["deterministic"] = True
                    trainer = L.Trainer(**trainer_kwargs)
                    trainer.fit(tft, train_dataloaders=loader)
                    tft_model[h_name] = tft
                    norm_params_cache[h_name] = ds
                except Exception as e:
                    print(f"Training exception for month {current_train_month}: {e}")
                    tft_model[h_name] = None
            norm_params = (med, iqr_arr, X_norm_all, norm_params_cache)

        # ── Inference ─────────────────────────────────────────────────────
        date_str = row_date.isoformat()
        out: dict[str, Any] = {"date": date_str, "symbol": symbol}

        for h_name in horizons.keys():
            tft_h = tft_model.get(h_name)
            if not tft_h or norm_params is None or i < seq_len:
                out[h_name] = {
                    "P_long": 0.33,
                    "P_flat": 0.34,
                    "P_short": 0.33,
                    "depth_long_bins": uniform_depth.copy(),
                    "depth_short_bins": uniform_depth.copy(),
                }
            else:
                try:
                    med, iqr_arr, X_norm_all, norm_params_cache = norm_params
                    ds_ref = norm_params_cache.get(h_name)
                    if not ds_ref:
                        raise ValueError("No ds_ref")
                    enc_len = min(seq_len, i)
                    rows_inf = []
                    dir_map_rev = {0: "short", 1: "flat", 2: "long"}
                    y_dir = y_dir_all[h_name]
                    for k in range(i - enc_len, i + 1):
                        fake_target = dir_map_rev[max(y_dir[k], 0)]
                        row = {"time_idx": k, "group": symbol, "target": fake_target}
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

                    probs = np.exp(logits) / np.exp(logits).sum()
                    p_short, p_flat, p_long = float(probs[0]), float(probs[1]), float(probs[2])
                except Exception as e:
                    print(f"Inference exception at {date_str}: {e}")
                    p_short, p_flat, p_long = 0.33, 0.34, 0.33

                out[h_name] = {
                    "P_long": p_long,
                    "P_flat": p_flat,
                    "P_short": p_short,
                    "depth_long_bins": uniform_depth.copy(),
                    "depth_short_bins": uniform_depth.copy(),
                }

        predictions.append(out)

    if not predictions:
        return {"error": "No predictions generated for the requested date range", "symbol": symbol}

    write_predictions_jsonl(predictions, symbol, settings)

    # Also write versioned consolidated predictions.jsonl
    try:
        write_versioned_predictions(predictions, symbol, "tft", settings, interval=interval)
    except Exception:
        pass

    return {"symbol": symbol, "start": start, "end": end, "predictions": len(predictions)}
