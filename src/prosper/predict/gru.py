"""GRU-based Deep Learning model for probabilistic directional predictions."""
from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from prosper.config import Settings, get_settings
from prosper.storage.layout import get_features_parquet_path, get_prediction_report_path
from prosper.utils.time import parse_date

# ── constants ────────────────────────────────────────────────────────────────
DIRECTION_CLASSES = ["short", "flat", "long"]
DIR_TO_IDX = {c: i for i, c in enumerate(DIRECTION_CLASSES)}

DEPTH_BIN_LABELS = ["1-2", "2-3", "3-5", "5-8", "8-13", "13-21", "21-34", "34+"]
N_DEPTH_BINS = len(DEPTH_BIN_LABELS)


# ── tiny helpers ─────────────────────────────────────────────────────────────
def _direction_from_return(r: float, thr: float) -> str:
    if abs(r) <= thr:
        return "flat"
    return "long" if r > thr else "short"


def _depth_bin(abs_pct: float) -> int:
    edges = [1, 2, 3, 5, 8, 13, 21, 34]
    for i, e in enumerate(edges):
        if abs_pct < e:
            return i
    return len(edges)


# ── dataset ──────────────────────────────────────────────────────────────────
class SequenceDataset(Dataset):
    """Sliding-window sequences of feature rows with direction target."""

    def __init__(
        self,
        X: np.ndarray,          # (T, F)  – feature matrix, already normalised
        y_dir: np.ndarray,      # (T,)    – int labels 0/1/2
        seq_len: int,
    ):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y_dir, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.X) - self.seq_len)

    def __getitem__(self, idx: int):
        x_seq = self.X[idx : idx + self.seq_len]           # (seq_len, F)
        y_lbl = self.y[idx + self.seq_len]                  # scalar
        return x_seq, y_lbl


# ── model ────────────────────────────────────────────────────────────────────
class GRUClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_classes: int = 3,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)           # (B, T, H)
        last = out[:, -1, :]           # (B, H) – last timestep
        last = self.norm(last)
        return self.head(last)         # (B, C)


# ── normalisation ─────────────────────────────────────────────────────────────
def _robust_normalise(X_train: np.ndarray, X_all: np.ndarray):
    """Median/IQR normalisation fitted on train, applied to all."""
    # Replace NaN before statistics so numpy doesn't warn on all-NaN columns
    X_safe = np.where(np.isfinite(X_train), X_train, 0.0)
    med = np.median(X_safe, axis=0)
    q75 = np.percentile(X_safe, 75, axis=0)
    q25 = np.percentile(X_safe, 25, axis=0)
    iqr = q75 - q25
    iqr[iqr == 0] = 1.0
    X_norm = (X_all - med) / iqr
    X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=3.0, neginf=-3.0)
    X_norm = np.clip(X_norm, -5.0, 5.0)
    return X_norm


# ── main entry-point ──────────────────────────────────────────────────────────
def predict_gru(
    symbol: str,
    start: str,
    end: str,
    settings: Settings | None = None,
    seq_len: int = 30,
    train_window_days: int = 365,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    flat_threshold: float = 0.01,
    forward_days: int = 1,
) -> dict[str, Any]:
    """
    Train a GRU classifier on rolling windows and produce daily JSONL predictions.
    Output format is identical to predict_baseline / predict_ml.
    """
    if settings is None:
        settings = get_settings()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 1. Load features ──────────────────────────────────────────────────────
    feat_path = get_features_parquet_path(symbol, "1d", settings=settings)
    if not feat_path.exists():
        return {"error": f"No features parquet for {symbol}. Run: features build", "symbol": symbol}

    df = pl.read_parquet(feat_path, hive_partitioning=False).sort("open_time")
    if df.is_empty():
        return {"error": "Empty features DataFrame", "symbol": symbol}

    df = df.with_columns(pl.col("open_time").dt.date().alias("_date"))

    exclude_cols = {"open_time", "close_time", "_date", "symbol"}
    feature_cols = [c for c in df.columns if c not in exclude_cols and c != "close"]
    if not feature_cols:
        return {"error": "No feature columns found", "symbol": symbol}

    dates = df["_date"].to_list()
    closes = df["close"].to_list()
    X_all_raw = df.select(feature_cols).to_numpy().astype(np.float32)
    n = len(df)

    start_dt = parse_date(start).date()
    end_dt = parse_date(end).date()

    # ── 2. Pre-compute forward labels ─────────────────────────────────────────
    y_dir_all = np.full(n, -1, dtype=np.int64)          # -1 = unknown / future
    for i in range(n - forward_days):
        j = i + forward_days
        if closes[i] and closes[j]:
            r = closes[j] / closes[i] - 1.0
            y_dir_all[i] = DIR_TO_IDX[_direction_from_return(r, flat_threshold)]

    # ── 3. Rolling-month training + inference ─────────────────────────────────
    predictions: list[dict[str, Any]] = []
    # We retrain once per calendar month to balance speed vs. freshness
    current_train_month = None
    model: GRUClassifier | None = None
    scaler_params: tuple[np.ndarray, np.ndarray] | None = None  # (med, iqr)

    for i in range(n):
        row_date = dates[i]
        if row_date < start_dt or row_date > end_dt:
            continue

        train_end = i - 1
        train_start = max(0, i - train_window_days)

        # Retrain once per month
        month_key = (row_date.year, row_date.month)
        if month_key != current_train_month and train_end - train_start >= seq_len + 10:
            current_train_month = month_key

            # Indices with valid labels in train window
            valid = [k for k in range(train_start, train_end + 1) if y_dir_all[k] >= 0]
            if len(valid) >= seq_len + 5 and len(set(y_dir_all[valid])) > 1:
                X_train_raw = X_all_raw[train_start : train_end + 1]
                X_norm = _robust_normalise(X_train_raw, X_all_raw)
                scaler_params = (X_norm,)      # store the full normalised matrix

                # Build dataset from train window
                y_norm = y_dir_all[train_start : train_end + 1]
                ds = SequenceDataset(X_norm[train_start : train_end + 1], y_norm, seq_len)

                if len(ds) < 8:
                    model = None
                    continue

                loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
                m = GRUClassifier(len(feature_cols), hidden_size, num_layers, dropout).to(device)
                opt = torch.optim.Adam(m.parameters(), lr=lr)
                loss_fn = nn.CrossEntropyLoss()

                m.train()
                for _ in range(epochs):
                    for xb, yb in loader:
                        xb, yb = xb.to(device), yb.to(device)
                        opt.zero_grad()
                        loss_fn(m(xb), yb).backward()
                        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                        opt.step()

                model = m
                # keep track of the normalised matrix
                scaler_params = X_norm

        # ── Inference ─────────────────────────────────────────────────────────
        date_str = row_date.isoformat()
        out: dict[str, Any] = {"date": date_str, "symbol": symbol}

        uniform_depth = {lbl: 1.0 / N_DEPTH_BINS for lbl in DEPTH_BIN_LABELS}

        if model is None or scaler_params is None or i < seq_len:
            for h in ("short", "medium", "long"):
                out[h] = {
                    "P_long": 0.33, "P_flat": 0.34, "P_short": 0.33,
                    "depth_long_bins": uniform_depth.copy(),
                    "depth_short_bins": uniform_depth.copy(),
                }
        else:
            X_norm = scaler_params
            x_seq = torch.tensor(X_norm[i - seq_len : i], dtype=torch.float32).unsqueeze(0).to(device)

            model.eval()
            with torch.no_grad():
                logits = model(x_seq)           # (1, 3)
                probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

            p_short, p_flat, p_long = float(probs[0]), float(probs[1]), float(probs[2])
            horizon_out = {
                "P_long": p_long, "P_flat": p_flat, "P_short": p_short,
                "depth_long_bins": uniform_depth.copy(),
                "depth_short_bins": uniform_depth.copy(),
            }
            for h in ("short", "medium", "long"):
                out[h] = horizon_out.copy()

        predictions.append(out)

    if not predictions:
        return {"error": "No predictions generated for the requested date range", "symbol": symbol}

    # ── 4. Write per-month JSONL ───────────────────────────────────────────────
    by_month: dict[str, list[dict[str, Any]]] = {}
    for row in predictions:
        mk = row["date"][:7]
        by_month.setdefault(mk, []).append(row)

    for mk, rows in by_month.items():
        y, m = int(mk[:4]), int(mk[5:7])
        path = get_prediction_report_path(symbol, y, m, settings=settings)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, sort_keys=True) + "\n")

    return {"symbol": symbol, "start": start, "end": end, "predictions": len(predictions), "device": str(device)}
