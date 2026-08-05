"""Prediction-quality evaluation for versioned model runs."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import polars as pl

from prosper.config import Settings, get_settings
from prosper.domain import (
    DEFAULT_HORIZONS,
    DEPTH_BIN_LABELS,
    DIRECTION_CLASSES,
    assign_depth_bin,
    depth_scheme_of,
    direction_from_return,
    parse_depth_bins,
)
from prosper.planner.windows import (
    calculate_edge,
    calculate_risk_metric,
    clears_cost,
    expected_move,
    map_to_recommendation,
)
from prosper.storage.layout import (
    get_evaluation_run_dir,
    get_features_parquet_path,
    get_versioned_prediction_file,
    parse_versioned_prediction_folder,
)
from prosper.storage.parquet import load_parquet

EPS = 1e-12
# Scored over the classes a run actually used, not a fixed tuple: a run
# written before the flat class was dropped names a `flat` probability, and
# reading it under today's two classes would renormalise its mass into the
# other two and silently rescore it.
DIRECTION_ORDER: tuple[str, ...] = tuple(DIRECTION_CLASSES)
INTERVAL_SECONDS = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "8h": 28800,
    "12h": 43200,
    "1d": 86400,
    "3d": 259200,
    "1w": 604800,
}
FEATURE_SNAPSHOT_COLUMNS = (
    "log_return_1b",
    "rolling_vol_14",
    "rolling_vol_30",
    "candle_range",
    "ma_10",
    "ma_30",
    "ma_diff",
    "bb_upper",
    "bb_mid",
    "bb_lower",
    "rsi_14",
    "macd_hist",
    "atr_14",
    "obv",
    "realized_vol_1d",
    "jump_count",
    "max_intraday_drawdown",
)


@dataclass(frozen=True)
class EvaluationSettings:
    """Tunable thresholds used by prediction-quality evaluation."""

    calibration_bins: int = 10
    low_confidence_threshold: float = 0.45
    high_entropy_ratio: float = 0.92
    confident_wrong_threshold: float = 0.65
    large_return_error_pct: float = 5.0
    worst_predictions_limit: int = 100
    # The full ranking is recoverable from predictions_quality.parquet; this
    # file only needs the head that a reader would actually look at.
    recommendations_limit: int = 500


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def direction_scheme_of(values: dict[str, Any]) -> tuple[str, ...]:
    """The direction classes a stored prediction carries, as `P_<class>` keys."""
    present = tuple(
        key[2:] for key in values if key.startswith("P_") and values.get(key) is not None
    )
    return present or DIRECTION_ORDER


def normalize_probabilities(
    values: dict[str, Any], classes: Iterable[str] | None = None
) -> dict[str, float]:
    """Return normalized direction probabilities over the run's own classes."""
    names = tuple(classes) if classes is not None else direction_scheme_of(values)
    raw = {name: max(0.0, _finite_float(values.get(f"P_{name}"))) for name in names}
    total = sum(raw.values())
    if total <= EPS:
        return {name: 1.0 / max(1, len(names)) for name in names}
    return {name: prob / total for name, prob in raw.items()}


def normalize_distribution(values: dict[str, Any], labels: Iterable[str]) -> dict[str, float]:
    """Normalize a labeled probability distribution with uniform fallback."""
    labels_list = list(labels)
    raw = {label: max(0.0, _finite_float(values.get(label))) for label in labels_list}
    total = sum(raw.values())
    if total <= EPS:
        return {label: 1.0 / max(1, len(labels_list)) for label in labels_list}
    return {label: value / total for label, value in raw.items()}


def brier_score(probs: dict[str, float], y_true: str) -> float:
    """Multiclass Brier score over whatever classes *probs* carries."""
    return sum(
        (prob - (1.0 if direction == y_true else 0.0)) ** 2 for direction, prob in probs.items()
    )


def negative_log_likelihood(probs: dict[str, float], y_true: str) -> float:
    """Multiclass negative log-likelihood."""
    return -math.log(max(EPS, probs.get(y_true, 0.0)))


def probability_entropy(probs: dict[str, float]) -> float:
    """Shannon entropy in natural-log units."""
    return -sum(prob * math.log(max(prob, EPS)) for prob in probs.values())


def kl_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    """KL(P || Q) for matching labeled distributions."""
    labels = set(p) | set(q)
    total = 0.0
    for label in labels:
        p_value = p.get(label, 0.0)
        if p_value <= EPS:
            continue
        total += p_value * math.log(p_value / max(EPS, q.get(label, 0.0)))
    return total


def js_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    """Jensen-Shannon divergence for matching labeled distributions."""
    labels = set(p) | set(q)
    midpoint = {label: 0.5 * (p.get(label, 0.0) + q.get(label, 0.0)) for label in labels}
    return 0.5 * kl_divergence(p, midpoint) + 0.5 * kl_divergence(q, midpoint)


def expected_calibration_error(
    samples: list[dict[str, float | bool]],
    bins: int = 10,
) -> tuple[float, list[dict[str, float | int]]]:
    """Compute ECE and return calibration-bin diagnostics."""
    if not samples:
        return 0.0, []

    buckets: list[list[dict[str, float | bool]]] = [[] for _ in range(bins)]
    for sample in samples:
        confidence = min(1.0, max(0.0, float(sample["confidence"])))
        idx = min(bins - 1, int(confidence * bins))
        buckets[idx].append(sample)

    total = len(samples)
    ece = 0.0
    curve: list[dict[str, float | int]] = []
    for idx, bucket in enumerate(buckets):
        lower = idx / bins
        upper = (idx + 1) / bins
        if bucket:
            count = len(bucket)
            avg_confidence = sum(float(s["confidence"]) for s in bucket) / count
            accuracy = sum(1.0 for s in bucket if bool(s["correct"])) / count
            gap = abs(accuracy - avg_confidence)
            ece += (count / total) * gap
        else:
            count = 0
            avg_confidence = 0.0
            accuracy = 0.0
            gap = 0.0
        curve.append(
            {
                "bin": idx,
                "lower": lower,
                "upper": upper,
                "count": count,
                "accuracy": accuracy,
                "confidence": avg_confidence,
                "gap": gap,
            }
        )
    return ece, curve


def _date_key(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=UTC)
        return dt.astimezone(UTC).date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value)
    if not text:
        return None
    if len(text) >= 10:
        return text[:10]
    return text


def _bar_key(value: Any) -> str | None:
    """Identity of a single bar, precise to the second.

    Sub-daily runs put many bars on the same calendar date, so keying by date
    alone would score every hour of a day against the same candle.
    """
    dt = _datetime_from_value(value)
    if dt is None:
        return None
    return dt.astimezone(UTC).isoformat()


def _prediction_bar_key(prediction: dict[str, Any]) -> str | None:
    """Prefer the precise ``open_time``; fall back to legacy date-only rows."""
    return _bar_key(prediction.get("open_time") or prediction.get("date"))


def _datetime_from_value(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=UTC)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            try:
                parsed = datetime.strptime(value[:10], "%Y-%m-%d")
            except ValueError:
                return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return None


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _read_predictions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def _load_ohlc(symbol: str, interval: str, settings: Settings) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    base_path = settings.processed_binance_spot_klines_dir / interval / f"symbol={symbol}"
    if not base_path.exists():
        raise FileNotFoundError(f"No OHLC directory found for {symbol} {interval}: {base_path}")

    parts = [load_parquet(p) for p in sorted(base_path.glob("**/*.parquet")) if p.is_file() and p.stat().st_size > 0]
    if not parts:
        raise FileNotFoundError(f"No OHLC parquet files found for {symbol} {interval}: {base_path}")

    df = (
        pl.concat(parts)
        .with_columns(pl.col("open_time").cast(pl.Datetime("us", "UTC")).alias("open_time"))
        .sort("open_time")
        .unique(subset=["open_time"], keep="first")
    )
    rows = df.select(["open_time", "open", "high", "low", "close", "volume"]).to_dicts()
    return rows, _detect_time_gaps(rows, interval)


def _detect_time_gaps(rows: list[dict[str, Any]], interval: str) -> list[dict[str, Any]]:
    expected_seconds = INTERVAL_SECONDS.get(interval)
    if not expected_seconds or len(rows) < 2:
        return []

    gaps: list[dict[str, Any]] = []
    prev_dt = _datetime_from_value(rows[0].get("open_time"))
    for row in rows[1:]:
        current_dt = _datetime_from_value(row.get("open_time"))
        if prev_dt and current_dt:
            delta_seconds = (current_dt - prev_dt).total_seconds()
            if delta_seconds > expected_seconds * 1.5:
                gaps.append(
                    {
                        "from": prev_dt.isoformat(),
                        "to": current_dt.isoformat(),
                        "missing_intervals_estimate": max(1, round(delta_seconds / expected_seconds) - 1),
                    }
                )
        prev_dt = current_dt
    return gaps


def _load_feature_snapshots(symbol: str, interval: str, settings: Settings) -> dict[str, dict[str, Any]]:
    path = get_features_parquet_path(symbol, interval, settings=settings)
    if not path.exists():
        return {}
    try:
        df = load_parquet(path)
    except Exception:
        return {}
    if "open_time" not in df.columns:
        return {}

    available_cols = ["open_time", *[col for col in FEATURE_SNAPSHOT_COLUMNS if col in df.columns]]
    snapshots: dict[str, dict[str, Any]] = {}
    for row in df.select(available_cols).to_dicts():
        key = _date_key(row.get("open_time"))
        if not key:
            continue
        snapshot: dict[str, Any] = {}
        for col, value in row.items():
            if col == "open_time" or value is None:
                continue
            numeric = _finite_float(value, default=float("nan"))
            if math.isfinite(numeric):
                snapshot[col] = numeric
        snapshots.setdefault(key, snapshot)
    return snapshots


def _run_depth_scheme(predictions: list[dict[str, Any]]) -> list[str]:
    """The depth-bin labels a run was written with.

    Read from the first horizon that carries a distribution; runs that predate
    the extended tail name their own `34+` bin and are scored on it.
    """
    for prediction in predictions:
        for horizon in DEFAULT_HORIZONS:
            horizon_pred = prediction.get(horizon.name) or {}
            for key in ("depth_long_bins", "depth_short_bins"):
                bins = horizon_pred.get(key)
                if bins:
                    return depth_scheme_of(bins)
    return list(DEPTH_BIN_LABELS)


def _depth_midpoints(labels: list[str]) -> dict[str, float]:
    ranges, parsed_labels = parse_depth_bins(",".join(labels))
    midpoints: dict[str, float] = {}
    for label, (low, high) in zip(parsed_labels, ranges, strict=True):
        midpoints[label] = (low + high) / 2.0 if high is not None else low * 1.25
    return midpoints


def _expected_depth_pct(depth_probs: dict[str, float], midpoints: dict[str, float]) -> float:
    return sum(depth_probs.get(label, 0.0) * midpoint for label, midpoint in midpoints.items())


def _recommendation_payload(
    horizon_pred: dict[str, Any],
    probs: dict[str, float],
    round_trip_cost: float,
) -> dict[str, Any]:
    # Normalise over the bins the run itself used, not the current constant:
    # runs written under the 8-bin scheme name a `34+` bin that no longer
    # exists, and normalising them against today's labels would drop the whole
    # tail — most of their probability mass — without raising anything.
    raw_long = horizon_pred.get("depth_long_bins", {}) or {}
    raw_short = horizon_pred.get("depth_short_bins", {}) or {}
    depth_long = normalize_distribution(raw_long, depth_scheme_of(raw_long))
    depth_short = normalize_distribution(raw_short, depth_scheme_of(raw_short))
    edge = calculate_edge(probs["long"], probs["short"])
    risk = calculate_risk_metric(probs["long"], probs["short"], depth_long, depth_short)
    move = expected_move(probs["long"], probs["short"], depth_long, depth_short)
    cost_cleared = clears_cost(move, round_trip_cost)
    return {
        "edge": edge,
        "risk": risk,
        "expected_move": move,
        "cost_cleared": cost_cleared,
        "recommendation": map_to_recommendation(edge, risk, cost_cleared=cost_cleared),
        "depth_long_bins": depth_long,
        "depth_short_bins": depth_short,
    }


def max_entropy(n_classes: int) -> float:
    """Entropy of a uniform guess over *n_classes* — the "knows nothing" point.

    Every normalisation below is expressed against it. It used to be written as
    `math.log(3.0)` in three places, which was right while direction had three
    classes and silently wrong afterwards: a coin-flip forecast over two classes
    has NLL ln 2 = 0.693, and dividing that by ln 3 = 1.099 credited pure
    guessing with 37% of the points meant for skill.
    """
    return math.log(max(2, n_classes))


def _score_prediction(
    *,
    correct: bool,
    confidence: float,
    brier: float,
    nll: float,
    depth_abs_error: float | None,
    n_classes: int,
) -> float:
    direction_component = 45.0 if correct else 0.0
    brier_component = 25.0 * max(0.0, 1.0 - (brier / 2.0))
    nll_component = 20.0 * max(0.0, 1.0 - (nll / max_entropy(n_classes)))
    confidence_component = 10.0 * (confidence if correct else 1.0 - confidence)
    depth_component = 10.0
    if depth_abs_error is not None:
        depth_component = 10.0 * max(0.0, 1.0 - (depth_abs_error / 20.0))
    return round(min(100.0, direction_component + brier_component + nll_component + confidence_component + depth_component), 4)


def _quality_flags(
    *,
    correct: bool,
    confidence: float,
    entropy: float,
    settings: EvaluationSettings,
    depth_abs_error: float | None,
    predicted_depth_bin: str | None,
    actual_depth_bin: str | None,
    n_classes: int,
) -> list[str]:
    flags: list[str] = []
    if confidence < settings.low_confidence_threshold:
        flags.append("low_confidence")
    if entropy > max_entropy(n_classes) * settings.high_entropy_ratio:
        flags.append("high_entropy")
    if not correct and confidence >= settings.confident_wrong_threshold:
        flags.append("calibration_mismatch")
    if depth_abs_error is not None and depth_abs_error >= settings.large_return_error_pct:
        flags.append("large_return_error")
    if actual_depth_bin and predicted_depth_bin and predicted_depth_bin != actual_depth_bin:
        flags.append("depth_mismatch")
    return flags


def _quality_reason(row: dict[str, Any]) -> str:
    actual = row.get("actual", {})
    pred = row.get("prediction", {})
    metrics = row.get("metrics", {})
    if "missing_target" in row.get("flags", []):
        return "Target candle is not available yet for this horizon."
    verdict = "correct" if metrics.get("correct") else "wrong"
    return (
        f"{verdict} {pred.get('direction')} vs {actual.get('direction')} "
        f"at {pred.get('confidence', 0.0):.1%} confidence; "
        f"return {actual.get('return_pct', 0.0):.2f}%."
    )


def _safe_mean(values: list[float]) -> float | None:
    values = [value for value in values if math.isfinite(value)]
    if not values:
        return None
    return sum(values) / len(values)


def _safe_rmse(values: list[float]) -> float | None:
    values = [value for value in values if math.isfinite(value)]
    if not values:
        return None
    return math.sqrt(sum(value * value for value in values) / len(values))


def _aggregate_quality(
    quality_rows: list[dict[str, Any]],
    settings: EvaluationSettings,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics_by_horizon: dict[str, Any] = {}
    calibration_by_horizon: dict[str, Any] = {}
    scored_rows = [row for row in quality_rows if row.get("status") == "scored"]

    for horizon in [h.name for h in DEFAULT_HORIZONS]:
        rows = [row for row in scored_rows if row.get("horizon") == horizon]
        cal_samples = [
            {
                "confidence": float(row["prediction"]["confidence"]),
                "correct": bool(row["metrics"]["correct"]),
            }
            for row in rows
        ]
        ece, curve = expected_calibration_error(cal_samples, settings.calibration_bins)
        calibration_by_horizon[horizon] = {"ece": ece, "bins": curve}

        briers = [float(row["metrics"]["brier"]) for row in rows]
        nlls = [float(row["metrics"]["nll"]) for row in rows]
        entropies = [float(row["metrics"]["entropy"]) for row in rows]
        depth_kls = [float(row["metrics"]["depth_kl"]) for row in rows if row["metrics"].get("depth_kl") is not None]
        depth_jss = [float(row["metrics"]["depth_js"]) for row in rows if row["metrics"].get("depth_js") is not None]
        depth_errors = [float(row["metrics"]["depth_abs_error_pct"]) for row in rows if row["metrics"].get("depth_abs_error_pct") is not None]
        accuracy = (sum(1 for row in rows if row["metrics"].get("correct")) / len(rows)) if rows else None
        avg_brier = _safe_mean(briers)
        avg_nll = _safe_mean(nlls)
        avg_entropy = _safe_mean(entropies)
        avg_confidence = _safe_mean([float(row["prediction"]["confidence"]) for row in rows])
        mae_depth = _safe_mean(depth_errors)
        rmse_depth = _safe_rmse(depth_errors)
        # Every row of a run carries the same classes, so the first is enough.
        horizon_classes = len(rows[0]["prediction"]["probabilities"]) if rows else len(
            DIRECTION_ORDER
        )
        horizon_score = None
        if rows and accuracy is not None and avg_brier is not None and avg_nll is not None:
            horizon_score = 100.0 * (
                0.35 * accuracy
                + 0.25 * max(0.0, 1.0 - avg_brier / 2.0)
                + 0.20 * max(0.0, 1.0 - avg_nll / max_entropy(horizon_classes))
                + 0.10 * max(0.0, 1.0 - ece)
                + 0.10 * (max(0.0, 1.0 - (rmse_depth or 0.0) / 25.0) if rmse_depth is not None else 1.0)
            )

        metrics_by_horizon[horizon] = {
            "samples": len(rows),
            "accuracy": accuracy,
            "brier": avg_brier,
            "nll": avg_nll,
            "ece": ece,
            "entropy": avg_entropy,
            "confidence": avg_confidence,
            "kl_depth": _safe_mean(depth_kls),
            "js_depth": _safe_mean(depth_jss),
            "mae_expected_depth_pct": mae_depth,
            "rmse_expected_depth_pct": rmse_depth,
            "score": horizon_score,
        }

    all_samples = [
        {
            "confidence": float(row["prediction"]["confidence"]),
            "correct": bool(row["metrics"]["correct"]),
        }
        for row in scored_rows
    ]
    overall_ece, overall_curve = expected_calibration_error(all_samples, settings.calibration_bins)
    calibration = {
        "overall": {"ece": overall_ece, "bins": overall_curve},
        "by_horizon": calibration_by_horizon,
    }

    total_samples = sum(item["samples"] for item in metrics_by_horizon.values())
    weighted_score_sum = sum(
        item["score"] * item["samples"]
        for item in metrics_by_horizon.values()
        if item["score"] is not None and item["samples"] > 0
    )
    model_score = weighted_score_sum / total_samples if total_samples else None
    overall_accuracy = (
        sum(1 for row in scored_rows if row["metrics"].get("correct")) / len(scored_rows)
        if scored_rows
        else None
    )

    metrics = {
        "schema_version": 1,
        "metric_definitions": {
            "accuracy": "Share of horizons where predicted direction equals realized direction.",
            "brier": "Multiclass Brier score over the run's direction classes; lower is better.",
            "nll": "Negative log-likelihood of the realized direction; lower is better.",
            "ece": "Expected calibration error over confidence bins; lower is better.",
            "entropy": "Prediction entropy in natural-log units; lower means sharper predictions.",
            "kl_depth": "KL divergence between realized depth-bin one-hot distribution and predicted depth distribution.",
            "js_depth": "Jensen-Shannon divergence for depth-bin distributions.",
            "mae_expected_depth_pct": "Mean absolute error between expected depth percent and realized absolute return percent.",
            "rmse_expected_depth_pct": "RMSE of expected depth percent against realized absolute return percent.",
            "score": "Composite 0-100 quality score; higher is better.",
            "untrained_rows": (
                "Rows the model could not train for this horizon. They carry a uniform "
                "placeholder distribution and are excluded from every metric above."
            ),
        },
        "overall": {
            "samples": len(scored_rows),
            "missing_targets": sum(1 for row in quality_rows if row.get("status") != "scored"),
            "accuracy": overall_accuracy,
            "brier": _safe_mean([float(row["metrics"]["brier"]) for row in scored_rows]),
            "nll": _safe_mean([float(row["metrics"]["nll"]) for row in scored_rows]),
            "ece": overall_ece,
            "entropy": _safe_mean([float(row["metrics"]["entropy"]) for row in scored_rows]),
            "model_score": model_score,
        },
        "by_horizon": metrics_by_horizon,
    }
    return metrics, calibration


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


QUALITY_COLUMNS: tuple[str, ...] = (
    "symbol",
    "model_type",
    "timestamp",
    "interval",
    "date",
    "open_time",
    "horizon",
    "forward_steps",
    "forward_days",
    "status",
    "target_date",
    "pred_direction",
    "pred_confidence",
    "pred_p_short",
    "pred_p_long",
    "pred_edge",
    "pred_risk",
    "pred_expected_move",
    "pred_cost_cleared",
    "pred_recommendation",
    "pred_expected_depth_pct",
    "pred_depth_bin",
    "actual_direction",
    "actual_return",
    "actual_return_pct",
    "actual_depth_bin",
    "correct",
    "brier",
    "nll",
    "entropy",
    "depth_kl",
    "depth_js",
    "depth_abs_error_pct",
    "quality_score",
    "error_score",
    "flags",
    "reason",
)


def flatten_quality_row(row: dict[str, Any]) -> dict[str, Any]:
    """Flatten one nested quality row into a tabular record.

    The nested JSON shape is convenient to build but expensive to store and
    impossible to scan lazily; a 1h run produced a 331 MB JSONL that the API
    then parsed in full to return a hundred rows.
    """
    prediction = row.get("prediction", {}) or {}
    actual = row.get("actual", {}) or {}
    metrics = row.get("metrics", {}) or {}
    probabilities = prediction.get("probabilities", {}) or {}

    record: dict[str, Any] = {
        "symbol": row.get("symbol"),
        "model_type": row.get("model_type"),
        "timestamp": row.get("timestamp"),
        "interval": row.get("interval"),
        "date": row.get("date"),
        "open_time": row.get("open_time"),
        "horizon": row.get("horizon"),
        "forward_steps": row.get("forward_steps"),
        "forward_days": row.get("forward_days"),
        "status": row.get("status"),
        "target_date": row.get("target_date"),
        "pred_direction": prediction.get("direction"),
        "pred_confidence": prediction.get("confidence"),
        "pred_p_short": probabilities.get("short"),
        "pred_p_long": probabilities.get("long"),
        "pred_edge": prediction.get("edge"),
        "pred_risk": prediction.get("risk"),
        "pred_expected_move": prediction.get("expected_move"),
        "pred_cost_cleared": prediction.get("cost_cleared"),
        "pred_recommendation": prediction.get("recommendation"),
        "pred_expected_depth_pct": prediction.get("expected_depth_pct"),
        "pred_depth_bin": prediction.get("predicted_depth_bin"),
        "actual_direction": actual.get("direction"),
        "actual_return": actual.get("return"),
        "actual_return_pct": actual.get("return_pct"),
        "actual_depth_bin": actual.get("depth_bin"),
        "correct": metrics.get("correct"),
        "brier": metrics.get("brier"),
        "nll": metrics.get("nll"),
        "entropy": metrics.get("entropy"),
        "depth_kl": metrics.get("depth_kl"),
        "depth_js": metrics.get("depth_js"),
        "depth_abs_error_pct": metrics.get("depth_abs_error_pct"),
        "quality_score": row.get("quality_score"),
        "error_score": row.get("error_score"),
        "flags": "|".join(row.get("flags", []) or []),
        "reason": row.get("reason"),
    }
    for name, value in (row.get("feature_snapshot", {}) or {}).items():
        record[f"feat_{name}"] = value
    return record


def _write_quality_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write quality rows as Parquet so consumers can scan them lazily."""
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [flatten_quality_row(row) for row in rows]
    if not records:
        pl.DataFrame({column: [] for column in QUALITY_COLUMNS}).write_parquet(path)
        return

    frame = pl.DataFrame(records, infer_schema_length=None)
    ordered = [c for c in QUALITY_COLUMNS if c in frame.columns]
    extras = sorted(c for c in frame.columns if c not in QUALITY_COLUMNS)
    frame.select([*ordered, *extras]).write_parquet(path, compression="zstd")


def _write_worst_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "date",
        "horizon",
        "target_date",
        "actual_direction",
        "predicted_direction",
        "confidence",
        "quality_score",
        "error_score",
        "recommendation",
        "flags",
        "reason",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "date": row.get("date"),
                    "horizon": row.get("horizon"),
                    "target_date": row.get("target_date"),
                    "actual_direction": row.get("actual", {}).get("direction"),
                    "predicted_direction": row.get("prediction", {}).get("direction"),
                    "confidence": row.get("prediction", {}).get("confidence"),
                    "quality_score": row.get("quality_score"),
                    "error_score": row.get("error_score"),
                    "recommendation": row.get("prediction", {}).get("recommendation"),
                    "flags": "|".join(row.get("flags", [])),
                    "reason": row.get("reason"),
                }
            )


def evaluate_predictions(
    symbol: str,
    model_type: str,
    timestamp: str,
    interval: str = "1d",
    settings: Settings | None = None,
    eval_settings: EvaluationSettings | None = None,
) -> dict[str, Any]:
    """Evaluate a versioned prediction run and write quality artifacts."""
    settings = settings or get_settings()
    eval_settings = eval_settings or EvaluationSettings()
    symbol = symbol.upper()
    model_type = model_type.lower()

    prediction_path = get_versioned_prediction_file(
        symbol,
        model_type,
        timestamp,
        settings=settings,
        interval=interval,
    )
    if not prediction_path.exists():
        raise FileNotFoundError(f"Versioned predictions not found: {prediction_path}")

    predictions = _read_predictions(prediction_path)
    ohlc_rows, gaps = _load_ohlc(symbol, interval, settings)
    feature_snapshots = _load_feature_snapshots(symbol, interval, settings)

    # Index bars by their exact timestamp; legacy date-only prediction files
    # additionally resolve through the first bar of each calendar date.
    bar_to_index: dict[str, int] = {}
    date_to_index: dict[str, int] = {}
    for idx, row in enumerate(ohlc_rows):
        bar_key = _bar_key(row.get("open_time"))
        if bar_key and bar_key not in bar_to_index:
            bar_to_index[bar_key] = idx
        date_key = _date_key(row.get("open_time"))
        if date_key and date_key not in date_to_index:
            date_to_index[date_key] = idx

    # Score every run against the bin scheme it was written with. Binning the
    # realised return under one scheme while the forecast used another would
    # compare distributions over different label sets — the depth Brier and the
    # expected-depth error would both be meaningless.
    depth_labels = _run_depth_scheme(predictions)
    depth_ranges, _ = parse_depth_bins(",".join(depth_labels))
    midpoints = _depth_midpoints(depth_labels)
    quality_rows: list[dict[str, Any]] = []
    matched_predictions = 0
    untrained_by_horizon: dict[str, int] = {h.name: 0 for h in DEFAULT_HORIZONS}

    for prediction in predictions:
        pred_date = _date_key(prediction.get("open_time") or prediction.get("date"))
        if not pred_date:
            continue
        pred_key = _prediction_bar_key(prediction)
        pred_index = bar_to_index.get(pred_key) if pred_key else None
        if pred_index is None:
            pred_index = date_to_index.get(pred_date)
        if pred_index is not None:
            matched_predictions += 1

        for horizon in DEFAULT_HORIZONS:
            horizon_pred = prediction.get(horizon.name) or {}
            forward_steps = horizon.steps(interval)

            # A horizon the model could not train emits a uniform placeholder.
            # Scoring it would report the prior as if it were a forecast.
            if horizon_pred.get("trained") is False:
                untrained_by_horizon[horizon.name] += 1
                continue

            probs = normalize_probabilities(horizon_pred)
            predicted_direction = max(probs, key=probs.get)
            confidence = probs[predicted_direction]
            entropy = probability_entropy(probs)
            recommendation = _recommendation_payload(
                horizon_pred, probs, settings.planner_round_trip_cost
            )

            base_row: dict[str, Any] = {
                "symbol": symbol,
                "model_type": model_type,
                "timestamp": timestamp,
                "interval": interval,
                "date": pred_date,
                "open_time": pred_key,
                "horizon": horizon.name,
                "forward_steps": forward_steps,
                "forward_days": horizon.forward_days,
                "feature_snapshot": feature_snapshots.get(pred_date, {}),
                "prediction": {
                    "direction": predicted_direction,
                    "confidence": confidence,
                    "probabilities": probs,
                    "edge": recommendation["edge"],
                    "risk": recommendation["risk"],
                    "expected_move": recommendation["expected_move"],
                    "cost_cleared": recommendation["cost_cleared"],
                    "recommendation": recommendation["recommendation"],
                },
                "flags": [],
            }

            target_index = None if pred_index is None else pred_index + forward_steps
            if pred_index is None or target_index is None or target_index >= len(ohlc_rows):
                base_row.update(
                    {
                        "status": "missing_target",
                        "target_date": None,
                        "actual": {"direction": None, "return": None, "return_pct": None},
                        "metrics": {"correct": False, "brier": None, "nll": None, "entropy": entropy},
                        "quality_score": 0.0,
                        "error_score": 100.0,
                        "flags": ["missing_ohlc" if pred_index is None else "missing_target"],
                    }
                )
                base_row["reason"] = _quality_reason(base_row)
                quality_rows.append(base_row)
                continue

            start_close = _finite_float(ohlc_rows[pred_index].get("close"))
            target_close = _finite_float(ohlc_rows[target_index].get("close"))
            realized_return = (target_close / start_close - 1.0) if start_close > EPS else 0.0
            realized_return_pct = realized_return * 100.0
            actual_direction = direction_from_return(realized_return)
            target_date = _date_key(ohlc_rows[target_index].get("open_time"))

            # A forward return of exactly zero has no side, so there is nothing
            # to score the forecast against. Excluding it is the same rule an
            # untrained horizon follows: never invent an outcome.
            if actual_direction is None:
                base_row.update(
                    {
                        "status": "unclassifiable_outcome",
                        "target_date": target_date,
                        "actual": {
                            "direction": None,
                            "return": realized_return,
                            "return_pct": realized_return_pct,
                        },
                        "metrics": {
                            "correct": False,
                            "brier": None,
                            "nll": None,
                            "entropy": entropy,
                        },
                        "quality_score": 0.0,
                        "error_score": 0.0,
                        "flags": ["zero_return"],
                    }
                )
                base_row["reason"] = _quality_reason(base_row)
                quality_rows.append(base_row)
                continue

            actual_depth_idx = assign_depth_bin(abs(realized_return_pct), depth_ranges)
            actual_depth_bin = depth_labels[actual_depth_idx] if actual_depth_idx is not None else None

            predicted_depth_distribution = None
            if predicted_direction == "long":
                predicted_depth_distribution = recommendation["depth_long_bins"]
            elif predicted_direction == "short":
                predicted_depth_distribution = recommendation["depth_short_bins"]
            elif probs["long"] >= probs["short"]:
                predicted_depth_distribution = recommendation["depth_long_bins"]
            else:
                predicted_depth_distribution = recommendation["depth_short_bins"]
            expected_depth_pct = _expected_depth_pct(predicted_depth_distribution, midpoints)
            predicted_depth_bin = max(predicted_depth_distribution, key=predicted_depth_distribution.get)

            actual_depth_one_hot = None
            depth_kl = None
            depth_js = None
            depth_abs_error = None
            if actual_depth_bin:
                actual_depth_one_hot = {label: 1.0 if label == actual_depth_bin else 0.0 for label in depth_labels}
                depth_kl = kl_divergence(actual_depth_one_hot, predicted_depth_distribution)
                depth_js = js_divergence(actual_depth_one_hot, predicted_depth_distribution)
                depth_abs_error = abs(expected_depth_pct - abs(realized_return_pct))

            correct = predicted_direction == actual_direction
            row_brier = brier_score(probs, actual_direction)
            row_nll = negative_log_likelihood(probs, actual_direction)
            quality_score = _score_prediction(
                n_classes=len(probs),
                correct=correct,
                confidence=confidence,
                brier=row_brier,
                nll=row_nll,
                depth_abs_error=depth_abs_error,
            )
            flags = _quality_flags(
                n_classes=len(probs),
                correct=correct,
                confidence=confidence,
                entropy=entropy,
                settings=eval_settings,
                depth_abs_error=depth_abs_error,
                predicted_depth_bin=predicted_depth_bin,
                actual_depth_bin=actual_depth_bin,
            )

            base_row.update(
                {
                    "status": "scored",
                    "target_date": target_date,
                    "actual": {
                        "direction": actual_direction,
                        "return": realized_return,
                        "return_pct": realized_return_pct,
                        "depth_bin": actual_depth_bin,
                    },
                    "prediction": {
                        **base_row["prediction"],
                        "expected_depth_pct": expected_depth_pct,
                        "predicted_depth_bin": predicted_depth_bin,
                    },
                    "metrics": {
                        "correct": correct,
                        "brier": row_brier,
                        "nll": row_nll,
                        "entropy": entropy,
                        "depth_kl": depth_kl,
                        "depth_js": depth_js,
                        "depth_abs_error_pct": depth_abs_error,
                    },
                    "quality_score": quality_score,
                    "error_score": round(100.0 - quality_score, 4),
                    "flags": flags,
                }
            )
            base_row["reason"] = _quality_reason(base_row)
            quality_rows.append(base_row)

    metrics, calibration = _aggregate_quality(quality_rows, eval_settings)
    for horizon_name, untrained in untrained_by_horizon.items():
        metrics["by_horizon"][horizon_name]["untrained_rows"] = untrained
    metrics["overall"]["untrained_rows"] = sum(untrained_by_horizon.values())

    run_dir = get_evaluation_run_dir(
        symbol, model_type, timestamp, settings=settings, interval=interval
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    scored_rows = [row for row in quality_rows if row.get("status") == "scored"]
    worst_rows = sorted(scored_rows, key=lambda row: row.get("error_score", 0.0), reverse=True)[
        : eval_settings.worst_predictions_limit
    ]
    # Only the head of this ranking is ever shown. Serialising every scored row
    # produced a 97 MB recommendations.json that the API loaded in full.
    recommendations = sorted(
        scored_rows,
        key=lambda row: (
            row.get("quality_score", 0.0),
            abs(row.get("prediction", {}).get("edge", 0.0)),
        ),
        reverse=True,
    )[: eval_settings.recommendations_limit]

    metadata = {
        "symbol": symbol,
        "model_type": model_type,
        "timestamp": timestamp,
        "interval": interval,
        "prediction_path": str(prediction_path),
        "artifacts_dir": str(run_dir),
        "predictions_loaded": len(predictions),
        "predictions_matched": matched_predictions,
        "quality_rows": len(quality_rows),
        "scored_rows": len(scored_rows),
        "untrained_horizons": untrained_by_horizon,
        "gaps_detected": len(gaps),
        "gap_preview": gaps[:20],
        "created_at": datetime.now(tz=UTC).isoformat(),
    }
    metrics = {**metadata, **metrics}
    metrics["artifacts"] = {
        "metrics": str(run_dir / "metrics.json"),
        "predictions_quality": str(run_dir / "predictions_quality.parquet"),
        "calibration": str(run_dir / "calibration.json"),
        "worst_predictions": str(run_dir / "worst_predictions.csv"),
        "recommendations": str(run_dir / "recommendations.json"),
    }

    recommendations_payload = {
        "schema_version": 2,
        "symbol": symbol,
        "model_type": model_type,
        "timestamp": timestamp,
        "interval": interval,
        "limit": eval_settings.recommendations_limit,
        "scored_rows": len(scored_rows),
        "truncated": len(scored_rows) > eval_settings.recommendations_limit,
        "items": [
            {
                "date": row["date"],
                "horizon": row["horizon"],
                "target_date": row.get("target_date"),
                "recommendation": row["prediction"].get("recommendation"),
                "quality_score": row.get("quality_score"),
                "edge": row["prediction"].get("edge"),
                "risk": row["prediction"].get("risk"),
                "confidence": row["prediction"].get("confidence"),
                "predicted_direction": row["prediction"].get("direction"),
                "actual_direction": row.get("actual", {}).get("direction"),
                "flags": row.get("flags", []),
            }
            for row in recommendations
        ],
    }

    _write_json(run_dir / "metrics.json", metrics)
    _write_quality_parquet(run_dir / "predictions_quality.parquet", quality_rows)
    _write_json(run_dir / "calibration.json", calibration)
    _write_worst_csv(run_dir / "worst_predictions.csv", worst_rows)
    _write_json(run_dir / "recommendations.json", recommendations_payload)

    return {
        "status": "ok",
        "metrics": metrics,
        "calibration": calibration,
        "artifacts": metrics["artifacts"],
        "worst_predictions": worst_rows,
        "recommendations": recommendations_payload["items"],
    }




def evaluate_available_model_runs(
    symbol: str | None = None,
    model_type: str | None = None,
    interval: str | None = None,
    limit: int | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Evaluate saved prediction runs in bulk for scheduler/cron usage."""
    settings = settings or get_settings()
    base = settings.reports_predictions_dir
    symbol_filter = symbol.upper() if symbol else None
    model_filter = model_type.lower() if model_type else None
    interval_filter = interval or None
    results: list[dict[str, Any]] = []

    if not base.exists():
        return {"status": "ok", "evaluated": 0, "failed": 0, "results": []}

    for symbol_dir in sorted(base.iterdir()):
        if not symbol_dir.is_dir():
            continue
        if symbol_filter and symbol_dir.name.upper() != symbol_filter:
            continue
        for run_dir in sorted(symbol_dir.iterdir()):
            if limit is not None and len(results) >= limit:
                break
            if not run_dir.is_dir():
                continue
            parsed = parse_versioned_prediction_folder(run_dir.name)
            if not parsed:
                continue
            parsed_model, parsed_interval, parsed_timestamp = parsed
            if model_filter and parsed_model.lower() != model_filter:
                continue
            if interval_filter and parsed_interval != interval_filter:
                continue
            try:
                evaluation = evaluate_predictions(
                    symbol=symbol_dir.name,
                    model_type=parsed_model,
                    timestamp=parsed_timestamp,
                    interval=parsed_interval,
                    settings=settings,
                )
                results.append(
                    {
                        "status": "ok",
                        "symbol": symbol_dir.name,
                        "model_type": parsed_model,
                        "interval": parsed_interval,
                        "timestamp": parsed_timestamp,
                        "model_score": evaluation["metrics"].get("overall", {}).get("model_score"),
                        "artifacts_dir": evaluation["metrics"].get("artifacts_dir"),
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "status": "failed",
                        "symbol": symbol_dir.name,
                        "model_type": parsed_model,
                        "interval": parsed_interval,
                        "timestamp": parsed_timestamp,
                        "error": str(exc),
                    }
                )
        if limit is not None and len(results) >= limit:
            break

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(tz=UTC).isoformat(),
        "evaluated": sum(1 for item in results if item["status"] == "ok"),
        "failed": sum(1 for item in results if item["status"] == "failed"),
        "results": results,
    }
    status_path = settings.reports_dir / "evaluations" / "batch_status.json"
    _write_json(status_path, payload)
    payload["status_path"] = str(status_path)
    return payload


def collect_evaluation_summaries(
    symbol: str | None = None,
    interval: str | None = None,
    settings: Settings | None = None,
) -> list[dict[str, Any]]:
    """Read every evaluation's metrics.json into a flat comparison table.

    Sorted by composite score, best first. `untrained_rows` is carried through
    deliberately: a horizon the model could not train contributes no samples,
    so a score is only comparable once you know that column is zero.
    """
    settings = settings or get_settings()
    base = settings.reports_dir / "evaluations"
    if not base.exists():
        return []

    symbol_filter = symbol.upper() if symbol else None
    rows: list[dict[str, Any]] = []
    for symbol_dir in sorted(base.iterdir()):
        if not symbol_dir.is_dir() or symbol_dir.name.startswith("."):
            continue
        if symbol_filter and symbol_dir.name.upper() != symbol_filter:
            continue
        for run_dir in sorted(symbol_dir.iterdir()):
            metrics_path = run_dir / "metrics.json"
            if not metrics_path.is_file():
                continue
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if interval and metrics.get("interval") != interval:
                continue

            overall = metrics.get("overall", {})
            by_horizon = metrics.get("by_horizon", {})
            rows.append(
                {
                    "symbol": metrics.get("symbol", symbol_dir.name),
                    "model_type": metrics.get("model_type"),
                    "interval": metrics.get("interval"),
                    "timestamp": metrics.get("timestamp"),
                    "samples": overall.get("samples"),
                    "untrained_rows": overall.get("untrained_rows", 0),
                    "model_score": overall.get("model_score"),
                    "accuracy": overall.get("accuracy"),
                    "brier": overall.get("brier"),
                    "nll": overall.get("nll"),
                    "ece": overall.get("ece"),
                    "entropy": overall.get("entropy"),
                    **{
                        f"acc_{name}": by_horizon.get(name, {}).get("accuracy")
                        for name in ("short", "medium", "long")
                    },
                }
            )

    rows.sort(key=lambda row: (row["model_score"] is not None, row["model_score"] or 0.0), reverse=True)
    return rows
