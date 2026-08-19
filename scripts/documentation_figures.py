"""Render the documentation's figures from the project's own artifacts.

Every chart here is drawn from `data/reports/`, for the same reason the tables
are: a figure redrawn by hand from remembered numbers is a second, unverifiable
copy of the results. Charts are written as PNG into a working directory and
embedded by `build_documentation.py`; nothing is committed, because the figures
are outputs and this file is the source.

Style is deliberately plain — grayscale-safe, no gridline clutter, no decoration
that would survive printing as noise rather than information.
"""

from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from prosper.domain import HORIZON_NAMES  # noqa: E402

REPORT_ROOT = "data/reports"
FIGURE_DIR = "data/_figures"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
MODELS = ("baseline", "ml", "xgboost", "gru", "tft")
ACCENT = "#1F3B63"
MUTED = "#8894A3"

plt.rcParams.update({
    "figure.dpi": 160,
    "font.size": 8,
    "axes.edgecolor": "#4A5563",
    "axes.labelcolor": "#26303C",
    "axes.titlesize": 9,
    "axes.titleweight": "bold",
    "xtick.color": "#4A5563",
    "ytick.color": "#4A5563",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _save(fig, name: str) -> str:
    os.makedirs(FIGURE_DIR, exist_ok=True)
    path = os.path.join(FIGURE_DIR, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _runs() -> dict[tuple[str, str], list[dict]]:
    found: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for path in sorted(glob.glob(f"{REPORT_ROOT}/evaluations/*/*/metrics.json")):
        parts = path.replace("\\", "/").split("/")
        symbol, slug = parts[-3], parts[-2]
        model = slug.split("_")[0]
        if symbol not in SYMBOLS or model not in MODELS:
            continue
        try:
            payload = json.load(open(path, encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        found[(symbol, model)].append(payload.get("by_horizon") or {})
    return found


# ── figures ──────────────────────────────────────────────────────────────────

def architecture_comparison() -> str | None:
    """Mean accuracy per architecture, with the spread across runs.

    The figure the research question asks for: if the whiskers of every bar
    overlap the chance line and each other, the ranking carries no information.
    """
    runs = _runs()
    if not runs:
        return None

    fig, axes = plt.subplots(1, len(HORIZON_NAMES), figsize=(9.5, 2.6), sharey=True)
    for axis, horizon in zip(np.atleast_1d(axes), HORIZON_NAMES, strict=False):
        means, errors, labels = [], [], []
        for model in MODELS:
            values = []
            for symbol in SYMBOLS:
                for by_horizon in runs.get((symbol, model), []):
                    cell = by_horizon.get(horizon)
                    if cell and cell.get("accuracy") is not None:
                        values.append(cell["accuracy"])
            if not values:
                continue
            arr = np.array(values)
            means.append(arr.mean())
            errors.append(arr.std(ddof=1) if len(arr) > 1 else 0.0)
            labels.append(model)
        if not means:
            continue
        positions = np.arange(len(means))
        axis.bar(positions, means, yerr=errors, capsize=2.5,
                 color=ACCENT, edgecolor="white", linewidth=0.6,
                 error_kw={"ecolor": "#C0392B", "elinewidth": 1.0})
        axis.axhline(0.5, color="#C0392B", linewidth=0.9, linestyle="--")
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        axis.set_title(horizon)
        axis.set_ylim(0.35, 0.62)
    np.atleast_1d(axes)[0].set_ylabel("trafność")
    return _save(fig, "porownanie_architektur.png")


def stability_over_time() -> str | None:
    """Monthly accuracy for one representative run per architecture."""
    series: dict[str, list[tuple[str, float]]] = {}
    for path in sorted(glob.glob(f"{REPORT_ROOT}/eval/*/*/stability_summary.json")):
        parts = path.replace("\\", "/").split("/")
        symbol, model = parts[-3], parts[-2].split("_")[0]
        if symbol != "BTCUSDT" or model in series:
            continue
        try:
            payload = json.load(open(path, encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        points = []
        for month in payload.get("months", []):
            cell = (month.get("horizons") or {}).get("week")
            if cell and not cell.get("sparse") and cell.get("accuracy") is not None:
                points.append((str(month.get("month") or month.get("period")), cell["accuracy"]))
        if len(points) > 12:
            series[model] = sorted(points)
    if not series:
        return None

    fig, axis = plt.subplots(figsize=(9.5, 3.0))
    for model, points in sorted(series.items()):
        values = [v for _, v in points]
        axis.plot(range(len(values)), values, linewidth=1.0, label=model, alpha=0.85)
    axis.axhline(0.5, color="#C0392B", linewidth=1.0, linestyle="--", label="poziom losowy")
    axis.set_xlabel("kolejny miesiąc okresu oceny")
    axis.set_ylabel("trafność w miesiącu")
    axis.set_title("BTCUSDT, horyzont tygodniowy - jakość miesiąc po miesiącu")
    axis.legend(frameon=False, ncol=6, fontsize=7, loc="upper center",
                bbox_to_anchor=(0.5, -0.22))
    return _save(fig, "stabilnosc.png")


def seed_spread() -> str | None:
    """Every run as a point, grouped by architecture.

    Shows in one glance what the tables state in numbers: the scatter within an
    architecture is comparable to the distance between architectures.
    """
    runs = _runs()
    if not runs:
        return None

    fig, axis = plt.subplots(figsize=(9.5, 3.0))
    offsets = np.linspace(-0.28, 0.28, len(SYMBOLS))
    for index, model in enumerate(MODELS):
        for offset, symbol in zip(offsets, SYMBOLS, strict=False):
            values = [
                by_horizon["week"]["accuracy"]
                for by_horizon in runs.get((symbol, model), [])
                if by_horizon.get("week") and by_horizon["week"].get("accuracy") is not None
            ]
            if not values:
                continue
            axis.scatter([index + offset] * len(values), values, s=22,
                         edgecolor=ACCENT, facecolor="none", linewidth=0.9)
    axis.axhline(0.5, color="#C0392B", linewidth=1.0, linestyle="--")
    axis.set_xticks(range(len(MODELS)))
    axis.set_xticklabels(MODELS)
    axis.set_ylabel("trafność, horyzont tygodniowy")
    axis.set_title("Każdy przebieg osobno - rozrzut wewnątrz architektury wobec różnic między nimi")
    return _save(fig, "rozrzut_ziarna.png")


def calibration_by_horizon() -> str | None:
    """ECE per architecture and horizon."""
    runs = _runs()
    if not runs:
        return None

    fig, axis = plt.subplots(figsize=(9.5, 2.8))
    width = 0.15
    positions = np.arange(len(HORIZON_NAMES))
    for index, model in enumerate(MODELS):
        means = []
        for horizon in HORIZON_NAMES:
            values = [
                by_horizon[horizon]["ece"]
                for symbol in SYMBOLS
                for by_horizon in runs.get((symbol, model), [])
                if by_horizon.get(horizon) and by_horizon[horizon].get("ece") is not None
            ]
            means.append(float(np.mean(values)) if values else np.nan)
        axis.bar(positions + index * width - 2 * width, means, width * 0.9, label=model)
    axis.set_xticks(positions)
    axis.set_xticklabels(HORIZON_NAMES)
    axis.set_ylabel("ECE (mniej = lepiej)")
    axis.set_title("Błąd kalibracji według horyzontu i architektury")
    axis.legend(frameon=False, ncol=5, fontsize=7)
    return _save(fig, "kalibracja.png")


def backtest_percentiles() -> str | None:
    """Percentile against a run's own circular shifts — the timing test."""
    values: dict[str, list[float]] = defaultdict(list)
    for path in sorted(glob.glob(f"{REPORT_ROOT}/backtests/*/*/backtest.json")):
        parts = path.replace("\\", "/").split("/")
        symbol, model = parts[-3], parts[-2].split("_")[0]
        if symbol not in SYMBOLS or model not in MODELS:
            continue
        try:
            payload = json.load(open(path, encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        percentile = ((payload.get("nulls") or {}).get("mistimed_replay") or {}).get("percentile")
        if percentile is not None:
            values[model].append(float(percentile))
    if not values:
        return None

    fig, axis = plt.subplots(figsize=(9.5, 2.8))
    labels = [m for m in MODELS if m in values]
    axis.boxplot([values[m] for m in labels], tick_labels=labels, widths=0.5)
    axis.axhline(50, color="#C0392B", linewidth=1.0, linestyle="--",
                 label="ułożenie przypadkowe")
    axis.axhline(95, color="#2E7D32", linewidth=1.0, linestyle=":", label="próg sygnału")
    axis.set_ylabel("percentyl wobec własnych przesunięć")
    axis.set_title("Test wyczucia momentu - wartości poniżej 50 oznaczają gorzej niż przypadek")
    axis.legend(frameon=False, fontsize=7)
    return _save(fig, "backtest_percentyle.png")


def reliability_diagram() -> str | None:
    """Declared confidence against realised frequency.

    The picture behind the ECE column: a perfectly calibrated forecast lies on
    the diagonal. Deviation upward means the model is underconfident, downward
    that it claims more than it delivers.
    """
    import polars as pl

    fig, axes = plt.subplots(1, len(HORIZON_NAMES), figsize=(9.5, 2.6), sharey=True)
    drawn = False
    for axis, horizon in zip(np.atleast_1d(axes), HORIZON_NAMES, strict=False):
        confidence, correct = [], []
        for path in glob.glob(f"{REPORT_ROOT}/evaluations/*/*/predictions_quality.parquet"):
            symbol = path.replace("\\", "/").split("/")[-3]
            if symbol not in SYMBOLS:
                continue
            try:
                frame = pl.read_parquet(path).filter(
                    (pl.col("horizon") == horizon) & (pl.col("status") == "scored")
                )
            except Exception:
                continue
            if frame.is_empty():
                continue
            confidence.extend(frame["pred_confidence"].to_list())
            correct.extend(bool(x) for x in frame["correct"].to_list())
        if len(confidence) < 100:
            continue
        conf = np.array(confidence, dtype=float)
        hit = np.array(correct, dtype=float)
        edges = np.linspace(0.5, min(1.0, conf.max()), 9)
        xs, ys = [], []
        for low, high in zip(edges[:-1], edges[1:], strict=False):
            mask = (conf >= low) & (conf < high)
            if mask.sum() < 30:
                continue
            xs.append(conf[mask].mean())
            ys.append(hit[mask].mean())
        if not xs:
            continue
        drawn = True
        axis.plot([0.45, 0.75], [0.45, 0.75], color="#C0392B", linewidth=0.9, linestyle="--")
        axis.plot(xs, ys, marker="o", markersize=3, linewidth=1.1, color=ACCENT)
        axis.set_title(horizon)
        axis.set_xlabel("deklarowana pewność")
        axis.set_xlim(0.47, 0.72)
        axis.set_ylim(0.30, 0.75)
    if not drawn:
        plt.close(fig)
        return None
    np.atleast_1d(axes)[0].set_ylabel("zrealizowana częstość")
    return _save(fig, "niezawodnosc.png")


def equity_curves() -> str | None:
    """Portfolio value over time against buy and hold."""
    fig, axis = plt.subplots(figsize=(9.5, 3.0))
    drawn = 0
    for model in MODELS:
        matches = sorted(glob.glob(f"{REPORT_ROOT}/backtests/BTCUSDT/{model}_1d_*/backtest.json"))
        if not matches:
            continue
        try:
            payload = json.load(open(matches[0], encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        points = payload.get("chart_data") or []
        if len(points) < 100:
            continue
        values = [p.get("value") for p in points if p.get("value") is not None]
        axis.plot(range(len(values)), values, linewidth=1.0, label=model, alpha=0.85)
        drawn += 1
        if drawn == 1:
            prices = [p.get("price") for p in points if p.get("price") is not None]
            if prices and prices[0]:
                scaled = [10000.0 * price / prices[0] for price in prices]
                axis.plot(range(len(scaled)), scaled, linewidth=1.4, linestyle="--",
                          color="#C0392B", label="kup i trzymaj")
    if not drawn:
        plt.close(fig)
        return None
    axis.set_yscale("log")
    axis.set_xlabel("kolejna świeca okresu symulacji")
    axis.set_ylabel("wartość portfela [USDT, skala log.]")
    axis.set_title("BTCUSDT - przebieg kapitału wobec strategii kup i trzymaj")
    axis.legend(frameon=False, ncol=6, fontsize=7, loc="upper center",
                bbox_to_anchor=(0.5, -0.22))
    return _save(fig, "krzywe_kapitalu.png")


def feature_importance() -> str | None:
    """Which columns the gradient model leaned on."""
    matches = sorted(glob.glob(
        f"{REPORT_ROOT}/predictions/BTCUSDT/xgboost_1d_*/feature_importances.json"))
    if not matches:
        return None
    try:
        payload = json.load(open(matches[-1], encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    horizon = "week" if "week" in payload else next(iter(payload), None)
    if not horizon:
        return None
    ranked = sorted(payload[horizon].items(), key=lambda kv: -kv[1])[:18]
    if not ranked:
        return None

    fig, axis = plt.subplots(figsize=(9.5, 3.4))
    names = [name for name, _ in ranked][::-1]
    values = [value for _, value in ranked][::-1]
    axis.barh(range(len(names)), values, color=ACCENT, height=0.7)
    axis.set_yticks(range(len(names)))
    axis.set_yticklabels(names, fontsize=7)
    axis.set_xlabel("względna ważność")
    axis.set_title(f"XGBoost, BTCUSDT, horyzont „{horizon}” - 18 najważniejszych cech")
    return _save(fig, "waznosc_cech.png")


def observation_power() -> str | None:
    """Independent observations per horizon — the binding constraint, drawn."""
    counts: dict[str, list[float]] = defaultdict(list)
    for path in glob.glob(f"{REPORT_ROOT}/evaluations/*/*/metrics.json"):
        if path.replace("\\", "/").split("/")[-3] not in SYMBOLS:
            continue
        try:
            by_horizon = json.load(open(path, encoding="utf-8")).get("by_horizon") or {}
        except (OSError, json.JSONDecodeError):
            continue
        for horizon, cell in by_horizon.items():
            effective = (cell.get("accuracy_ci") or {}).get("effective_samples")
            if effective:
                counts[horizon].append(float(effective))
    if not counts:
        return None

    fig, axis = plt.subplots(figsize=(9.5, 2.8))
    labels = [h for h in HORIZON_NAMES if h in counts]
    means = [float(np.mean(counts[h])) for h in labels]
    axis.bar(range(len(labels)), means, color=ACCENT, width=0.55)
    axis.axhline(10, color="#C0392B", linewidth=1.0, linestyle="--",
                 label="próg wyznaczania przedziału (10 bloków)")
    axis.axhline(3, color="#8894A3", linewidth=1.0, linestyle=":",
                 label="próg sensowności metryki (3 obserwacje)")
    axis.set_yscale("log")
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels)
    axis.set_ylabel("obserwacje niezależne (skala log.)")
    axis.set_title("Moc statystyczna według horyzontu - wiążące ograniczenie projektu")
    axis.legend(frameon=False, fontsize=7)
    return _save(fig, "moc_statystyczna.png")


def build_all() -> dict[str, str]:
    """Render every figure; return name -> path for the ones that had data."""
    figures = {
        "porownanie": architecture_comparison(),
        "rozrzut": seed_spread(),
        "moc": observation_power(),
        "stabilnosc": stability_over_time(),
        "kalibracja": calibration_by_horizon(),
        "niezawodnosc": reliability_diagram(),
        "waznosc": feature_importance(),
        "backtest": backtest_percentiles(),
        "kapital": equity_curves(),
    }
    return {name: path for name, path in figures.items() if path}


if __name__ == "__main__":
    for name, path in build_all().items():
        print(f"{name:<14} {path} ({os.path.getsize(path) / 1024:.0f} kB)")
