"""Shared JSONL prediction writing utilities."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from prosper.config import Settings, get_settings
from prosper.storage.layout import get_versioned_prediction_dir


def _json_converter(o: object) -> object:
    """Convert numpy and other non-serializable types to built-in types.

    This makes JSON dumping of predictions robust when values are numpy types
    (e.g., float32) coming from model outputs.
    """
    # numpy scalar
    if isinstance(o, np.floating | np.float32 | np.float64):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    # fallback for objects exposing item()
    if hasattr(o, "item"):
        try:
            return o.item()
        except Exception:
            pass
    if isinstance(o, set):
        return list(o)
    raise TypeError(f"Type {type(o)} not serializable")


def write_versioned_predictions(
    predictions: list[dict[str, Any]],
    symbol: str,
    model_type: str,
    settings: Settings | None = None,
    interval: str = "1d",
) -> Path:
    """Write a consolidated `predictions.jsonl` into a versioned folder.

    The folder structure is:
    `reports/predictions/{symbol}/{model_type}_{interval}_{timestamp}/predictions.jsonl`.
    """
    if settings is None:
        settings = get_settings()

    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    out_dir = get_versioned_prediction_dir(
        symbol, model_type, timestamp, settings=settings, interval=interval
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "predictions.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for r in predictions:
            f.write(json.dumps(r, default=_json_converter, sort_keys=True) + "\n")
    return out_dir


# The libraries whose version can change a prediction. A retrained model is
# only comparable with an old one if these match, and "which torch was this"
# is unanswerable after the fact unless it was written down at the time.
RESULT_AFFECTING_PACKAGES = (
    "numpy", "polars", "scikit-learn", "xgboost", "torch", "lightning",
    "pytorch-forecasting", "pandas",
)


def run_environment() -> dict[str, Any]:
    """What produced a run, beyond its own settings.

    This is what `--save-metadata` promised and never did: the flag was accepted
    by fifteen commands and read by none. Written for every run instead, because
    provenance that has to be asked for is provenance that is missing from the
    run nobody expected to need it for.
    """
    packages: dict[str, str | None] = {}
    for name in RESULT_AFFECTING_PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    # The distribution metadata drops torch's build tag, and a CPU and a CUDA
    # build of the same version do not produce the same bytes. The imported
    # module keeps it; torch is only imported if the run used it.
    if "torch" in sys.modules:
        packages["torch"] = str(sys.modules["torch"].__version__)

    source = Path(__file__).resolve().parent.parent
    commit: str | None = None
    dirty: bool | None = None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=source, capture_output=True,
            text=True, timeout=10, check=True,
        ).stdout.strip() or None
        # Tracked changes to the package only: a run from uncommitted code
        # cannot be rebuilt from its commit, and that is the thing worth knowing.
        # Data, docs and scripts do not change a prediction, so they do not count.
        dirty = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", "."], cwd=source,
            capture_output=True, timeout=10,
        ).returncode != 0
    except (OSError, subprocess.SubprocessError):
        pass

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "git_commit": commit,
        "git_dirty": dirty,
    }


def write_run_summary(
    run_dir: Path, summary: dict[str, Any], settings: Settings | None = None
) -> Path:
    """Persist a run's own description beside its predictions.

    Everything a predictor learns about its own execution used to die with the
    process: the result dict went to the caller, the CLI printed two lines of it
    and the rest was gone. `epochs_used` in particular is a *result* — the epoch
    count is decided by early stopping, not by the flag — and it was invisible
    the moment the command exited, so checking whether stopping fired at all
    needed a throwaway instrumented probe every time.

    Invariant 5 says every derived artifact names its source run; the same logic
    says a run should name its own configuration.

    The **seed and the deterministic flag are stamped here**, not by the callers,
    because a run that cannot say what produced it cannot be reproduced — and the
    project's entire reproducibility claim rests on being able to re-run one. The
    omission surfaced while comparing two seeds of the same configuration: the
    artifacts were indistinguishable, and which seed made which had to be
    recovered from a shell log. Stamping it in one place means no predictor can
    forget it, which five separately-maintained result dicts had all managed to.
    """
    if settings is None:
        settings = get_settings()

    model_type, _, rest = run_dir.name.partition("_")
    interval = rest.rpartition("_")[0]

    stamped = {
        "model_type": model_type or None,
        "interval": interval or None,
        "seed": settings.seed,
        "deterministic": bool(settings.deterministic),
        "generated_at": datetime.now(UTC).isoformat(),
        "environment": run_environment(),
        **summary,
    }

    path = run_dir / "run_summary.json"
    path.write_text(
        json.dumps(stamped, indent=2, sort_keys=True, default=_json_converter),
        encoding="utf-8",
    )
    return path
