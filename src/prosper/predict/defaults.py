"""Training defaults, defined once.

The CLI and the API each used to carry their own copy. Typer's default only
applies when a flag is absent, and `api/commands.py` appends `--epochs`
unconditionally, so the API's value won every time: the same model trained
from the dashboard ran 5 epochs where the CLI ran 20. Nothing was wrong with
either number — the defect was that there were two of them.

Everything that needs a default reads it from here: `cli.py` as its Typer
default, `api/models.py` as its field default, and the browser through
`/api/meta/training-defaults` so the form is pre-filled from the same source
rather than from a value typed into the HTML.
"""

from __future__ import annotations

from prosper.domain import DEFAULT_HORIZONS

# Epoch budgets, not targets. With early stopping on the held-out slice inside
# the training window, a run normally stops well short of these; the number is
# a ceiling on how long one horizon may train.
#
# Measured on BTCUSDT 1d (2023-01..2026-06, 42 retrainings, seed 42): GRU's
# short horizon improves to 20 epochs and plateaus by 40, while medium and long
# stop gaining accuracy after ~5 and their NLL degrades monotonically past it —
# more epochs make them confidently wrong rather than more often right. That
# spread per horizon is exactly what early stopping is for.
DEFAULT_EPOCH_BUDGET: dict[str, int] = {
    "gru": 20,
    "tft": 10,
}

# Models with no epoch concept at all; the flag is not offered for them.
EPOCHLESS_MODELS: frozenset[str] = frozenset({"baseline", "ml", "xgboost"})

HORIZON_NAMES: tuple[str, ...] = tuple(h.name for h in DEFAULT_HORIZONS)

# How many epochs without improvement on the held-out slice before training
# stops. Small because a retraining window holds only a few hundred sequences.
EARLY_STOPPING_PATIENCE = 3

# Below this many held-out rows the stopping signal is noise, so the budget is
# simply spent. Matches the floor the calibrator uses for the same slice.
MIN_EARLY_STOPPING_SAMPLES = 20


def epoch_budget(model_type: str) -> int:
    """Default epoch ceiling for *model_type*, 0 when it does not train epochs."""
    return DEFAULT_EPOCH_BUDGET.get(model_type.lower(), 0)


def resolve_epochs(
    budget: int,
    overrides: dict[str, int] | None = None,
) -> dict[str, int]:
    """Epoch ceiling per horizon: the budget, unless a horizon overrides it."""
    resolved = dict.fromkeys(HORIZON_NAMES, int(budget))
    for name, value in (overrides or {}).items():
        if name in resolved and value:
            resolved[name] = int(value)
    return resolved


def parse_horizons(raw: str | None) -> list[str]:
    """Parse ``'short,long'`` into horizon names, defaulting to all of them.

    Unknown names are an error rather than a silent drop: a typo that quietly
    trained nothing would look exactly like a model that could not learn.
    """
    if raw is None or not raw.strip():
        return list(HORIZON_NAMES)

    requested = [part.strip().lower() for part in raw.split(",") if part.strip()]
    unknown = [name for name in requested if name not in HORIZON_NAMES]
    if unknown:
        raise ValueError(
            f"Unknown horizon(s): {', '.join(unknown)}. "
            f"Known horizons: {', '.join(HORIZON_NAMES)}"
        )
    # Keep the canonical order so artifacts are comparable whatever the caller typed.
    return [name for name in HORIZON_NAMES if name in requested]


def parse_epoch_overrides(raw: str | None) -> dict[str, int]:
    """Parse ``'short=20,medium=5'`` into per-horizon epoch ceilings.

    An unknown horizon or a non-numeric value is an error: silently ignoring
    it would leave the caller believing a budget was applied when it was not.
    """
    if raw is None or not raw.strip():
        return {}

    overrides: dict[str, int] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Expected 'horizon=epochs', got {part!r}")
        name, _, value = part.partition("=")
        name = name.strip().lower()
        if name not in HORIZON_NAMES:
            raise ValueError(
                f"Unknown horizon {name!r}. Known horizons: {', '.join(HORIZON_NAMES)}"
            )
        try:
            epochs = int(value.strip())
        except ValueError as exc:
            raise ValueError(f"Epoch count for {name!r} must be an integer, got {value!r}") from exc
        if epochs < 1:
            raise ValueError(f"Epoch count for {name!r} must be at least 1, got {epochs}")
        overrides[name] = epochs
    return overrides


# Rolling training window, in calendar days. It must stay wider than the
# longest horizon plus a usable sample count or `validate_train_window` refuses
# the configuration; see `prosper.predict.window`. Defined here because the CLI
# repeated the literal in five commands and every predictor signature carried
# its own copy.
DEFAULT_TRAIN_WINDOW_DAYS = 730
