# Prosper - specification

This document states what the system guarantees, and why. It is not a tutorial (see
[`README.md`](README.md)) and not an API reference (that is generated into the technical
documentation from the code itself). It is the list of properties that must hold for a number
produced by this project to be worth reading.

Most of the rules below were not designed up front. They were written after a measurement showed
that their absence produced a result that looked correct - which is the reason each one names the
test that keeps it from coming back.

---

## 1. Scope

**In scope.** Historical Binance SPOT klines; aggregation to higher intervals; technical features
computed from price, volume and order flow; direction and depth labels over four calendar
horizons; five forecasting architectures under one walk-forward protocol; proper-scoring
evaluation with overlap-aware confidence intervals; a decision layer producing Strong Buy through
Strong Sell; a trading simulation reported against four null hypotheses.

**Out of scope, deliberately.** Fundamental data and sentiment (the research question is about
what the quotation series alone carries); order placement and exchange credentials (the measured
forecast quality does not justify acting on it); markets other than crypto (calendar, session
hours and corporate actions each need their own acquisition layer - the pipeline itself is
market-agnostic); order-book microstructure; portfolio construction; automatic hyperparameter
search (with the measured number of independent observations, a search would select the
configuration best fitted to the noise of the evaluation period).

---

## 2. Guarantees

### 2.1 No look-ahead, anywhere

A label at bar `k` needs `close[k + forward_steps]`, so it is trainable at bar `i` only when
`k + forward_steps < i`. `predict/window.py::training_bounds` encodes this and every predictor
goes through it. The trading simulation obeys the same rule from the other end: a signal formed on
bar `t` executes at the **open of bar `t+1`**.

Calibrators are fitted *inside* the training window. `calibration_split` carves its held-out slice
out of the region `training_bounds` already returned, never from later bars - otherwise the
calibrator sees labels the forecast could not.

*Guarded by* `tests/test_predict_leakage.py`, `tests/test_backtest_execution.py`,
`tests/test_calibration.py`.

### 2.2 Every horizon actually trains, or says it did not

`resolve_train_windows` raises when the training window is not wider than the horizon plus
`MIN_TRAIN_SAMPLES`. There is no silent fallback, because the project once shipped for months
emitting a constant `0.33/0.34/0.33` for two of its horizons while the evaluation layer scored the
constant as a forecast.

Where a horizon cannot be trained, its rows carry `trained: false` and evaluation, planner and
backtest all exclude them. Anything that averages the uniform prior into a result is a bug.

*Guarded by* `tests/test_train_window.py`, `tests/test_predictor_training_edges.py`.

### 2.3 The training window is derived, never configured

`window.py::dynamic_window_steps` is `forward + max(MIN_TRAINING_ROWS, 3 x forward)`, clipped by
the available history. The row floor binds the short horizons; the independent-observation floor
binds the long ones.

One shared width meant 103 independent observations for the week horizon against **1.0** for the
year. Widening it globally is not the fix: the history is fixed, so every bar given to training is
a bar that cannot be scored. On BTCUSDT the annual horizon runs 1.0/5.9 train/scored at 730 days,
3.0/3.9 at 1456, 4.0/**2.9** at 1825; the best any window achieves is 3.45 a side. SOLUSDT's
ceiling is 1.95, so its annual horizon is unanswerable at any setting.

`warn_low_power` must therefore be called with **both** sides. A warning about the training side
alone reads as an invitation to widen the window until the evaluation dies.

*Guarded by* `tests/test_train_window.py`, `tests/test_significance.py`.

### 2.4 Horizons are calendar spans

`HorizonSpec.forward_days` is converted per interval via `.steps(interval)`, so a 1h run and a 1d
run of the same horizon mean the same span. The set is `week / month / quarter / year` =
7/28/91/364 days - whole weeks (1/4/13/52), because `days_to_steps` rounds and a 365-day horizon
became 364 at the 1w interval while staying 365 at 1d. A year is exactly four quarters.

The names state the span. The former `short`/`medium`/`long` were vague (`short` was four weeks)
and *collided*: `short` and `long` are also the two direction classes.

Nothing may hardcode the names or the count. Read `HORIZON_NAMES`, and `DEFAULT_TRADED_HORIZON`
for "the one to trade". A run whose horizons this build does not know makes `evaluate_predictions`
raise, rather than score zero rows and look like a model that predicted nothing.

*Guarded by* `tests/test_horizon_set.py`.

### 2.5 Predictions are keyed by `open_time`

Not by calendar date: sub-daily runs put many bars on one date. The key is timezone-aware UTC and
must stay that way - a `datetime64` conversion once silently dropped `+00:00` from every
`open_time`, and only a byte-for-byte comparison against the arithmetic it replaced caught it.

*Guarded by* `tests/test_pooled_training.py`.

### 2.6 Every derived artifact names its source run

`storage/runs.py::resolve_run` is the only way to pick predictions. Artifacts live under
`{report_root}/{symbol}/{run_slug}/`, where `run_slug` is `{model}_{interval}_{timestamp}`. There
is no shared "latest" directory, and nothing may create one.

A run records what produced it - model, interval, seed, deterministic flag, generation time, the
versions of the libraries that can change a prediction (torch with its build tag, since a CPU and a
CUDA build do not produce the same bytes), the git commit and whether the package had uncommitted
changes, and how many retraining windows raised - in `run_summary.json`, so a result can be traced
from the artifact alone without reference to notes.

Deleting a run removes its backtest, evaluation, recommendation and stability directories too.
Otherwise a reader globbing those trees reports a dead run's numbers beside current ones.

*Guarded by* `tests/test_run_attribution.py`, `tests/test_run_provenance.py`,
`tests/test_versioned_predictions.py`.

### 2.7 Training features are scale-free, by an include list

Training reads `domain.py::TRAINING_FEATURE_COLUMNS`. It is an **include** list, and a new column
must be named in it before anything trains on that column.

The predictors used to take every column in the features parquet bar a handful of names, which fed
them raw price levels: measured across BTCUSDT and SOLUSDT over one period, `ma_10` differed 553x,
`atr_14` 360x and `macd_hist` 387x - and that last one has nothing in its name to warn you.
Harmless on one symbol, where the normaliser removes the level; fatal when pooling, where the level
becomes a symbol label. The absolute columns stay in the parquet because the dashboard shows them.

An exclude list would have adopted every relative column *alongside* its absolute twin, which is
exactly why this one is inverted.

*Guarded by* `tests/test_scale_free_features.py`.

### 2.8 Pooled training rows are selected by instant, never by row index

`predict/pool.py` is reached by `predict ml` and `predict xgboost` via `--symbols`. Row 400 of
BTCUSDT is September 2018 and row 400 of SOLUSDT is September 2021, so an index-based pooled window
trains a 2021 forecast on bars that had not happened yet - and the only symptom is better accuracy.
The slice is sorted by instant as well, because `calibration_split` takes its newest rows and
"newest" has to mean newest *overall*.

**Single-symbol runs of those two models take the pooled path too**, so each of them has one
implementation of which rows it may see. The sequence models and the baseline slice one symbol by
index through the same `training_bounds`, which is correct for a single series and the reason they
cannot pool yet.

Pooling does not add observations one for one. `domain.py::effective_symbols` is
`n / (1 + (n-1)rho)`. Mean pairwise correlation of forward returns here is 0.60 weekly and **0.76
annually**, so three symbols are worth 1.36 and 1.19 - and since the limit is `1/rho`, no number of
crypto symbols lifts the annual horizon past about 1.3. Reporting the row count instead (three
symbols, triple the rows) would support a claim the data does not.

*Guarded by* `tests/test_pooled_training.py`, `tests/test_api_pool_command.py`.

### 2.9 Randomness belongs to one trained model

`--deterministic --seed N` reproduces a configuration; it does not isolate one. Both deep-learning
predictors seed per `(model, horizon, year, month, symbol)` via `predict/defaults.py::window_seed`.

Seeding once before the walk-forward loop left every model downstream of every other: the number of
training steps taken anywhere shifted weight init, dropout and shuffling for everything after it.
Raising the `long` epoch ceiling once moved `medium` (+0.070 accuracy) more than `long` itself
(+0.022). Omitting the symbol from the tuple made runs on different symbols share an initialisation
stream, so four "independent" runs were not independent and a significance test reported an effect
that does not exist.

Any hyperparameter experiment must keep an untouched horizon as an in-run control, and check that
it did not move.

*Guarded by* `tests/test_window_seed.py`.

### 2.10 Early stopping measures the calibrated loss

`--epochs` is a ceiling, not a target: GRU and TFT stop when the held-out loss stops improving. The
measured loss is the one *after* temperature scaling (`calibrated_nll`), not raw cross-entropy.

Raw CE charges for confidence. The network sharpens long before it stops discriminating and the
calibrator undoes that a step later, so stopping on the raw value returned a one-epoch model while
accuracy was still rising. That holdout is the calibrator's slice as well; splitting it again would
put both halves under the sample floor at the long horizon.

*Guarded by* `tests/test_early_stopping.py`.

### 2.11 Direction has two classes; depth has one bin scheme

A direction label is the sign of the forward return. There is no `flat` and no threshold: the cost
hurdle lives at the decision layer (`Settings.planner_round_trip_cost`) and magnitude lives in the
depth head, whose `0-1` bin holds exactly the rows `flat` used to. A zero return has no side and is
excluded.

The depth-bin scheme has one definition (`domain.py::DEFAULT_DEPTH_BINS_STR`, currently 12 bins)
and artifacts name their own bins. Never hardcode the labels elsewhere, and never derive the
planner's risk set by naming bins - use `is_large_move_bin`, which reads the edges. A run written
under the old 8-bin scheme still carries `34+`; a reader assuming today's labels drops it, and with
it most of the long-horizon mass.

*Guarded by* `tests/test_labels.py`, `tests/test_depth_scheme.py`.

### 2.12 One object decides a verdict

`planner/windows.py::SequenceGrader`. `plan_windows`, `run_backtest` and `eval/predictions.py` all
use it. They had drifted before - the quantile risk gates and the cost gate went into the planner
only, so a bar could be `Accumulate` in the simulation and `Hold` in the report the simulation was
supposed to be trading. Three paths, found one at a time, hours apart.

The grader is stateful; feed it in chronological order. Its **risk** gates are quantiles of the
risk already seen, never including the bar being graded. Its **edge** thresholds are fixed on
purpose: `edge` is a difference of two probabilities, so 0.3 means the same under any bin scheme,
while `risk` is a product whose range moves whenever the bins or the calibrator do. Edge is bounded
by 0.5 by construction, since `min(p_long, p_short) <= 0.5`.

Risk counts the **adverse side only** - large moves against the position the edge implies. Summing
both sides made an expected rally a reason not to buy, and held 90% of medium and long bars.

`map_to_recommendation` is symmetric: test `|edge|` strongest-first and mirror both directions.
Ordering the sell branches weakest-first once made `Strong Sell` unreachable for every
(edge, risk) pair.

*Guarded by* `tests/test_one_grader.py`, `tests/test_planner_windows.py`,
`tests/test_risk_gates.py`, `tests/test_risk_is_directional.py`, `tests/test_cost_threshold.py`.

### 2.13 The backtest is not the model's score

A model is judged by proper scoring rules - accuracy, Brier, logloss, ECE from
`eval/predictions.py`, and their stability over time from `eval/stability.py`. Those have no free
parameters, so there is nothing to tune and nothing to flatter.

The backtest answers a different question: "would *this strategy* on this forecast have made money,
under these costs, over this period". Its answer moves with every one of those choices - replaying
one run under five exposure policies spans -80% to +267% ROI. Quote it as economic context, never
as evidence the forecast is good.

It therefore reports **four comparisons, not one**. Buy and hold is the weakest: a long-only
strategy that sits flat through volatile months beats it by compounding alone, with no forecast at
all (ETHUSDT tft cleared holding by 11.06 pp while its cumulative monthly excess was negative in
all 40 months). So `nulls` also carries the same average exposure held throughout, the same
exposure path mistimed by every circular shift, and a 7-bar momentum rule containing no model. That
last is the binding one: across the 1d runs it scores a median 90.5 percentile against its own
shifts, where the models score 34.1.

*Guarded by* `tests/test_backtest_nulls.py`, `tests/test_backtest_parameters.py`.

### 2.14 TFT output columns are named by the encoder, never by position

`NaNLabelEncoder` numbers its classes by sorting the labels, so it yields
`{'nan': 0, 'long': 1, 'short': 2}` while `DIRECTION_CLASSES` is `['short', 'long']`. Reading the
columns positionally after dropping the `nan` one exchanged `P_long` and `P_short` in every run ever
written, and no metric the project reports could see it: swapping two classes leaves entropy,
confidence and Brier untouched, and turns accuracy into `1 - accuracy`, which near a coin flip looks
like noise. Always go through `_class_columns`.

*Guarded by* `tests/test_tft_class_order.py`.

### 2.15 There is one walk-forward protocol

`predict/walk_forward.py` owns the loop every model runs in: resolve each horizon's window from the
history (`plan_run`, which also issues the power warning), refit at the first bar of each calendar
month in range, write one row per bar with an explicit untrained payload wherever a horizon could
not be fitted, refuse an empty range, and write the predictions with their run summary
(`finish_run`). A model supplies only `fit_month` and `predict`; `predict/tabular.py` and
`predict/sequence.py` hold what the pooled and the sequence models share.

The loop used to be written out in each of the five predictors, and a rule reached them one at a
time: the baseline wrote an empty run where the others refused to, and TFT skipped the first bar of
a month with no history instead of writing it untrained. The refactor was verified byte for byte:
fifteen configurations covering every model, pooled and single, 1d and 1w, fixed and derived windows,
a horizon subset, seeded and deterministic modes, and a symbol whose history starts inside the range
produce identical predictions before and after - except that one recovered TFT row.

GRU and TFT still normalise missing features differently (GRU after scaling, TFT before). Every
symbol's first 89 bars have missing features and early windows reach them, so aligning the two
would change published numbers; it is recorded in `predict/sequence.py` rather than changed.

*Guarded by* `tests/test_walk_forward.py`, and the structural checks in `tests/test_train_window.py`,
`tests/test_run_provenance.py` and `tests/test_window_seed.py` that every model goes through it.

---

## 3. Operational rules

- **The HTTP layer never runs long work inline.** It spawns `python -m prosper.cli ...` with
  `shell=False`. The browser posts typed JSON; only `api/commands.py` builds argv. Training defaults
  live in `predict/defaults.py` and nowhere else - Typer's default applies only when a flag is
  absent, so the API must *omit* a parameter the caller did not choose. Appending it unconditionally
  is what once made the dashboard train 5 epochs where the CLI trained 20.
  (`tests/test_cli_api_contract.py`, `tests/test_api_pool_command.py`)
- **The work queue lives on the server** (`api/tasks.py`), drained by one worker because the steps
  write the same parquet partitions. The browser renders `/api/task/queue`; it must not keep its own
  copy, or a reload silently loses pending jobs. (`tests/test_task_queue.py`)
- **Lightning `Trainer` must set `enable_checkpointing=False, logger=False`** - otherwise it litters
  the repository root (one run once left 18k directories, 260 MB). `model.predict` builds its own
  Trainer and needs the same kwargs, handed to it as a **copy**: pytorch-forecasting runs
  `trainer_kwargs.setdefault("callbacks", ...)` on the dict it is given, so a shared dict makes every
  call after the first collect into the previous call's callback and return nothing - the horizon
  then degrades to untrained in silence.
- **TFT predicts a whole retraining month in one batched pass.** Per-bar datasets, not the model,
  are what made a run take ~1.5 h. (`tests/test_tft_batched_inference.py`)
- **Parquet is partitioned** `symbol=/year=/month=` (`week=` for 1w). Polars everywhere, not pandas,
  except TFT where pytorch-forecasting requires it.
- **The SPA uses native ES modules** - no build step. `<script type="module">` is deferred, so
  `main.js` boots on `readyState` rather than waiting for `DOMContentLoaded`, which has usually
  already fired. Markup declares behaviour with `data-action`, dispatched by one delegated listener
  in `js/actions.js`; no inline `onclick`.

---

## 4. Acceptance

The system is fit for its purpose when it **measures correctly**, whatever it measures. None of the
criteria below refer to forecast accuracy: that would be unfalsifiable at the time of writing, and
an invitation to repeat the measurement until it returns the expected value.

| Criterion | How it is checked |
| --- | --- |
| Test suite passes | `pytest` |
| Static analysis is clean | `ruff check .` |
| The data lake has no quality defects | `prosper qa check` over all partitions |
| Every horizon trains, or is explicitly `trained: false` | `trained` field per run; `untrained_rows` in evaluation |
| A run under the same seed reproduces | comparison of two runs' output files (exact for the gradient models; for the networks, subject to deterministic kernels) |
| Any reported number is reconstructible from disk | `run_summary.json` carries model, interval, seed, mode, library versions and commit |
| Every result carries an overlap-aware interval | block bootstrap, minimum 10 blocks |
| Every architecture ran under at least three seeds | run count per cell |
| The whole pipeline is reachable from CLI and browser | `prosper --help`; the HTTP route list |
| Deleting a run leaves no derived artifacts | cascade delete |

---

## 5. Known invalid artifacts

- Evaluations whose horizons show `entropy == 1.0985126` (ln 3) and `confidence == 0.34` predate
  the training fix and are invalid.
- `data/reports/predictions/{symbol}/daily/` is orphaned output from before the run-attribution
  refactor. Nothing reads it; do not resurrect it.
- Klines written before 2026-07-29 hold only OHLCV. `candles_to_dataframe` now keeps the order-flow
  columns Binance ships, and `features/build.py` derives `taker_buy_ratio` and friends from them,
  but only where they exist. To get them on existing data, re-run `backfill` (the raw ZIPs are
  already on disk), then rebuild features and recompute models, because the feature set changes.
