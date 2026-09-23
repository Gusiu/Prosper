# Prosper

Prosper is a research-grade crypto market analysis platform for Binance SPOT data. It provides an end-to-end pipeline covering data acquisition, aggregation, technical indicator generation, labeling, and machine learning predictions. The system features a lightweight Web UI for data management together with a CLI for automated processing and experiments.

For what the system guarantees and why - no look-ahead, derived training windows, run attribution, and the rest of the rules a result depends on - see [SPECIFICATION.md](SPECIFICATION.md).

---

## 🚀 Features

### Web Dashboard & Data Manager

- **Task Queue Pipeline**: Build complex multi-interval data pipelines (Download -> Aggregate -> Features -> Labels) and execute them asynchronously via an interactive UI.
- **Inventory Matrix**: View your entire data lake at a glance. Visual indicators show precisely where data gaps, missing indicators, or missing labels exist.
- **Auto-Repair System**: One-click "Fix" buttons automatically dispatch background tasks to repair missing or outdated parquet files.
- **Interactive Analysis Drawer**: Slide-out TradingView-style Lightweight Charts implementation.
  - **Bidirectional Sync**: Pan, zoom, and move your crosshair on the main chart, and watch RSI, MACD, and ATR subcharts synchronize perfectly in real-time.
  - **Data Health Overlay**: Visually identify time-gaps in your dataset directly on the chart with warning bands.

### Research & Machine Learning

- **Reproducible runs**: deterministic training and a fixed seed reproduce a run byte for byte on the CPU. Every run records its seed, mode, library versions and git commit in `run_summary.json`.
- **Deep Learning Ready**: Built-in support for time-series forecasting using GRU and TFT (Temporal Fusion Transformer) models via PyTorch.
- **Feature Engineering**: Automated generation of technical indicators (Moving Averages, Bollinger Bands, RSI, MACD, ATR, OBV) mapped natively to Parquet files via Polars.
- **Action Windows**: Convert raw model predictions into actionable time-windows with clearly defined recommendations (Strong Buy/Sell, Accumulate, etc.) based on predicted edge and risk.

---

## 💻 Quickstart & Installation

Two supported ways to set up the project locally: (A) Poetry (recommended if you use Poetry), or (B) a standard venv + pip workflow.

A) Poetry (recommended)

1. Install Python (3.12 recommended) and Poetry.
2. Clone the repository and enter the folder:
   ```powershell
   cd Prosper
   ```
3. (Optional) create venvs inside project:
   ```powershell
   poetry config virtualenvs.in-project true --local
   ```
4. Install:
   ```powershell
   poetry install
   ```

B) Python venv + pip

1. Create and activate a venv (PowerShell):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
3. (Optional) Install package in editable mode to get the `prosper` console command:
   ```powershell
   pip install -e .
   ```

Notes:

- If you use the Poetry route the `prosper` console script is available as `poetry run prosper`.
- Editable install (`pip install -e .`) makes `prosper` available in the active venv.

---

## 🖥️ Starting the API & Web UI (Data Manager)

The simplest route opens the dashboard for you:

```powershell
poetry run prosper ui
```

Add `--no-open-browser` to keep it headless, or `--port` to move it off 8000.
Under the hood it serves the ASGI app, which you can also run directly:

```powershell
poetry run uvicorn prosper.api.server:app --host 127.0.0.1 --port 8000
```

Then open <http://127.0.0.1:8000>.

> The SPA must be served by the API - opening `frontend/index.html` from disk
> leaves every `/api/...` call pointing at the wrong origin, so the dashboard
> loads but stays empty.

The dashboard has no authentication and can delete data, so it listens on
`127.0.0.1` only. It accepts requests from its own origin alone - no CORS - and
refuses any `Host` header that does not name the machine, which closes DNS
rebinding. `prosper ui --host` widens that, with a warning, when you bind to
another interface. Everything the page loads, chart libraries and fonts
included, is served from `frontend/vendor`, so it works offline.

---

## ⚙️ CLI Usage (Headless Mode)

Once dependencies are installed and the package is available (`poetry run` or editable `pip install -e .`), the `prosper` console entrypoint can be used to run headless pipelines.

### Quick smoke pipeline (example)

```powershell
# run with poetry
poetry run prosper backfill --symbol BTCUSDT --start 2020-01 --end 2020-01 --workers 2

# or in a venv with editable install
prosper backfill --symbol BTCUSDT --start 2020-01 --end 2020-01 --workers 2
```

### Predict, evaluate, compare

A prediction run is identified by `{model}_{interval}_{timestamp}`, and every
artifact derived from it - action windows, stability reports, backtests -
is stored under that identity. Nothing shares a "latest" directory, so a result
can always be traced back to the model that produced it.

```powershell
prosper predict xgboost --symbol BTCUSDT --start 2020-09-01 --end 2025-06-30 --interval 1d
prosper eval predictions --symbol BTCUSDT --model-type xgboost --timestamp 20260728224925
prosper eval compare
```

`eval compare` ranks every evaluated run side by side. Check the `Untrained`
column first: a nonzero value means that horizon had no trainable data, so its
metrics cover fewer samples and the scores are not directly comparable.

Horizons (`week` 7d, `month` 28d, `quarter` 91d, `year` 364d) are **calendar
spans**. They are converted to bar counts per interval, so `year` means the same
span whether the run is 1d or 1h. The spans are whole weeks, which keeps them
commensurate at every interval: a year is exactly four quarters.

The training window is derived per horizon from its span and the history
available, so `--train-window-days` is not needed and defaults to unset. Pass it
to force one width on every horizon, and the run fails loudly rather than
emitting a placeholder if that width cannot train the longest horizon.

### Reproducibility flags

Every `predict` command accepts:

- `--deterministic`: enforce deterministic algorithms in PyTorch.
- `--seed <int>`: seed every source of randomness. Each retraining window derives its own
  stream from the seed, model, horizon, month and symbol, so changing one horizon's settings
  does not move the others.

Whatever the flags, each run writes `run_summary.json` beside its predictions with the seed,
the mode, the versions of the libraries that compute a prediction, and the git commit it was
produced from.

See `--help` for individual commands: `prosper --help` or `poetry run prosper --help`.

---

## 📂 Data Storage Architecture

All data is stored in the `./data` directory, utilizing Parquet files partitioned by year and month. Typical layout:

```text
data/
├── raw/                   # Original Binance ZIP files
├── processed/
│   └── binance/spot/
│       ├── klines/        # Processed OHLCV data partitioned by interval
│       │   ├── 1m/symbol={SYMBOL}/year=YYYY/month=MM/part.parquet
│       │   └── 1d/symbol={SYMBOL}/year=YYYY/month=MM/part.parquet
│       ├── features/      # Technical indicators per interval
│       └── labels/        # Ground-truth targets for ML
└── reports/
    ├── predictions/{SYMBOL}/{model}_{interval}_{timestamp}/predictions.jsonl
    ├── evaluations/{SYMBOL}/{model}_{interval}_{timestamp}/
    │       metrics.json, calibration.json, predictions_quality.parquet,
    │       recommendations.json, worst_predictions.csv
    ├── recommendations/{SYMBOL}/{run_slug}/{YYYY-MM}.json
    ├── eval/{SYMBOL}/{run_slug}/stability_summary.json
    ├── backtests/{SYMBOL}/{run_slug}/backtest.json
    └── qa/{SYMBOL}/{interval}/{YYYY-MM}.json
```

`{run_slug}` is the `{model}_{interval}_{timestamp}` of the prediction run the
artifact was derived from.

---

## ⚖️ Disclaimer

**This software is for educational and research purposes only. It is not investment advice.**
Trading cryptocurrencies involves substantial risk of loss. Past performance does not guarantee future results. The authors and contributors of this software are not responsible for any financial losses incurred through the use of this software.

---

## ✅ Running tests

Run unit tests and integration checks with:

```powershell
pytest -q
```

## 📜 License

MIT License - see LICENSE file for details.
