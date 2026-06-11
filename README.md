# Prosper

Prosper is a research-grade crypto market analysis platform for Binance SPOT data. It provides an end-to-end pipeline covering data acquisition, aggregation, technical indicator generation, labeling, and machine learning predictions. The system features a lightweight Web UI for data management together with a CLI for automated processing and experiments.

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

- **Research Mode**: Toggle strict data validation, deterministic CUDA execution, and global seeds for 100% reproducible experiments. Meta-data tracking saves environment variables alongside models.
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

Start the API server (uvicorn):

```powershell
# using poetry
poetry run uvicorn src.prosper.api.server:app --reload --host 127.0.0.1 --port 8000

# or from a venv where the package is installed
uvicorn src.prosper.api.server:app --reload --host 127.0.0.1 --port 8000
```

Frontend (static SPA) can be opened directly in the browser or served with a tiny static server. Quick options:

```powershell
# open file directly (development only)
start frontend\index.html

# or serve via a simple HTTP server and open http://127.0.0.1:8001
python -m http.server --directory frontend 8001
```

Alternatively run the packaged UI entry if available:

```powershell
poetry run prosper ui
```

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

### Research Mode Flags

When executing CLI commands or background tasks, you can enforce reproducibility:

- `--strict`: Fail fast on data gaps instead of interpolating.
- `--deterministic`: Enforce deterministic algorithms in PyTorch.
- `--seed <int>`: Set a global random seed for reproducibility.
- `--save-metadata`: Generate a `meta.json` audit trail alongside results.

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
│       └── labels/        # Ground-truth targets for ML
└── reports/               # QA reports, predictions, and recommendations
```

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
