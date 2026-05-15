# Prosper

Prosper is a research-grade crypto market analysis platform for Binance SPOT data. It provides an end-to-end pipeline covering data acquisition, aggregation, technical indicator generation, labeling, and machine learning predictions. The system features a comprehensive Web UI for data management alongside a powerful CLI for automated processing.

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

## 💻 Installation

1. Install Python (3.12 recommended) and [Poetry](https://python-poetry.org/docs/).
2. Clone the repository and navigate into it:
   ```powershell
   cd Prosper
   ```
3. Configure Poetry to create virtual environments inside the project (optional but recommended):
   ```powershell
   poetry config virtualenvs.in-project true --local
   ```
4. Install dependencies:
   ```powershell
   poetry install
   ```

---

## 🖥️ Starting the Web UI (Data Manager)

To launch the visual dashboard:

```powershell
poetry run prosper ui
```
The server will start on `http://127.0.0.1:8000`. Open this address in your browser to access the Data Manager, Inventory, Task Queue, and Analysis Drawer.

---

## ⚙️ CLI Usage (Headless Mode)

Prosper can also be driven entirely via the Command Line Interface.

### Smoke Pipeline Example
Run a full end-to-end test on BTCUSDT for January 2020:
```powershell
poetry run prosper backfill --symbol BTCUSDT --start 2020-01 --end 2020-01 --workers 2
poetry run prosper qa check --symbol BTCUSDT --interval 1m --year 2020 --month 01
poetry run prosper aggregate --symbol BTCUSDT --from 1m --to 1h 1d 1w --start 2020-01 --end 2020-01
poetry run prosper features build --symbol BTCUSDT --base_interval 1d --start 2020-01-01 --end 2020-01-31
poetry run prosper labels build --symbol BTCUSDT --base_interval 1d --forward_days 1
poetry run prosper predict baseline --symbol BTCUSDT --start 2020-01-01 --end 2020-01-31
poetry run prosper planner windows --symbol BTCUSDT --start 2020-01-01 --end 2020-01-31
```

### Research Mode Flags
When executing CLI commands or background tasks, you can enforce reproducibility:
- `--strict`: Fails fast on data gaps instead of interpolating.
- `--deterministic`: Enforces deterministic algorithms in PyTorch.
- `--seed <int>`: Sets a global random seed for exact reproducibility.
- `--save-metadata`: Generates a `meta.json` audit trail alongside results.

---

## 📂 Data Storage Architecture

All data is stored in the `./data` directory, utilizing highly optimized Parquet files partitioned by year and month.

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

## 📜 License
MIT License - see LICENSE file for details.
