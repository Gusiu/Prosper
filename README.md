# Prosper

Prosper is a crypto market analysis tool for Binance SPOT data. It provides a complete pipeline for downloading, processing, quality assurance, and generating predictions and recommendations for cryptocurrency trading pairs.

## Features

- **Data Download**: Backfill historical 1-minute klines from Binance Public Data
- **Data Processing**: Convert raw CSV to optimized Parquet format with automatic timestamp normalization
- **Quality Assurance**: Comprehensive QA checks for data integrity (gaps, duplicates, OHLC invariants)
- **Aggregation**: Resample 1m data to 1h, 1d, and 1w intervals
- **Label Building**: Generate direction (long/flat/short) and depth labels for daily predictions
- **Baseline Predictions**: Simple rolling window frequency-based predictions
- **Action Windows**: Segment predictions into actionable time windows with recommendations

## Requirements

- Python >= 3.12,<3.13
- Poetry for dependency management

## Installation

### Windows

1. Install Poetry (if not already installed):
   ```powershell
   (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
   ```

2. Clone or navigate to the project directory:
   ```powershell
   cd A:\_\PJATK\Praca_inzynierska
   ```

3. Install dependencies:
   ```powershell
   poetry install
   ```

4. Activate the virtual environment:
   ```powershell
   poetry shell
   ```

## Usage

### List Available Symbols

```powershell
poetry run prosper symbols list --quote-asset USDT
```

### Smoke Pipeline (BTCUSDT, 2020-01)

Poniższy smoke pipeline pobiera i przetwarza dane tylko dla `BTCUSDT` i tylko dla `2020-01` (root można zmienić przez `--root`).

```powershell
poetry run prosper backfill --symbol BTCUSDT --start 2020-01 --end 2020-01 --workers 2 --root data_smoke
poetry run prosper qa check --symbol BTCUSDT --interval 1m --year 2020 --month 01 --root data_smoke
poetry run prosper aggregate --symbol BTCUSDT --from 1m --to 1h 1d 1w --start 2020-01 --end 2020-01 --root data_smoke
poetry run prosper features build --symbol BTCUSDT --base_interval 1d --start 2020-01-01 --end 2020-01-31 --root data_smoke
poetry run prosper labels build --symbol BTCUSDT --base_interval 1d --forward_days 1 --root data_smoke
poetry run prosper predict baseline --symbol BTCUSDT --start 2020-01-01 --end 2020-01-31 --root data_smoke
poetry run prosper planner windows --symbol BTCUSDT --start 2020-01-01 --end 2020-01-31 --root data_smoke
```

### Backfill Historical Data

Download and process monthly 1-minute klines:

```powershell
poetry run prosper backfill --symbol BTCUSDT --start 2019-01 --end 2024-12 --workers 4
```

Options:
- `--start`: Start date in YYYY-MM format
- `--end`: End date in YYYY-MM format
- `--workers`: Number of parallel download workers (default: 4)
- `--dry-run`: Simulate without downloading

### Quality Assurance

Run QA checks on processed data:

```powershell
poetry run prosper qa check --symbol BTCUSDT --interval 1m --year 2024 --month 01
```

Checks performed:
- Data sorting
- Duplicate timestamps
- Time gaps
- OHLC invariants (high >= max(open,close), low <= min(open,close), no negative values)

### Aggregate Data

Resample 1m data to higher intervals:

```powershell
poetry run prosper aggregate --symbol BTCUSDT --from 1m --to 1h 1d 1w --start 2019-01 --end 2024-12
```

Supported intervals:
- `1h`: Hourly aggregation (UTC hour boundaries)
- `1d`: Daily aggregation (UTC day boundaries)
- `1w`: Weekly aggregation (ISO weeks, Monday start)

### Build Labels

Generate direction and depth labels for daily predictions:

```powershell
poetry run prosper labels build --symbol BTCUSDT --base_interval 1d --forward_days 1
```

Options:
- `--base-interval`: Base interval for labels (default: 1d)
- `--flat-threshold`: Threshold for flat label (default: 0.01 = 1%)
- `--depth-bins`: Depth bin definitions (default: "1-2,2-3,3-5,5-8,8-13,13-21,21-34,34+")

Label definitions:
- **Direction**:
  - `flat`: |r_{t+1}| <= threshold
  - `long`: r_{t+1} > threshold
  - `short`: r_{t+1} < -threshold
- **Depth**: Conditional binning based on absolute return percentage (only for long/short)

### Generate Baseline Predictions

Generate predictions using rolling window frequencies:

```powershell
poetry run prosper predict baseline --symbol BTCUSDT --start 2024-01-01 --end 2024-12-31
```

Options:
- `--horizon`: Prediction horizon (short/medium/long) - not used in baseline
- `--start`: Start date in YYYY-MM-DD format
- `--end`: End date in YYYY-MM-DD format
- `--window-days`: Rolling window size in days (default: 180)

### Plan Action Windows

Generate actionable recommendations based on predictions:

```powershell
poetry run prosper planner windows --symbol BTCUSDT --start 2024-01-01 --end 2024-12-31
```

Options:
- `--short-weeks`: Short horizon weeks range (default: 1-26)
- `--medium-weeks`: Medium horizon weeks range (default: 13-52)
- `--long-weeks`: Long horizon weeks range (default: 26-104)

Recommendations:
- **Strong Buy**: edge > 0.3, risk < 0.2
- **Buy**: edge > 0.15, risk < 0.3
- **Accumulate**: edge > 0.05, risk < 0.4
- **Hold**: |edge| <= 0.05
- **Reduce**: edge < -0.05, risk < 0.4
- **Sell**: edge < -0.15, risk < 0.3
- **Strong Sell**: edge < -0.3, risk < 0.2

Where:
- `edge = P_long - P_short`
- `risk` = probability of large moves (>5% depth bins)

## Data Storage Layout

All data is stored in the `./data` directory:

```
data/
├── raw/
│   └── binance/spot/klines/1m/{SYMBOL}/
│       ├── {SYMBOL}-1m-YYYY-MM.zip
│       └── {SYMBOL}-1m-YYYY-MM.zip.CHECKSUM
├── processed/
│   └── binance/spot/
│       ├── klines/
│       │   ├── 1m/symbol={SYMBOL}/year=YYYY/month=MM/part.parquet
│       │   ├── 1h/symbol={SYMBOL}/year=YYYY/month=MM/part.parquet
│       │   ├── 1d/symbol={SYMBOL}/year=YYYY/month=MM/part.parquet
│       │   └── 1w/symbol={SYMBOL}/year=YYYY/week=WW/part.parquet
│       └── labels/
│           └── 1d/symbol={SYMBOL}/labels.parquet
└── reports/
    ├── qa/{SYMBOL}/{INTERVAL}/YYYY-MM.json
    ├── predictions/{SYMBOL}/daily/YYYY-MM.jsonl
    └── recommendations/{SYMBOL}/windows/YYYY-MM.json
```

## Weekly Aggregation

Weekly data uses ISO week numbering (weeks start on Monday, UTC). Week boundaries are determined by the Monday of each ISO week.

## Testing

Run tests with pytest:

```powershell
poetry run pytest
```

Run with coverage:

```powershell
poetry run pytest --cov=prosper --cov-report=html
```

## Project Structure

```
prosper/
├── src/prosper/
│   ├── binance/          # Binance API clients
│   ├── baseline/         # Baseline prediction models
│   ├── labels/           # Label building
│   ├── pipeline/         # Data processing pipelines
│   ├── planner/          # Action window planning
│   ├── qa/               # Quality assurance
│   ├── storage/          # Storage layout and Parquet operations
│   ├── utils/            # Utility functions
│   ├── cli.py            # CLI entrypoint
│   └── config.py         # Configuration management
├── tests/                # Test suite
├── pyproject.toml        # Poetry configuration
└── README.md             # This file
```

## Configuration

Configuration is managed through Pydantic Settings. You can:

1. Set environment variables with `PROSPER_` prefix (e.g., `PROSPER_DATA_ROOT`)
2. Create a `config.yaml` file (not implemented in MVP)
3. Use defaults defined in `prosper/config.py`

## Limitations

- MVP focuses only on Binance SPOT market
- Only 1-minute klines are supported for backfill
- Baseline model is simple frequency-based (no machine learning)
- Weekly aggregation uses ISO weeks (Monday start)

## Disclaimer

**This software is for educational and research purposes only. It is not investment advice.**

Trading cryptocurrencies involves substantial risk of loss. Past performance does not guarantee future results. Always do your own research and consult with a qualified financial advisor before making investment decisions.

The authors and contributors of this software are not responsible for any financial losses incurred through the use of this software.

## License

MIT License - see LICENSE file for details.

## Contributing

This is a research project. Contributions are welcome but please ensure:
- Code follows existing style (ruff formatting)
- Tests are added for new features
- Type hints are used throughout
- Documentation is updated
