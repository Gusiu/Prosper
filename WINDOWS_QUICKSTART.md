# Fresh Windows Quickstart (PowerShell)

The following checklist is intended to run a complete smoke pipeline on a fresh Windows 10/11 (PowerShell) installation, assuming no Poetry or venv is present.

## Checklist

1. Install Python 3.12 (check the following in the installer):
   - `Add to PATH`
   - `pip`
   - `venv`

2. Verification commands:
   ```powershell
   py -0p
   py -3.12 -V
   ```

3. Install Poetry (prefer `pipx`):
   ```powershell
   py -m pip install --user pipx
   py -m pipx ensurepath
   # restart PowerShell
   ```

4. Install Poetry:
   ```powershell
   pipx install poetry
   poetry --version
   ```

5. Clone repo / enter project folder:
   ```powershell
   cd Prosper
   ```

6. Configure Poetry (so venv is in-project and doesn't break on cache):
   ```powershell
   poetry config virtualenvs.in-project true --local
   ```

7. Install dependencies:
   ```powershell
   poetry install
   ```

8. Smoke pipeline for `BTCUSDT` in a separate root:
   ```powershell
   poetry run prosper backfill --symbol BTCUSDT --start 2020-01 --end 2020-01 --workers 2 --root data_smoke
   poetry run prosper qa check --symbol BTCUSDT --interval 1m --year 2020 --month 01 --root data_smoke
   poetry run prosper aggregate --symbol BTCUSDT --from 1m --to 1h 1d 1w --start 2020-01 --end 2020-01 --root data_smoke
   poetry run prosper features build --symbol BTCUSDT --base_interval 1d --start 2020-01-01 --end 2020-01-31 --root data_smoke
   poetry run prosper labels build --symbol BTCUSDT --base_interval 1d --forward_days 1 --root data_smoke
   poetry run prosper predict baseline --symbol BTCUSDT --start 2020-01-01 --end 2020-01-31 --root data_smoke
   poetry run prosper planner windows --symbol BTCUSDT --start 2020-01-01 --end 2020-01-31 --root data_smoke
   ```

9. Run tests:
   ```powershell
   poetry run pytest
   ```

10. Alternative without Poetry (venv + pip):
   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   py -m pip install -U pip
   py -m pip install -r requirements.txt
   prosper --help
   ```

## Troubleshooting

1. `poetry` not found
   - Check `PATH` and restart PowerShell.
   - If using `pipx`, run again: `py -m pipx ensurepath`.

2. Poetry creates venv in cache and breaks
   - Set: `poetry config virtualenvs.in-project true --local`

3. Wrong Python version (3.13 instead of 3.12)
   - Explicitly set:
     ```powershell
     poetry env use $(py -3.12 -c "import sys;print(sys.executable)")
     ```

4. Execution Policy in PowerShell (activate)
   - For current user:
     ```powershell
     Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
     ```

5. No internet
   - Generate offline fixture (synthetic 1m Parquet) in `data_smoke`, then run smoke pipeline as in step 8.
   - Offline fixture command:
     ```powershell
     py scripts\offline_smoke_fixture.py --root data_smoke --symbol BTCUSDT --year 2020 --month 1
     ```
   - Then you can run from step 8, including `backfill` (the pipeline should skip downloading as Parquet already exists).
