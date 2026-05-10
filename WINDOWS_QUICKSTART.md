# Fresh Windows Quickstart (PowerShell)

Poniższa checklist ma uruchomić kompletny smoke pipeline na świeżym Windows 10/11 (PowerShell), zakładając brak Poetry i brak venv.

## Checklist

1. Instalacja Python 3.12 (zaznacz w instalatorze):
   - `Add to PATH`
   - `pip`
   - `venv`

2. Komendy weryfikacyjne:
   ```powershell
   py -0p
   py -3.12 -V
   ```

3. Instalacja Poetry (preferowane `pipx`):
   ```powershell
   py -m pip install --user pipx
   py -m pipx ensurepath
   restart PowerShell
   ```

4. Instalacja Poetry:
   ```powershell
   pipx install poetry
   poetry --version
   ```

5. Klonowanie repo / wejście do folderu projektu:
   ```powershell
   cd Prosper
   ```

6. Konfiguracja Poetry (żeby venv było w projekcie i nie psuło się na cache):
   ```powershell
   poetry config virtualenvs.in-project true --local
   ```

7. Instalacja zależności:
   ```powershell
   poetry install
   ```

8. Smoke pipeline na `BTCUSDT` w osobnym root:
   ```powershell
   poetry run prosper backfill --symbol BTCUSDT --start 2020-01 --end 2020-01 --workers 2 --root data_smoke
   poetry run prosper qa check --symbol BTCUSDT --interval 1m --year 2020 --month 01 --root data_smoke
   poetry run prosper aggregate --symbol BTCUSDT --from 1m --to 1h 1d 1w --start 2020-01 --end 2020-01 --root data_smoke
   poetry run prosper features build --symbol BTCUSDT --base_interval 1d --start 2020-01-01 --end 2020-01-31 --root data_smoke
   poetry run prosper labels build --symbol BTCUSDT --base_interval 1d --forward_days 1 --root data_smoke
   poetry run prosper predict baseline --symbol BTCUSDT --start 2020-01-01 --end 2020-01-31 --root data_smoke
   poetry run prosper planner windows --symbol BTCUSDT --start 2020-01-01 --end 2020-01-31 --root data_smoke
   ```

9. Uruchomienie testów:
   ```powershell
   poetry run pytest
   ```

10. Alternatywa bez Poetry (venv + pip):
   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   py -m pip install -U pip
   py -m pip install -r requirements.txt
   prosper --help
   ```

## Troubleshooting

1. `poetry` nie znalezione
   - Sprawdź `PATH` i zrestartuj PowerShell.
   - Jeśli używasz `pipx`, wykonaj jeszcze raz: `py -m pipx ensurepath`.

2. Poetry tworzy venv w cache i się psuje
   - Ustaw: `poetry config virtualenvs.in-project true --local`

3. Zła wersja Pythona (3.13 zamiast 3.12)
   - Ustaw explicite:
     ```powershell
     poetry env use $(py -3.12 -c "import sys;print(sys.executable)")
     ```

4. Policy w PowerShell (activate)
   - Dla bieżącego użytkownika:
     ```powershell
     Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
     ```

5. Brak internetu
   - Wygeneruj offline fixture (syntetyczne 1m Parquet) w `data_smoke`, a następnie uruchom smoke pipeline jak w kroku 8.
   - Komenda offline fixture:
     ```powershell
     py scripts\offline_smoke_fixture.py --root data_smoke --symbol BTCUSDT --year 2020 --month 1
     ```
   - Potem możesz uruchomić z kroku 8, w tym `backfill` (pipeline powinien przeskoczyć pobieranie, bo Parquet już istnieje).

