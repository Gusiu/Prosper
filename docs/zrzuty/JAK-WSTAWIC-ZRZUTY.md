# Zrzuty ekranu do dokumentacji

Dokument techniczny ma w rozdziale 11.6 jedenaście gniazd na zrzuty panelu.
Dopóki pliku nie ma, generator wstawia widoczną ramkę z opisem tego, co powinno
się w niej znaleźć. Gdy plik się pojawi — osadza go automatycznie.

## Procedura

1. Uruchom serwer:

```bash
.venv/Scripts/python.exe -m uvicorn prosper.api.server:app --port 8000
```

2. Otwórz `http://localhost:8000`, zrób zrzuty i zapisz je **w tym katalogu**
   pod dokładnie takimi nazwami, jak w tabeli niżej.
3. Przegeneruj dokument:

```bash
.venv/Scripts/python.exe scripts/build_documentation.py
```

Nie trzeba nic zmieniać w kodzie. Brakujące pliki zostawiają ramkę, obecne są
osadzane w szerokości 15,5 cm. Zdanie „Obecnie dostępnych jest N z 11" w treści
rozdziału aktualizuje się samo.

## Czego dotyczy każdy zrzut

| Plik | Co ma być widoczne |
|---|---|
| `01-menedzer-danych.png` | zakładka *Data Manager*, tabela inwentarza, co najmniej trzy symbole |
| `02-pobieranie.png` | formularz pobierania z wypełnionym symbolem i zakresem dat |
| `03-agregacja.png` | panel agregacji z wybranymi interwałami docelowymi |
| `04-trening-formularz.png` | zakładka *AI Models*: wybór modelu, zakres dat, horyzonty |
| `05-trening-pula.png` | rozwinięta sekcja *Advanced: train on several symbols*, model `xgboost`, kilka symboli zaznaczonych |
| `06-kolejka.png` | kolejka z co najmniej jednym zadaniem oczekującym lub wykonywanym |
| `07-ocena-metryki.png` | zakładka *Evaluation*, tabela metryk z przedziałami ufności |
| `08-ocena-stabilnosc.png` | wykres stabilności miesięcznej, widoczne cztery horyzonty |
| `09-backtest-parametry.png` | zakładka *Backtest*, rozwinięty formularz parametrów |
| `10-backtest-wynik.png` | wynik symulacji z widoczną sekcją hipotez zerowych |
| `11-analiza-wykres.png` | wykres analizy z naniesionymi prognozami |

## Wskazówki techniczne

- **Format PNG.** Osadzanie sprawdzone; JPEG też zadziała, ale zrzuty interfejsu
  na PNG są ostrzejsze.
- **Szerokość co najmniej 1400 px.** Obraz jest skalowany do 15,5 cm, więc przy
  mniejszej szerokości tekst na zrzucie będzie nieczytelny w druku.
- **Okno przeglądarki 1600×1000** daje proporcje pasujące do szerokości strony.
- **Wytnij tylko obszar treści**, bez paska adresu i zakładek przeglądarki —
  dokument opisuje aplikację, nie przeglądarkę.
- Jeśli któryś widok wymaga danych, których nie ma (np. pusta kolejka), zrób
  zrzut po uruchomieniu dowolnego zadania — pusty panel niczego nie ilustruje.

## Katalog jest ignorowany przez git

Zrzuty to materiał do dokumentu, nie kod. Wraz z wygenerowanym `.docx` i
wykresami z `data/_figures/` pozostają lokalnie; źródłem prawdy jest
`scripts/build_documentation.py`.
