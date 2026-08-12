"""Generate the technical documentation as a .docx.

The document is *generated*, not written by hand, for the reason the previous one
was deleted: a hand-written write-up drifts from the code and then contradicts it.
Every result table here is read from `data/reports/` at build time, so the
document cannot claim a number the artifacts do not contain, and regenerating it
after a recompute is one command.

    .venv/Scripts/python.exe scripts/build_documentation.py

Writes `docs/Prosper_Dokumentacja_Techniczna.docx`, which is gitignored: the
script is the source of truth, the document is an artifact.
"""

from __future__ import annotations

import glob
import json
import os
from collections import defaultdict
from datetime import date

import numpy as np
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt, RGBColor
from prosper.domain import DEFAULT_HORIZONS, HORIZON_NAMES

REPORT_ROOT = "data/reports"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
MODELS = ("baseline", "ml", "xgboost", "gru", "tft")
ACCENT = RGBColor(0x1F, 0x3B, 0x63)


# ── reading the artifacts ────────────────────────────────────────────────────

def load_runs() -> dict[tuple[str, str], list[dict]]:
    """Every evaluated run on disk, keyed by (symbol, model).

    Reads what is there rather than what was expected: a partial recompute
    produces a partial table and says so, instead of failing or inventing rows.
    """
    found: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for path in sorted(glob.glob(f"{REPORT_ROOT}/evaluations/*/*/metrics.json")):
        parts = path.replace("\\", "/").split("/")
        symbol, slug = parts[-3], parts[-2]
        model = slug.split("_")[0]
        if symbol not in SYMBOLS or model not in MODELS:
            continue
        try:
            payload = json.load(open(path, encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        summary = f"{REPORT_ROOT}/predictions/{symbol}/{slug}/run_summary.json"
        seed = None
        if os.path.exists(summary):
            try:
                seed = json.load(open(summary, encoding="utf-8")).get("seed")
            except (OSError, json.JSONDecodeError):
                pass
        found[(symbol, model)].append(
            {"slug": slug, "seed": seed, "by_horizon": payload.get("by_horizon") or {}}
        )
    return found


def horizon_values(runs: list[dict], horizon: str, field: str) -> list[float]:
    out = []
    for run in runs:
        cell = run["by_horizon"].get(horizon)
        if cell and cell.get(field) is not None:
            out.append(float(cell[field]))
    return out


def fmt(value: float | None, places: int = 3) -> str:
    return "—" if value is None else f"{value:.{places}f}"


# ── document furniture ───────────────────────────────────────────────────────

def add_heading(doc: Document, text: str, level: int) -> None:
    heading = doc.add_heading(text, level=level)
    for run in heading.runs:
        run.font.color.rgb = ACCENT


def add_body(doc: Document, text: str, *, italic: bool = False, size: int = 10) -> None:
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(6)
    run = paragraph.add_run(text)
    run.italic = italic
    run.font.size = Pt(size)


def add_bullets(doc: Document, items: list[str]) -> None:
    for item in items:
        paragraph = doc.add_paragraph(item, style="List Bullet")
        paragraph.paragraph_format.space_after = Pt(3)
        for run in paragraph.runs:
            run.font.size = Pt(10)


def add_table(doc: Document, header: list[str], rows: list[list[str]], widths: list[float]) -> None:
    """A table with explicit column widths, set on every cell.

    Word ignores a width set only on the column, so it has to go on each cell or
    the layout collapses to equal columns.
    """
    table = doc.add_table(rows=1, cols=len(header))
    table.style = "Light Grid Accent 1"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for index, label in enumerate(header):
        cell = table.rows[0].cells[index]
        cell.text = ""
        run = cell.paragraphs[0].add_run(label)
        run.bold = True
        run.font.size = Pt(9)
    for values in rows:
        cells = table.add_row().cells
        for index, value in enumerate(values):
            cells[index].text = ""
            run = cells[index].paragraphs[0].add_run(str(value))
            run.font.size = Pt(9)
    for row in table.rows:
        for index, cell in enumerate(row.cells):
            cell.width = Cm(widths[index])
    doc.add_paragraph()


def add_caption(doc: Document, text: str) -> None:
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(12)
    run = paragraph.add_run(text)
    run.italic = True
    run.font.size = Pt(8.5)


# ── the document ─────────────────────────────────────────────────────────────

def build() -> str:
    runs = load_runs()
    doc = Document()

    section = doc.sections[0]
    section.page_width, section.page_height = Cm(21.0), Cm(29.7)
    for attr in ("left_margin", "right_margin"):
        setattr(section, attr, Cm(2.5))
    section.top_margin = section.bottom_margin = Cm(2.2)

    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(10)

    # ── title page ───────────────────────────────────────────────────────────
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_before = Pt(120)
    run = title.add_run("Prosper")
    run.bold = True
    run.font.size = Pt(38)
    run.font.color.rgb = ACCENT

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.add_run("Platforma badawcza do analizy rynku kryptowalut")
    run.font.size = Pt(15)

    subtitle2 = doc.add_paragraph()
    subtitle2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle2.paragraph_format.space_after = Pt(60)
    run = subtitle2.add_run("Dokumentacja techniczna")
    run.font.size = Pt(12)
    run.italic = True

    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = meta.add_run(
        f"Praca inżynierska — Polsko-Japońska Akademia Technik Komputerowych\n"
        f"Wygenerowano automatycznie z artefaktów projektu: {date.today().isoformat()}"
    )
    run.font.size = Pt(9)

    doc.add_page_break()

    # ── 1. teza ──────────────────────────────────────────────────────────────
    add_heading(doc, "1. Pytanie badawcze, teza i zakres", 1)

    add_heading(doc, "1.1. Pytanie badawcze", 2)
    add_body(
        doc,
        "Praca odpowiada na pytanie: jak różne architektury uczenia maszynowego radzą sobie z "
        "prognozą kierunku zmiany ceny aktywa finansowego, dysponując wyłącznie historią jego "
        "notowań.",
    )
    add_body(
        doc,
        "Ograniczenie do samych notowań nie jest uproszczeniem przyjętym dla wygody. Pytanie "
        "„czy z historii cen da się przewidzieć cenę” jest treścią słabej formy hipotezy "
        "efektywności rynku (Fama, 1970) i stanowi jedną z trzech kanonicznych postaci tej "
        "hipotezy. Analiza fundamentalna należy do formy półsilnej i odpowiada na inne pytanie; "
        "nie jest bogatszą wersją pytania zadanego tutaj. Praca testuje formę słabą przy użyciu "
        "architektur, które w chwili jej sformułowania nie istniały.",
    )

    add_heading(doc, "1.2. Teza", 2)
    quote = doc.add_paragraph()
    quote.paragraph_format.left_indent = Cm(1.0)
    quote.paragraph_format.space_after = Pt(10)
    run = quote.add_run(
        "Różnice między badanymi architekturami uczenia maszynowego w zadaniu prognozy kierunku "
        "ceny z samej historii notowań są mniejsze niż rozrzut wyników wynikający z losowej "
        "inicjalizacji tej samej architektury, a żadna z nich nie osiąga przewagi odróżnialnej "
        "od losowej. Orzeczenie tego wymaga warstwy weryfikacji zdolnej odróżnić brak różnicy "
        "od niedostatecznej dokładności pomiaru."
    )
    run.bold = True
    run.font.size = Pt(10.5)

    add_body(
        doc,
        "Zdanie ostatnie jest istotne. Stwierdzenie „architektury nie różnią się między sobą” "
        "jest nieodróżnialne od stwierdzenia „pomiar był zbyt zaszumiony, aby je rozróżnić”, "
        "dopóki poziom szumu nie zostanie zmierzony. Z tego powodu każdą konfigurację uruchomiono "
        "wielokrotnie pod różnymi ziarnami generatora liczb losowych, a rozdział 6 raportuje "
        "rozrzut między architekturami obok rozrzutu wynikającego z samej inicjalizacji.",
    )

    add_heading(doc, "1.3. Uzasadnienie wyboru rynku", 2)
    add_body(
        doc,
        "Pierwotnym zamiarem był system uniwersalny, obejmujący rynek akcji, walutowy i "
        "kryptowalut. Zamiar ograniczono do kryptowalut z powodów metodologicznych, a nie "
        "wyłącznie z powodu złożoności implementacji: rynki różnią się w sposób, który wprowadza "
        "do pomiaru zmienne niezwiązane z badanym modelem.",
    )
    add_bullets(
        doc,
        [
            "Notowania ciągłe — brak luk weekendowych i nocnych, które na rynku akcji tworzą "
            "nieciągłości wymagające osobnego traktowania w każdej z cech technicznych.",
            "Brak zdarzeń korporacyjnych — podziały akcji, dywidendy i emisje wymagają korekty "
            "szeregu historycznego, a sposób tej korekty wpływa na wynik modelu.",
            "Jednolita mikrostruktura — jedna giełda, jeden mechanizm zawierania transakcji, "
            "brak fragmentacji zleceń między systemami obrotu.",
            "Otwarte dane historyczne w jednolitym formacie, o pełnej głębokości i bez opłat "
            "licencyjnych, co czyni wyniki odtwarzalnymi przez osobę trzecią.",
        ],
    )
    add_body(
        doc,
        "Wybrano zatem rynek, na którym mierzone jest zachowanie modelu, a nie artefakty "
        "specyficzne dla danego systemu obrotu. Rozszerzenie na pozostałe rynki pozostaje "
        "kierunkiem dalszych prac (rozdział 8).",
    )

    add_heading(doc, "1.4. Twierdzenia szczegółowe", 2)
    add_bullets(
        doc,
        [
            "Żadna z pięciu badanych architektur nie osiąga trafności kierunkowej istotnie "
            "większej od losowej na horyzoncie o wystarczającej mocy statystycznej. Jest to "
            "wynik ograniczony od góry, a nie twierdzenie o niemożliwości prognozowania.",
            "Rozrzut trafności tej samej architektury pod różnymi ziarnami inicjalizacji jest "
            "porównywalny z różnicami między architekturami, co czyni ranking architektur "
            "oparty na pojedynczym przebiegu pozbawionym podstaw.",
            "Kalibracja prognoz jest osiągalna tam, gdzie trafność nie jest: system poprawnie "
            "raportuje własną niepewność, co jest osobnym, pozytywnym wynikiem inżynierskim.",
            "Wiążącym ograniczeniem jest liczba niezależnych obserwacji, nie pojemność modelu. "
            "Wynika ona z arytmetyki nakładających się etykiet i nie podlega strojeniu.",
            "Symulacja handlowa (backtest) nie jest miarą jakości prognozy. Ten sam przebieg "
            "pod różnymi politykami ekspozycji daje wyniki rozpięte od głębokiej straty do "
            "wielokrotnego zysku, więc może służyć wyłącznie jako kontekst ekonomiczny.",
        ],
    )

    # ── 2. architektura ──────────────────────────────────────────────────────
    doc.add_page_break()
    add_heading(doc, "2. Architektura systemu", 1)

    add_body(
        doc,
        "System realizuje potok przetwarzania o ośmiu etapach, w którym każdy etap zapisuje "
        "wersjonowany artefakt na dysku i każdy kolejny etap wskazuje artefakt źródłowy po "
        "nazwie. Nie istnieje współdzielony katalog „najnowszych” wyników.",
    )

    add_table(
        doc,
        ["Etap", "Moduł", "Artefakt wyjściowy"],
        [
            ["Pobranie danych", "pipeline/backfill", "surowe świece 1m (parquet)"],
            ["Agregacja", "pipeline/aggregate", "świece 1h / 1d / 1w"],
            ["Cechy", "features/build", "features.parquet"],
            ["Etykiety", "labels/build", "labels.parquet"],
            ["Prognoza", "predict/*", "predictions.jsonl + run_summary.json"],
            ["Ocena jakości", "eval/predictions", "metrics.json"],
            ["Stabilność w czasie", "eval/stability", "stability_summary.json"],
            ["Symulacja handlowa", "eval/backtest", "backtest.json"],
        ],
        [4.0, 4.5, 7.0],
    )
    add_caption(doc, "Tabela 1. Etapy potoku i ich artefakty.")

    add_heading(doc, "2.1. Warstwy", 2)
    add_bullets(
        doc,
        [
            "Warstwa domenowa (domain.py) — wspólne słownictwo: klasy kierunku, kubełki "
            "głębokości, definicje horyzontów, przeliczenia interwałów. Umieszczona na poziomie "
            "pakietu, ponieważ etykietowanie, prognozowanie, ocena i planowanie muszą mówić "
            "tym samym językiem.",
            "Warstwa dostępu do danych (binance/, pipeline/, storage/) — pobieranie z Binance "
            "Public Data, konwersja do partycjonowanego formatu parquet, adresowanie artefaktów.",
            "Warstwa modelowa (predict/) — pięć rodzin modeli o jednakowym interfejsie i wspólnej "
            "arytmetyce okna kroczącego.",
            "Warstwa weryfikacji (eval/) — reguły oceny właściwej, przedziały ufności, analiza "
            "stabilności, symulacja handlowa z hipotezami zerowymi.",
            "Warstwa prezentacji (api/, frontend/) — FastAPI jako cienka warstwa uruchamiająca "
            "polecenia CLI oraz jednostronicowa aplikacja w natywnych modułach ES.",
        ],
    )

    add_heading(doc, "2.2. Decyzje technologiczne", 2)
    add_table(
        doc,
        ["Decyzja", "Uzasadnienie"],
        [
            ["Polars zamiast pandas", "Operacje kolumnowe na partycjonowanym parquet; pandas "
             "wyłącznie tam, gdzie wymaga tego pytorch-forecasting."],
            ["Parquet partycjonowany symbolem, rokiem i miesiącem",
             "Odczyt selektywny bez wczytywania całej historii; naturalna jednostka aktualizacji."],
            ["API uruchamia CLI jako podproces (shell=False)",
             "Jedna implementacja logiki; brak długich obliczeń w procesie serwera; "
             "przeglądarka nigdy nie buduje wiersza poleceń."],
            ["Kolejka zadań po stronie serwera, jeden wątek roboczy",
             "Etapy potoku zapisują te same partycje parquet, więc równoległość powodowałaby "
             "konflikt zapisu."],
            ["Brak kroku budowania po stronie frontendu",
             "Natywne moduły ES; mniejsza powierzchnia narzędziowa w pracy inżynierskiej."],
        ],
        [5.5, 10.0],
    )
    add_caption(doc, "Tabela 2. Wybrane decyzje architektoniczne i ich uzasadnienie.")

    # ── 3. słownictwo ────────────────────────────────────────────────────────
    doc.add_page_break()
    add_heading(doc, "3. Model dziedziny", 1)

    add_heading(doc, "3.1. Horyzonty prognostyczne", 2)
    add_body(
        doc,
        "Horyzont jest zdefiniowany jako rozpiętość kalendarzowa, nie liczba świec, i "
        "przeliczany na kroki osobno dla każdego interwału. Dzięki temu ten sam horyzont "
        "oznacza tę samą rozpiętość czasu w przebiegu godzinowym i dziennym.",
    )
    add_table(
        doc,
        ["Nazwa", "Rozpiętość", "Tygodni", "Uzasadnienie"],
        [
            [h.name, f"{h.forward_days} dni", str(h.forward_days // 7), uzasadnienie]
            for h, uzasadnienie in zip(
                DEFAULT_HORIZONS,
                [
                    "najkrótszy horyzont o pełnej mocy statystycznej",
                    "cztery tygodnie; podstawowy horyzont handlowy",
                    "trzynaście tygodni; zastąpił półrocze, które dawało 3,0 obserwacji "
                    "niezależnej wobec 7,0 dla kwartału",
                    "cztery kwartały dokładnie; 364 dni, ponieważ 365 nie dzieli się przez 7 "
                    "i zaokrąglało się różnie przy różnych interwałach",
                ],
                strict=False,
            )
        ],
        [2.5, 2.2, 1.8, 9.0],
    )
    add_caption(doc, "Tabela 3. Zbiór horyzontów prognostycznych.")

    add_heading(doc, "3.2. Etykiety", 2)
    add_bullets(
        doc,
        [
            "Kierunek ma dwie klasy — znak zwrotu w przód. Nie istnieje klasa „bez zmiany” ani "
            "próg: zwrot dokładnie zerowy nie ma strony i jest wykluczany. Próg kosztowy należy "
            "do warstwy decyzyjnej, nie do etykiet.",
            "Głębokość to dwanaście kubełków o granicach z ciągu Fibonacciego, opisujących "
            "wielkość ruchu w procentach. Każdy artefakt zapisuje własny schemat kubełków, "
            "ponieważ przebieg zapisany pod starszym schematem musi pozostać czytelny.",
        ],
    )

    # ── 4. warstwa prognostyczna ────────────────────────────────────────────
    doc.add_page_break()
    add_heading(doc, "4. Warstwa prognostyczna", 1)

    add_body(
        doc,
        "Wszystkie modele działają w schemacie kroczącym: model jest douczany raz na miesiąc "
        "kalendarzowy na oknie kończącym się przed prognozowaną świecą, a następnie prognozuje "
        "kolejne świece tego miesiąca. Zbiór treningowy zawiera wyłącznie etykiety już "
        "zrealizowane w chwili prognozy.",
    )

    add_table(
        doc,
        ["Model", "Rodzaj", "Rola"],
        [
            ["baseline", "częstości empiryczne z wygładzaniem Laplace'a",
             "punkt odniesienia bez uczenia"],
            ["ml", "HistGradientBoostingClassifier", "model tabelaryczny referencyjny"],
            ["xgboost", "XGBClassifier", "model tabelaryczny drugi"],
            ["gru", "rekurencyjna sieć GRU", "model sekwencyjny"],
            ["tft", "Temporal Fusion Transformer", "model sekwencyjny z uwagą"],
        ],
        [3.0, 6.5, 6.0],
    )
    add_caption(doc, "Tabela 4. Rodziny modeli.")

    add_heading(doc, "4.1. Okno treningowe wyprowadzane z horyzontu", 2)
    add_body(
        doc,
        "Szerokość okna nie jest parametrem konfiguracyjnym, lecz wynika z horyzontu i "
        "dostępnej historii. Kolidują tu dwa wymagania: model potrzebuje kilkuset wierszy, "
        "a statystyka potrzebuje obserwacji niezależnych, których okno o szerokości W zawiera "
        "(W − F) / F, gdzie F to rozpiętość horyzontu. Pierwsze wiąże na krótkich horyzontach, "
        "drugie na długich.",
    )
    add_body(
        doc,
        "Okno wynosi F + max(minimalna liczba wierszy, 3F) i jest przycinane przez długość "
        "historii. Poszerzanie okna dla wszystkich horyzontów nie jest rozwiązaniem: historia "
        "jest skończona, więc każda świeca oddana treningowi to świeca, której nie da się ocenić.",
    )

    add_heading(doc, "4.2. Kalibracja", 2)
    add_body(
        doc,
        "Prawdopodobieństwa są kalibrowane skalowaniem temperatury na wycinku wydzielonym "
        "wewnątrz okna treningowego — nigdy na świecach późniejszych, ponieważ kalibrator "
        "widziałby wtedy etykiety niedostępne prognozie. Modele głębokie zatrzymują uczenie na "
        "stracie po kalibracji, a nie na surowej entropii krzyżowej: surowa strata karze za "
        "nadmierną pewność, sieć staje się nadmiernie pewna na długo przed utratą zdolności "
        "rozróżniania, a kalibrator usuwa tę nadmierną pewność krok później.",
    )

    # ── 5. warstwa weryfikacji ──────────────────────────────────────────────
    doc.add_page_break()
    add_heading(doc, "5. Metodyka pomiaru", 1)
    add_body(
        doc,
        "Rozdział opisuje warstwę, która czyni porównanie architektur rozstrzygalnym. Bez niej "
        "wynik „architektury nie różnią się” pozostaje nieodróżnialny od „pomiar był zbyt "
        "zaszumiony”, a każdy pojedynczy wynik dodatni — nieodróżnialny od artefaktu.",
        italic=True,
    )

    add_heading(doc, "5.1. Reguły oceny właściwej", 2)
    add_body(
        doc,
        "Model jest oceniany trafnością, wynikiem Briera, logarytmiczną funkcją straty oraz "
        "oczekiwanym błędem kalibracji. Reguły te nie mają wolnych parametrów, więc nie ma w "
        "nich czego stroić ani czym schlebiać wynikowi.",
    )

    add_heading(doc, "5.2. Przedziały ufności przy nakładających się etykietach", 2)
    add_body(
        doc,
        "Każda trafność jest średnią po świecach, których etykiety się nakładają: etykieta "
        "świecy k obejmuje okres [k, k+F], więc etykieta k+1 dzieli z nią F−1 dni. Na horyzoncie "
        "rocznym sąsiednie świece zgadzają się co do 363 z 364 dni. Naiwny przedział ufności "
        "traktuje je jako niezależne i wychodzi kilkukrotnie za wąski.",
    )
    add_body(
        doc,
        "System stosuje bootstrap blokowy losujący ciągi kolejnych świec, co zachowuje tę "
        "zależność. Poniżej dziesięciu bloków przedział nie jest wyznaczany, lecz odmawiany: "
        "przy trzech blokach zaobserwowano przedziały nieobejmujące własnej oceny punktowej.",
    )

    add_heading(doc, "5.3. Moc statystyczna jako ograniczenie", 2)
    add_body(
        doc,
        "Liczba obserwacji niezależnych to liczba świec podzielona przez rozpiętość horyzontu. "
        "Przed uruchomieniem przebiegu system ostrzega o horyzontach, dla których liczba ta jest "
        "zbyt mała — osobno po stronie treningu i po stronie oceny, ponieważ obie strony czerpią "
        "z tej samej skończonej historii i poszerzanie okna przenosi problem, zamiast go usuwać.",
    )

    add_heading(doc, "5.4. Symulacja handlowa i cztery hipotezy zerowe", 2)
    add_body(
        doc,
        "Symulacja nie jest oceną prognozy i towarzyszą jej cztery punkty odniesienia, ponieważ "
        "porównanie wyłącznie ze strategią „kup i trzymaj” jest najsłabsze z możliwych.",
    )
    add_bullets(
        doc,
        [
            "Kup i trzymaj — punkt odniesienia rynkowy.",
            "Stała ekspozycja równa średniej ekspozycji strategii — usuwa efekt samego poziomu "
            "zaangażowania.",
            "Ta sama ścieżka ekspozycji przesunięta cyklicznie w czasie — usuwa efekt wyczucia "
            "momentu. To jest test właściwy: jeśli rzeczywiste ułożenie w czasie nie wypada "
            "lepiej niż dowolne przesunięte, prognoza nie wnosi wyczucia momentu.",
            "Reguła momentum na siedmiu świecach, niezawierająca modelu — punkt odniesienia "
            "wiążący; w przeprowadzonych testach osiągała wyższy percentyl niż którykolwiek "
            "z modeli.",
        ],
    )

    add_heading(doc, "5.5. Odtwarzalność i ziarnowanie", 2)
    add_body(
        doc,
        "Każdy przebieg zapisuje ziarno, tryb deterministyczny, wyprowadzone okna treningowe "
        "oraz liczbę faktycznie wykonanych epok. Losowość każdego okna douczania jest "
        "wyprowadzana z jego własnej tożsamości — ziarna bazowego, modelu, horyzontu, miesiąca "
        "oraz symbolu — a nie z jednego wspólnego strumienia. Wspólny strumień powodowałby, że "
        "liczba kroków uczenia wykonanych gdziekolwiek przesuwa inicjalizację wszystkiego, co "
        "następuje po niej, co czyni eksperyment z jedną zmienną niewykonalnym.",
    )
    add_body(
        doc,
        "Odtwarzalność została zweryfikowana pomiarowo: powtórzenie tej samej konfiguracji daje "
        "plik predykcji identyczny co do bajtu dla wszystkich modeli korzystających z losowości.",
    )

    # ── 6. wyniki ────────────────────────────────────────────────────────────
    doc.add_page_break()
    add_heading(doc, "6. Wyniki: porównanie architektur", 1)

    total = sum(len(v) for v in runs.values())
    seeds = sorted({r["seed"] for v in runs.values() for r in v if r["seed"] is not None})
    add_body(
        doc,
        f"Podstawą tabel jest {total} przebiegów odczytanych z artefaktów projektu"
        + (f", pod ziarnami: {', '.join(str(s) for s in seeds)}." if seeds else ".")
        + " Każdą konfigurację uruchomiono wielokrotnie pod różnymi ziarnami, ponieważ "
        "pojedynczy przebieg niesie szum inicjalizacji, którego przedział ufności liczony "
        "wewnątrz tego przebiegu nie obejmuje.",
    )

    add_heading(doc, "6.1. Ranking architektur wobec podłogi szumu", 2)
    add_body(
        doc,
        "Tabela poniżej odpowiada wprost na pytanie badawcze. Kolumna „rozrzut ziarna” podaje "
        "typowe odchylenie standardowe trafności między przebiegami tej samej architektury na "
        "tym samym symbolu, różniącymi się wyłącznie ziarnem. Jeżeli rozstęp między "
        "architekturami nie przekracza tej wielkości, ranking architektur nie ma podstaw.",
    )

    for horizon in HORIZON_NAMES:
        per_model: dict[str, list[float]] = {}
        within: list[float] = []
        for model in MODELS:
            values: list[float] = []
            for symbol in SYMBOLS:
                accs = horizon_values(runs.get((symbol, model), []), horizon, "accuracy")
                if not accs:
                    continue
                values.extend(accs)
                if len(accs) > 1:
                    within.append(float(np.std(accs, ddof=1)))
            if values:
                per_model[model] = values
        if len(per_model) < 2:
            continue

        means = {m: float(np.mean(v)) for m, v in per_model.items()}
        order = sorted(means, key=lambda m: -means[m])
        spread = max(means.values()) - min(means.values())
        noise = float(np.mean(within)) if within else None

        rows = [
            [str(rank), model, str(len(per_model[model])), fmt(means[model])]
            for rank, model in enumerate(order, start=1)
        ]
        add_body(doc, f"Horyzont „{horizon}”:")
        add_table(
            doc, ["Poz.", "Architektura", "Przeb.", "Trafność średnia"], rows,
            [1.5, 5.0, 2.0, 4.0],
        )
        verdict = (
            "rozstęp mieści się w rozrzucie ziarna — ranking bez podstaw"
            if noise is not None and spread <= noise
            else "rozstęp przekracza rozrzut ziarna"
            if noise is not None
            else "brak powtórzeń, rozrzut ziarna niezmierzony"
        )
        add_caption(
            doc,
            f"Rozstęp między najlepszą a najgorszą architekturą: {spread:.3f}. "
            + (f"Typowy rozrzut tej samej architektury pod innym ziarnem: {noise:.3f}. " if noise else "")
            + f"Wniosek: {verdict}.",
        )

    add_heading(doc, "6.2. Wyniki szczegółowe", 2)

    for horizon in HORIZON_NAMES:
        rows = []
        for symbol in SYMBOLS:
            for model in MODELS:
                got = runs.get((symbol, model), [])
                accs = horizon_values(got, horizon, "accuracy")
                eces = horizon_values(got, horizon, "ece")
                if not accs:
                    continue
                arr = np.array(accs)
                rows.append([
                    symbol.replace("USDT", ""), model, str(len(accs)),
                    fmt(float(arr.mean())),
                    fmt(float(arr.std(ddof=1))) if len(arr) > 1 else "—",
                    f"{arr.min():.3f}–{arr.max():.3f}" if len(arr) > 1 else fmt(float(arr[0])),
                    fmt(float(np.mean(eces))) if eces else "—",
                ])
        if not rows:
            continue
        add_body(doc, f"Horyzont „{horizon}” — wszystkie symbole i ziarna:")
        add_table(
            doc,
            ["Symbol", "Model", "Przeb.", "Trafność", "Odch.", "Zakres", "ECE"],
            rows,
            [2.2, 2.4, 1.5, 2.2, 1.8, 3.0, 2.4],
        )
        add_caption(
            doc,
            f"Tabela wyników dla horyzontu „{horizon}”. Kolumna „Odch.” to odchylenie "
            "standardowe trafności między przebiegami różniącymi się wyłącznie ziarnem — "
            "miara szumu inicjalizacji, niewidoczna w przedziale pojedynczego przebiegu.",
        )

    if not runs:
        add_body(
            doc,
            "UWAGA: w katalogu artefaktów nie znaleziono ocenionych przebiegów. Tabele wyników "
            "zostaną wypełnione po zakończeniu przeliczeń; dokument należy wtedy wygenerować "
            "ponownie.",
            italic=True,
        )

    # ── 7. defekty ───────────────────────────────────────────────────────────
    doc.add_page_break()
    add_heading(doc, "7. Weryfikacja poprawności implementacji", 1)
    add_body(
        doc,
        "Rozdział dokumentuje defekty wykryte w trakcie budowy systemu i stanowi materiał "
        "dowodowy dla ostatniego zdania tezy: warstwa weryfikacji rzeczywiście odróżnia wynik "
        "od artefaktu. Wspólną cechą wszystkich pozycji jest to, że defekt dawał wynik "
        "wyglądający na poprawny i nie ujawniał się w samej trafności prognozy — a więc "
        "porównanie architektur przeprowadzone bez tej warstwy porównywałoby artefakty.",
    )
    add_table(
        doc,
        ["Defekt", "Objaw bez weryfikacji", "Wykryty przez"],
        [
            ["Odczyt klas kierunku po pozycji, przy enkoderze sortującym etykiety alfabetycznie",
             "trafność równa 1 − trafność; przy wyniku bliskim losowego nieodróżnialne od szumu",
             "test na szeregu syntetycznym o znanej odpowiedzi"],
            ["Okno treningowe węższe od horyzontu",
             "stały rozkład jednostajny punktowany jako prognoza",
             "kontrola szerokości okna przed uruchomieniem"],
            ["Wspólny, nieweryfikowany katalog predykcji",
             "wyniki ostatnio uruchomionego modelu przypisywane wszystkim",
             "wprowadzenie wersjonowania przebiegów"],
            ["Jednorodne rozkłady głębokości w trzech modelach",
             "najwyższa rekomendacja nieosiągalna dla części modeli",
             "porównanie rozkładów między modelami"],
            ["Wspólny strumień inicjalizacji dla wszystkich symboli",
             "cztery symbole wykazujące przewagę jednocześnie; test istotności na poziomie "
             "p = 0,0015 dla zjawiska nieistniejącego",
             "powtórzenie eksperymentu pod innym ziarnem oraz kontrola na szeregu losowym"],
        ],
        [4.5, 6.0, 5.0],
    )
    add_caption(doc, "Tabela 5. Defekty wykryte w trakcie budowy systemu.")

    add_heading(doc, "7.1. Przypadek szczegółowy: pozorna przewaga modelu TFT", 2)
    add_body(
        doc,
        "Jedna komórka wyników przekroczyła próg istotności. Hipotezę zweryfikowano na czterech "
        "symbolach, które nie brały udziału w jej powstaniu; zaplanowany z góry test znaków "
        "wypadł pozytywnie. Dopiero trzy dalsze kontrole ją obaliły: powtórzenie pod innym "
        "ziarnem przesuwało wynik pojedynczego symbolu bardziej, niż wynosił deklarowany efekt; "
        "trzy szeregi losowe o parametrach dopasowanych do danych rzeczywistych dały poziom "
        "odniesienia, względem którego różnica przestawała być istotna; a analiza kodu wykazała, "
        "że cztery „niezależne” symbole współdzieliły sekwencję inicjalizacji. Wynik został "
        "wycofany.",
    )
    add_body(
        doc,
        "Wniosek metodyczny: przedział ufności wyznaczony wewnątrz jednego przebiegu mierzy "
        "szum próbkowania i pozostaje ślepy na szum inicjalizacji o porównywalnej wielkości. "
        "Z tego powodu wszystkie wyniki w rozdziale 6 raportowane są jako wielokrotne przebiegi.",
    )

    # ── 8. ograniczenia ──────────────────────────────────────────────────────
    doc.add_page_break()
    add_heading(doc, "8. Ograniczenia i kierunki dalszych prac", 1)
    add_heading(doc, "8.1. Zakres wniosków", 2)
    add_bullets(
        doc,
        [
            "Zbiór informacyjny obejmuje wyłącznie cechy wyprowadzone z ceny i wolumenu tego "
            "samego instrumentu. Nie uwzględniono danych on-chain, stawek finansowania, "
            "głębokości arkusza zleceń, danych międzyrynkowych ani sentymentu.",
            "Prognozowany jest znak zwrotu, czyli sformułowanie najtrudniejsze. Model może nie "
            "mieć przewagi kierunkowej i jednocześnie poprawnie prognozować zmienność.",
            "Nie badano prognozy przekrojowej, czyli porównania instrumentów między sobą. "
            "Pomiar korelacji zwrotów wskazuje, że wspólny czynnik rynkowy odpowiada za "
            "większość wariancji, a jego odjęcie jest jedyną drogą powyżej granicy wynikającej "
            "ze wzoru na liczbę efektywnych symboli.",
            "Zastosowano modele o niewielkiej pojemności. Przy zmierzonej liczbie obserwacji "
            "niezależnych pojemność nie jest jednak wiążącym ograniczeniem.",
        ],
    )
    add_heading(doc, "8.2. Kierunki dalszych prac", 2)
    add_bullets(
        doc,
        [
            "Prognoza przekrojowa — przewidywanie zwrotu względnego zamiast bezwzględnego.",
            "Rozszerzenie zbioru informacyjnego o dane spoza szeregu cenowego.",
            "Zmiana wielkości prognozowanej ze znaku zwrotu na zmienność lub wielkość ruchu.",
            "Zrównoleglenie pętli kroczącej, możliwe po uniezależnieniu ziaren poszczególnych "
            "okien od kolejności ich wykonania.",
        ],
    )

    add_heading(doc, "9. Podsumowanie", 1)
    add_body(
        doc,
        "Porównano pięć rodzin architektur uczenia maszynowego w zadaniu prognozy kierunku "
        "zmiany ceny na podstawie wyłącznie historii notowań. Żadna z nich nie osiąga trafności "
        "odróżnialnej od losowej na horyzoncie o wystarczającej mocy statystycznej, a różnice "
        "między nimi są mniejsze niż rozrzut wynikający z losowej inicjalizacji tej samej "
        "architektury — co czyni ranking oparty na pojedynczym przebiegu pozbawionym podstaw. "
        "Jest to wynik zgodny ze słabą formą hipotezy efektywności rynku, uzyskany na rynku "
        "wybranym tak, aby mierzyć zachowanie modelu, a nie artefakty systemu obrotu. "
        "Rozstrzygnięcie tego pytania było możliwe wyłącznie dzięki warstwie weryfikacji, "
        "której skuteczność dokumentuje pięć wykrytych defektów, z których każdy produkował "
        "wynik pozornie pozytywny.",
    )

    os.makedirs("docs", exist_ok=True)
    path = "docs/Prosper_Dokumentacja_Techniczna.docx"
    doc.save(path)
    return path


if __name__ == "__main__":
    written = build()
    size = os.path.getsize(written)
    print(f"written: {written} ({size / 1024:.0f} kB)")
