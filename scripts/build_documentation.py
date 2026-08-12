"""Generate the technical documentation as a .docx.

The document is *generated*, not written by hand, for the reason the previous one
was deleted: a hand-written write-up drifts from the code and then contradicts
it. Every catalogue — features, CLI flags, HTTP routes, configuration, the test
inventory — is read from the running system, and every result table is read from
`data/reports/` at build time. The document therefore cannot state a number the
artifacts do not contain, nor list a flag the CLI does not have.

    .venv/Scripts/python.exe scripts/build_documentation.py

Writes `docs/Prosper_Dokumentacja_Techniczna.docx`, which is gitignored: this
script is the source of truth, the document is an artifact.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict
from datetime import date

import numpy as np
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from documentation_sources import (  # noqa: E402
    api_reference,
    cli_reference,
    dataset_summary,
    feature_catalogue,
    settings_reference,
    test_inventory,
)
from prosper.domain import (  # noqa: E402
    DEFAULT_DEPTH_BINS_STR,
    DEFAULT_HORIZONS,
    HORIZON_NAMES,
    parse_depth_bins,
)

REPORT_ROOT = "data/reports"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
MODELS = ("baseline", "ml", "xgboost", "gru", "tft")
ACCENT = RGBColor(0x1F, 0x3B, 0x63)
MUTED = RGBColor(0x55, 0x5F, 0x6B)

_counters: dict[str, int] = defaultdict(int)


# ── results read from the artifacts ──────────────────────────────────────────

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
        seed = None
        summary = f"{REPORT_ROOT}/predictions/{symbol}/{slug}/run_summary.json"
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


def add_body(doc: Document, text: str, *, italic: bool = False, size: float = 10) -> None:
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(6)
    paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    run = paragraph.add_run(text)
    run.italic = italic
    run.font.size = Pt(size)


def add_quote(doc: Document, text: str) -> None:
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.left_indent = Cm(1.0)
    paragraph.paragraph_format.space_after = Pt(10)
    paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    run = paragraph.add_run(text)
    run.bold = True
    run.font.size = Pt(10.5)


def add_bullets(doc: Document, items: list[str]) -> None:
    for item in items:
        paragraph = doc.add_paragraph(item, style="List Bullet")
        paragraph.paragraph_format.space_after = Pt(3)
        for run in paragraph.runs:
            run.font.size = Pt(10)


def add_code(doc: Document, lines: list[str]) -> None:
    for line in lines:
        paragraph = doc.add_paragraph()
        paragraph.paragraph_format.left_indent = Cm(0.8)
        paragraph.paragraph_format.space_after = Pt(0)
        run = paragraph.add_run(line)
        run.font.name = "Consolas"
        run.font.size = Pt(8.5)
    doc.add_paragraph()


def add_table(
    doc: Document, header: list[str], rows: list[list[str]], widths: list[float],
    *, font: float = 8.5,
) -> None:
    """A table with explicit column widths, set on every cell.

    Word ignores a width set only on the column, so it has to go on each cell or
    the layout collapses to equal columns. The header row repeats on every page,
    which matters here: several of these tables run for pages.
    """
    table = doc.add_table(rows=1, cols=len(header))
    table.style = "Light Grid Accent 1"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    for index, label in enumerate(header):
        cell = table.rows[0].cells[index]
        cell.text = ""
        run = cell.paragraphs[0].add_run(label)
        run.bold = True
        run.font.size = Pt(font)

    header_properties = table.rows[0]._tr.get_or_add_trPr()
    repeat = OxmlElement("w:tblHeader")
    repeat.set(qn("w:val"), "true")
    header_properties.append(repeat)

    for values in rows:
        cells = table.add_row().cells
        for index, value in enumerate(values):
            cells[index].text = ""
            paragraph = cells[index].paragraphs[0]
            paragraph.paragraph_format.space_after = Pt(0)
            run = paragraph.add_run(str(value))
            run.font.size = Pt(font)
    for row in table.rows:
        for index, cell in enumerate(row.cells):
            cell.width = Cm(widths[index])
    doc.add_paragraph()


def add_caption(doc: Document, kind: str, text: str) -> None:
    _counters[kind] += 1
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(12)
    run = paragraph.add_run(f"{kind} {_counters[kind]}. {text}")
    run.italic = True
    run.font.size = Pt(8.5)
    run.font.color.rgb = MUTED


def add_toc(doc: Document) -> None:
    """A field Word fills in on open; python-docx cannot compute page numbers."""
    paragraph = doc.add_paragraph()
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instruction = OxmlElement("w:instrText")
    instruction.set(qn("xml:space"), "preserve")
    instruction.text = r'TOC \o "1-3" \h \z \u'
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    placeholder = OxmlElement("w:t")
    placeholder.text = "Spis treści zostanie zbudowany po otwarciu dokumentu (F9)."
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    for element in (begin, instruction, separate, placeholder, end):
        run._r.append(element)


def page_break(doc: Document) -> None:
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


# ═════════════════════════════════════════════════════════════════════════════
def build() -> str:
    runs = load_runs()
    doc = Document()

    section = doc.sections[0]
    section.page_width, section.page_height = Cm(21.0), Cm(29.7)
    section.left_margin = section.right_margin = Cm(2.3)
    section.top_margin = section.bottom_margin = Cm(2.0)

    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(10)

    _title_page(doc)
    page_break(doc)

    add_heading(doc, "Spis treści", 1)
    add_toc(doc)
    page_break(doc)

    _chapter_1_introduction(doc)
    _chapter_2_theory(doc)
    _chapter_3_architecture(doc)
    _chapter_4_data(doc)
    _chapter_5_features(doc)
    _chapter_6_labels(doc)
    _chapter_7_models(doc)
    _chapter_8_methodology(doc)
    _chapter_9_results(doc, runs)
    _chapter_10_verification(doc)
    _chapter_11_interfaces(doc)
    _chapter_11a_worked_example(doc)
    _chapter_12_limits(doc)
    _chapter_13_bibliography(doc)
    _appendix_tests(doc)
    _appendix_cli(doc)
    _appendix_api(doc)
    _appendix_settings(doc)
    _appendix_glossary(doc)

    os.makedirs("docs", exist_ok=True)
    path = "docs/Prosper_Dokumentacja_Techniczna.docx"
    doc.save(path)
    return path


# ── front matter ─────────────────────────────────────────────────────────────

def _title_page(doc: Document) -> None:
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_before = Pt(150)
    run = title.add_run("Prosper")
    run.bold = True
    run.font.size = Pt(40)
    run.font.color.rgb = ACCENT

    for text, size, italic in (
        ("Platforma badawcza do analizy rynku kryptowalut", 15, False),
        ("Dokumentacja techniczna", 12, True),
    ):
        paragraph = doc.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        added = paragraph.add_run(text)
        added.font.size = Pt(size)
        added.italic = italic

    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(80)

    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = meta.add_run(
        "Praca inżynierska\n"
        "Polsko-Japońska Akademia Technik Komputerowych\n\n"
        f"Dokument wygenerowany automatycznie z artefaktów projektu\n"
        f"dnia {date.today().isoformat()} poleceniem "
        f"scripts/build_documentation.py"
    )
    run.font.size = Pt(9)
    run.font.color.rgb = MUTED


# ── 1 ────────────────────────────────────────────────────────────────────────

def _chapter_1_introduction(doc: Document) -> None:
    add_heading(doc, "1. Wprowadzenie", 1)

    add_heading(doc, "1.1. Kontekst", 2)
    add_body(
        doc,
        "Rynek kryptowalut jest młody, płynny i całodobowy. Dane historyczne o notowaniach są "
        "publicznie dostępne w jednolitym formacie i o pełnej głębokości, co czyni go wygodnym "
        "poligonem dla metod uczenia maszynowego — i zarazem miejscem, w którym łatwo o wynik "
        "pozornie dobry, ponieważ szereg cenowy jest niemal całkowicie szumem, a każdy błąd "
        "metodologiczny objawia się jako poprawa trafności, nie jako awaria.",
    )
    add_body(
        doc,
        "Praca opisuje system zbudowany po to, aby tę różnicę rozstrzygać: platformę realizującą "
        "pełny potok badawczy od pobrania surowych świec, przez konstrukcję cech i etykiet, "
        "trening pięciu rodzin modeli, aż po ocenę jakości prognozy i symulację handlową — z "
        "warstwą weryfikacji, której zadaniem jest odróżnić wynik od artefaktu.",
    )

    add_heading(doc, "1.2. Pytanie badawcze", 2)
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

    add_heading(doc, "1.3. Teza", 2)
    add_quote(
        doc,
        "Różnice między badanymi architekturami uczenia maszynowego w zadaniu prognozy kierunku "
        "ceny z samej historii notowań są mniejsze niż rozrzut wyników wynikający z losowej "
        "inicjalizacji tej samej architektury, a żadna z nich nie osiąga przewagi odróżnialnej "
        "od losowej. Orzeczenie tego wymaga warstwy weryfikacji zdolnej odróżnić brak różnicy od "
        "niedostatecznej dokładności pomiaru.",
    )
    add_body(
        doc,
        "Zdanie ostatnie jest istotne. Stwierdzenie „architektury nie różnią się między sobą” "
        "jest nieodróżnialne od stwierdzenia „pomiar był zbyt zaszumiony, aby je rozróżnić”, "
        "dopóki poziom szumu nie zostanie zmierzony. Z tego powodu każdą konfigurację uruchomiono "
        "wielokrotnie pod różnymi ziarnami generatora liczb losowych, a rozdział 9 raportuje "
        "rozstęp między architekturami obok rozrzutu wynikającego z samej inicjalizacji.",
    )

    add_heading(doc, "1.4. Uzasadnienie wyboru rynku", 2)
    add_body(
        doc,
        "Pierwotnym zamiarem był system uniwersalny, obejmujący rynek akcji, walutowy i "
        "kryptowalut. Zakres ograniczono do kryptowalut z powodów metodologicznych, a nie "
        "wyłącznie z powodu złożoności implementacji: rynki różnią się w sposób, który wprowadza "
        "do pomiaru zmienne niezwiązane z badanym modelem.",
    )
    add_bullets(
        doc,
        [
            "Notowania ciągłe — brak luk weekendowych i nocnych, które na rynku akcji tworzą "
            "nieciągłości wymagające osobnego traktowania w każdej z cech technicznych, a w "
            "modelach sekwencyjnych zaburzają pojęcie sąsiedztwa w czasie.",
            "Brak zdarzeń korporacyjnych — podziały akcji, dywidendy, emisje i wykupy wymagają "
            "korekty szeregu historycznego, a przyjęta metoda korekty sama wpływa na wynik "
            "modelu i staje się nieudokumentowanym parametrem badania.",
            "Jednolita mikrostruktura — jedna giełda, jeden mechanizm kojarzenia zleceń, brak "
            "fragmentacji obrotu między systemami i brak aukcji otwarcia oraz zamknięcia.",
            "Otwarte dane historyczne w jednolitym formacie, o pełnej głębokości i bez opłat "
            "licencyjnych, co czyni wyniki odtwarzalnymi przez osobę trzecią bez dostępu do "
            "płatnych źródeł.",
            "Brak ograniczeń krótkiej sprzedaży i jednolity mechanizm rozliczenia, dzięki czemu "
            "symulacja handlowa nie musi modelować asymetrii instytucjonalnych.",
        ],
    )
    add_body(
        doc,
        "Wybrano zatem rynek, na którym mierzone jest zachowanie modelu, a nie artefakty "
        "specyficzne dla danego systemu obrotu. Rozszerzenie na pozostałe rynki pozostaje "
        "kierunkiem dalszych prac (rozdział 12).",
    )

    add_heading(doc, "1.5. Twierdzenia szczegółowe", 2)
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
            "raportuje własną niepewność, co stanowi osobny, pozytywny wynik inżynierski.",
            "Wiążącym ograniczeniem jest liczba niezależnych obserwacji, nie pojemność modelu. "
            "Wynika ona z arytmetyki nakładających się etykiet i nie podlega strojeniu.",
            "Symulacja handlowa nie jest miarą jakości prognozy. Ten sam przebieg pod różnymi "
            "politykami ekspozycji daje wyniki rozpięte od głębokiej straty do wielokrotnego "
            "zysku, więc może służyć wyłącznie jako kontekst ekonomiczny.",
        ],
    )

    add_heading(doc, "1.6. Struktura dokumentu", 2)
    add_table(
        doc,
        ["Rozdział", "Zawartość"],
        [
            ["2", "Podstawy teoretyczne: efektywność rynku, reguły oceny właściwej, kalibracja, "
                  "bootstrap blokowy"],
            ["3", "Architektura systemu i decyzje technologiczne"],
            ["4", "Dane: źródło, format, agregacja, kontrola jakości"],
            ["5", "Cechy: zasada niezmienniczości względem skali i pełny katalog"],
            ["6", "Etykiety: kierunek, głębokość, horyzonty"],
            ["7", "Architektury modeli i wspólny protokół kroczący"],
            ["8", "Metodyka pomiaru"],
            ["9", "Wyniki: porównanie architektur"],
            ["10", "Weryfikacja poprawności implementacji — studia przypadków"],
            ["11", "Interfejsy: wiersz poleceń, HTTP, panel przeglądarkowy"],
            ["12", "Ograniczenia i kierunki dalszych prac"],
            ["A–D", "Załączniki: inwentarz testów, referencja CLI, referencja API, konfiguracja"],
        ],
        [2.5, 13.5],
    )
    add_caption(doc, "Tabela", "Układ dokumentu.")
    page_break(doc)


# ── 2 ────────────────────────────────────────────────────────────────────────

def _chapter_2_theory(doc: Document) -> None:
    add_heading(doc, "2. Podstawy teoretyczne", 1)

    add_heading(doc, "2.1. Hipoteza efektywności rynku", 2)
    add_body(
        doc,
        "Hipoteza efektywności rynku orzeka, że ceny odzwierciedlają dostępne informacje, przez "
        "co systematyczne osiąganie ponadprzeciętnych stóp zwrotu na podstawie tych informacji "
        "jest niemożliwe. Fama wyróżnił trzy formy, różniące się zakresem zbioru informacyjnego.",
    )
    add_table(
        doc,
        ["Forma", "Zbiór informacyjny", "Co wyklucza", "Związek z pracą"],
        [
            ["Słaba", "historyczne ceny i wolumeny",
             "zyskowność analizy technicznej",
             "przedmiot badania"],
            ["Półsilna", "wszystkie informacje publiczne",
             "zyskowność analizy fundamentalnej",
             "poza zakresem"],
            ["Silna", "informacje publiczne i niepubliczne",
             "zyskowność wykorzystania informacji poufnych",
             "poza zakresem"],
        ],
        [2.2, 4.5, 5.0, 4.3],
    )
    add_caption(doc, "Tabela", "Trzy formy hipotezy efektywności rynku wg Famy (1970).")
    add_body(
        doc,
        "Badanie ograniczone do historii notowań testuje formę słabą. Nie jest to wersja uboższa "
        "badania fundamentalnego, lecz inne pytanie: forma słaba pyta, czy w samym szeregu "
        "cenowym pozostaje struktura możliwa do wykorzystania. Odrzucenie tej hipotezy wymagałoby "
        "wykazania przewagi istotnie większej od losowej; jej nieodrzucenie — co jest wynikiem "
        "niniejszej pracy — nie dowodzi jej prawdziwości, lecz wyznacza górne ograniczenie na "
        "wielkość ewentualnej przewagi.",
    )

    add_heading(doc, "2.2. Własności finansowych szeregów czasowych", 2)
    add_bullets(
        doc,
        [
            "Zwroty są w przybliżeniu nieskorelowane w czasie, ale ich wartości bezwzględne już "
            "nie — zmienność skupia się w klastrach, co czyni ją prognozowalną wtedy, gdy sam "
            "kierunek nie jest.",
            "Rozkład zwrotów ma ciężkie ogony: zdarzenia oddalone o wiele odchyleń standardowych "
            "występują znacznie częściej, niż przewiduje rozkład normalny.",
            "Szereg jest niestacjonarny — poziom ceny, zmienność i zależności między cechami "
            "zmieniają się w czasie, co uzasadnia uczenie w schemacie kroczącym zamiast "
            "jednorazowego podziału na zbiór uczący i testowy.",
            "Stosunek sygnału do szumu jest bardzo niski. Nawet prawdziwa przewaga rzędu jednego "
            "punktu procentowego trafności jest trudna do odróżnienia od zera przy realnie "
            "dostępnej liczbie obserwacji.",
        ],
    )

    add_heading(doc, "2.3. Prognoza kierunku jako zadanie klasyfikacji", 2)
    add_body(
        doc,
        "Prognoza jest sformułowana jako klasyfikacja binarna: znak zwrotu w przód na zadanym "
        "horyzoncie. Wyjściem modelu jest rozkład prawdopodobieństwa na dwóch klasach, a nie "
        "twarda decyzja — dzięki czemu ocena może korzystać z reguł właściwych, a warstwa "
        "decyzyjna może osobno rozstrzygać, czy przewaga pokrywa koszty transakcyjne.",
    )
    add_body(
        doc,
        "Nie wprowadzono klasy „bez zmiany”. Próg oddzielający ruch istotny od nieistotnego "
        "byłby dodatkowym, arbitralnym parametrem badania, a jego wartość wpływałaby na "
        "raportowaną trafność. Wielkość ruchu jest natomiast prognozowana osobno, jako rozkład "
        "na kubełkach głębokości (rozdział 6).",
    )

    add_heading(doc, "2.4. Reguły oceny właściwej", 2)
    add_body(
        doc,
        "Regułą właściwą nazywamy funkcję oceny prognozy probabilistycznej, która osiąga "
        "optimum wtedy i tylko wtedy, gdy prognoza podaje prawdziwe prawdopodobieństwo. Reguła "
        "niewłaściwa nagradza zniekształcanie prognozy — na przykład sama trafność nagradza "
        "deklarowanie skrajnej pewności.",
    )
    add_table(
        doc,
        ["Miara", "Definicja", "Zakres", "Interpretacja"],
        [
            ["Trafność", "udział poprawnie wskazanych kierunków", "0–1",
             "niewłaściwa, ale bezpośrednio interpretowalna"],
            ["Brier", "średni kwadrat różnicy prognozy i wyniku", "0–1",
             "właściwa; kwadratowa kara za pewność błędną"],
            ["Logarytmiczna", "średnia ujemnego logarytmu prawdopodobieństwa trafnej klasy",
             "0–∞", "właściwa; kara nieograniczona"],
            ["ECE", "średnia rozbieżność deklarowanej pewności i częstości trafień", "0–1",
             "mierzy kalibrację, nie zdolność rozróżniania"],
        ],
        [2.6, 6.2, 1.8, 5.4],
    )
    add_caption(doc, "Tabela", "Miary jakości prognozy stosowane w systemie.")
    add_body(
        doc,
        "Rozkład Briera na składnik kalibracji i składnik rozróżniania pokazuje, dlaczego obie "
        "wielkości trzeba raportować osobno: model może być doskonale skalibrowany i całkowicie "
        "pozbawiony zdolności rozróżniania — wystarczy, że zawsze podaje częstość bazową. "
        "Odwrotnie, model dobrze rozróżniający może być źle skalibrowany i wtedy jego "
        "prawdopodobieństwa nie nadają się do podejmowania decyzji, mimo że ranking jest trafny.",
    )

    add_heading(doc, "2.5. Kalibracja probabilistyczna", 2)
    add_body(
        doc,
        "Kalibracja to zgodność deklarowanej pewności z rzeczywistą częstością: spośród świec, "
        "którym model przypisał prawdopodobieństwo 0,6, mniej więcej sześćdziesiąt procent "
        "powinno zrealizować prognozowany kierunek. System stosuje skalowanie temperaturą — "
        "jednoparametrową transformację logitów, która zmienia ostrość rozkładu, nie zmieniając "
        "porządku klas.",
    )
    add_body(
        doc,
        "Zaletą jednego parametru jest odporność na przeuczenie: kalibrator dopasowany na "
        "kilkudziesięciu obserwacjach pozostaje sensowny, podczas gdy metody wieloparametrowe "
        "wymagają zbiorów o rząd wielkości większych. Ograniczeniem jest to, że skalowanie "
        "temperaturą nie naprawia błędu zależnego od cechy — jeśli model jest przesadnie pewny "
        "wyłącznie w reżimie wysokiej zmienności, jeden parametr tego nie rozdzieli.",
    )

    add_heading(doc, "2.6. Bootstrap blokowy", 2)
    add_body(
        doc,
        "Klasyczny bootstrap losuje obserwacje niezależnie, co zakłada ich wzajemną "
        "niezależność. W szeregu czasowym z nakładającymi się etykietami założenie to jest "
        "fałszywe: etykieta świecy k obejmuje okres [k, k+F], więc etykieta świecy k+1 dzieli z "
        "nią F−1 okresów. Na horyzoncie rocznym sąsiednie świece zgadzają się co do 363 z 364 "
        "dni.",
    )
    add_body(
        doc,
        "Bootstrap blokowy losuje ciągi kolejnych obserwacji o długości równej rozpiętości "
        "horyzontu, dzięki czemu zachowuje zależność wewnątrz bloku i szacuje wariancję, która "
        "nie jest zaniżona. Efektywna liczba obserwacji wynosi w przybliżeniu liczbę świec "
        "podzieloną przez rozpiętość horyzontu; poniżej dziesięciu bloków system odmawia podania "
        "przedziału zamiast podać przedział fałszywie wąski.",
    )
    page_break(doc)


# ── 3 ────────────────────────────────────────────────────────────────────────

def _chapter_3_architecture(doc: Document) -> None:
    add_heading(doc, "3. Architektura systemu", 1)

    add_heading(doc, "3.1. Potok przetwarzania", 2)
    add_body(
        doc,
        "System realizuje potok o ośmiu etapach. Każdy etap zapisuje wersjonowany artefakt na "
        "dysku, a każdy kolejny etap wskazuje artefakt źródłowy po nazwie. Nie istnieje "
        "współdzielony katalog „najnowszych” wyników — wcześniejsza wersja systemu taki katalog "
        "miała i skutkowało to przypisywaniem wyników ostatnio uruchomionego modelu wszystkim "
        "pozostałym (rozdział 10).",
    )
    add_table(
        doc,
        ["Etap", "Moduł", "Wejście", "Artefakt wyjściowy"],
        [
            ["Pobranie", "pipeline/backfill", "Binance Public Data (ZIP)",
             "świece 1m w formacie parquet"],
            ["Agregacja", "pipeline/aggregate", "świece 1m", "świece 1h, 1d, 1w"],
            ["Cechy", "features/build", "świece wybranego interwału", "features.parquet"],
            ["Etykiety", "labels/build", "świece wybranego interwału", "labels.parquet"],
            ["Prognoza", "predict/*", "features.parquet",
             "predictions.jsonl, run_summary.json"],
            ["Ocena", "eval/predictions", "predictions.jsonl + świece",
             "metrics.json, predictions_quality.parquet"],
            ["Stabilność", "eval/stability", "predictions.jsonl + świece",
             "stability_summary.json"],
            ["Symulacja", "eval/backtest", "predictions.jsonl + świece", "backtest.json"],
        ],
        [2.3, 3.4, 5.0, 5.6],
    )
    add_caption(doc, "Tabela", "Etapy potoku, ich wejścia i artefakty.")

    add_heading(doc, "3.2. Warstwy", 2)
    add_bullets(
        doc,
        [
            "Warstwa domenowa (domain.py) — wspólne słownictwo: klasy kierunku, kubełki "
            "głębokości, definicje horyzontów, przeliczenia interwałów, miary liczby obserwacji "
            "efektywnych. Umieszczona na poziomie pakietu, ponieważ etykietowanie, prognozowanie, "
            "ocena i planowanie muszą posługiwać się tym samym słownictwem; przypisanie jej do "
            "którejkolwiek z tych warstw czyniłoby pozostałe jej klientami.",
            "Warstwa dostępu do danych (binance/, pipeline/, storage/) — pobieranie z Binance "
            "Public Data, konwersja do partycjonowanego formatu parquet, adresowanie i "
            "wersjonowanie artefaktów.",
            "Warstwa modelowa (predict/) — pięć rodzin modeli o jednakowym interfejsie, wspólna "
            "arytmetyka okna kroczącego (window.py) oraz mechanizm treningu wielosymbolowego "
            "(pool.py).",
            "Warstwa weryfikacji (eval/) — reguły oceny właściwej, przedziały ufności, analiza "
            "stabilności w czasie, symulacja handlowa wraz z hipotezami zerowymi.",
            "Warstwa decyzyjna (planner/) — przekształcenie prognozy w rekomendację "
            "siedmiostopniową, z bramkami ryzyka i progiem kosztowym.",
            "Warstwa prezentacji (api/, frontend/) — FastAPI jako cienka warstwa uruchamiająca "
            "polecenia wiersza poleceń oraz jednostronicowa aplikacja w natywnych modułach ES.",
        ],
    )

    add_heading(doc, "3.3. Decyzje technologiczne", 2)
    add_table(
        doc,
        ["Decyzja", "Uzasadnienie", "Koszt"],
        [
            ["Polars zamiast pandas",
             "Operacje kolumnowe na partycjonowanym parquet; leniwa ewaluacja przy odczycie "
             "wielu partycji.",
             "pandas pozostaje w module TFT, gdzie wymaga go pytorch-forecasting."],
            ["Parquet partycjonowany symbolem, rokiem i miesiącem",
             "Odczyt selektywny bez wczytywania całej historii; miesiąc jest naturalną jednostką "
             "aktualizacji, zgodną z formatem publikacji danych źródłowych.",
             "Większa liczba plików; konieczność spójnego schematu między partycjami."],
            ["API uruchamia CLI jako podproces (shell=False)",
             "Jedna implementacja logiki dla obu interfejsów; brak długotrwałych obliczeń w "
             "procesie serwera; przeglądarka nigdy nie konstruuje wiersza poleceń.",
             "Narzut uruchomienia procesu; konieczność przekazywania parametrów przez argumenty."],
            ["Kolejka zadań po stronie serwera z jednym wątkiem roboczym",
             "Etapy potoku zapisują te same partycje parquet, więc równoległość powodowałaby "
             "konflikt zapisu; kolejka po stronie serwera przeżywa przeładowanie przeglądarki.",
             "Brak równoległości między zadaniami użytkownika."],
            ["Brak kroku budowania po stronie interfejsu",
             "Natywne moduły ES; mniejsza powierzchnia narzędziowa i brak zależności od "
             "ekosystemu pakietów JavaScript.",
             "Brak transpilacji ogranicza użycie najnowszej składni."],
            ["Typer jako warstwa wiersza poleceń",
             "Deklaratywne definicje poleceń z typowaniem; pomoc generowana z sygnatur, dzięki "
             "czemu referencja w załączniku B jest odczytem, a nie przepisaniem.",
             "Domyślne wartości obowiązują tylko przy nieobecności flagi, co wymaga ostrożności "
             "w warstwie API."],
        ],
        [3.6, 7.4, 5.0],
    )
    add_caption(doc, "Tabela", "Decyzje architektoniczne wraz z ich kosztem.")

    add_heading(doc, "3.4. Wersjonowanie artefaktów", 2)
    add_body(
        doc,
        "Każdy przebieg prognozy zapisywany jest w katalogu o nazwie złożonej z modelu, "
        "interwału i znacznika czasu. Wszystkie artefakty pochodne — ocena, stabilność, "
        "rekomendacje, symulacja — powstają w katalogach o tej samej nazwie, w osobnych "
        "drzewach. Dzięki temu każda liczba w dokumentacji daje się prześledzić wstecz do "
        "przebiegu, który ją wytworzył.",
    )
    add_code(
        doc,
        [
            "data/reports/",
            "  predictions/{symbol}/{model}_{interwał}_{znacznik}/predictions.jsonl",
            "                                                    /run_summary.json",
            "  evaluations/{symbol}/{model}_{interwał}_{znacznik}/metrics.json",
            "                                                    /predictions_quality.parquet",
            "  eval/{symbol}/{model}_{interwał}_{znacznik}/stability_summary.json",
            "  backtests/{symbol}/{model}_{interwał}_{znacznik}/backtest.json",
            "  recommendations/{symbol}/{model}_{interwał}_{znacznik}/...",
        ],
    )
    add_body(
        doc,
        "Usunięcie przebiegu usuwa wszystkie jego artefakty pochodne. W przeciwnym razie "
        "narzędzie przeglądające drzewo ocen raportowałoby wyniki przebiegu, którego predykcje "
        "już nie istnieją, obok wyników bieżących.",
    )
    page_break(doc)


# ── 4 ────────────────────────────────────────────────────────────────────────

def _chapter_4_data(doc: Document) -> None:
    add_heading(doc, "4. Dane", 1)

    add_heading(doc, "4.1. Źródło", 2)
    add_body(
        doc,
        "Dane pochodzą z serwisu Binance Public Data, udostępniającego historyczne świece w "
        "postaci miesięcznych archiwów ZIP z plikami CSV. Pobierany jest interwał "
        "jednominutowy, z którego pozostałe interwały powstają przez agregację — dzięki czemu "
        "świeca godzinowa i dzienna pochodzą z tego samego źródła i są ze sobą spójne.",
    )

    add_heading(doc, "4.2. Schemat świecy", 2)
    add_table(
        doc,
        ["Kolumna", "Typ", "Znaczenie"],
        [
            ["open_time", "znacznik czasu UTC", "początek okresu świecy; klucz porządkujący"],
            ["open, high, low, close", "liczba zmiennoprzecinkowa",
             "cena otwarcia, maksimum, minimum, zamknięcia"],
            ["volume", "liczba zmiennoprzecinkowa", "wolumen w walucie bazowej"],
            ["close_time", "znacznik czasu UTC", "koniec okresu świecy"],
            ["quote_asset_volume", "liczba zmiennoprzecinkowa", "wolumen w walucie kwotowanej"],
            ["num_trades", "liczba całkowita", "liczba transakcji w okresie"],
            ["taker_buy_base_volume", "liczba zmiennoprzecinkowa",
             "wolumen inicjowany przez stronę kupującą"],
            ["taker_buy_quote_volume", "liczba zmiennoprzecinkowa",
             "jak wyżej, w walucie kwotowanej"],
        ],
        [4.4, 3.6, 8.0],
    )
    add_caption(doc, "Tabela", "Schemat pojedynczej świecy w warstwie przetworzonej.")
    add_body(
        doc,
        "Cztery ostatnie kolumny opisują przepływ zleceń i pozwalają wyznaczyć udział wolumenu "
        "inicjowanego przez kupujących. Archiwa pobrane przed rozszerzeniem schematu zawierają "
        "wyłącznie cenę i wolumen; cechy przepływu powstają wtedy tylko tam, gdzie dane na to "
        "pozwalają, a ich brak jest raportowany, nie uzupełniany wartością zastępczą.",
    )

    add_heading(doc, "4.3. Agregacja", 2)
    add_body(
        doc,
        "Agregacja z interwału jednominutowego przebiega według reguł: otwarcie z pierwszej "
        "świecy okresu, zamknięcie z ostatniej, maksimum i minimum jako skrajne wartości, "
        "wolumeny i liczba transakcji jako sumy. Interwał tygodniowy partycjonowany jest numerem "
        "tygodnia zamiast miesiąca, ponieważ tydzień może przekraczać granicę miesiąca.",
    )

    add_heading(doc, "4.4. Kontrola jakości", 2)
    add_bullets(
        doc,
        [
            "Kompletność — wykrywanie brakujących okresów przez porównanie liczby świec z "
            "liczbą oczekiwaną dla danego zakresu i interwału.",
            "Duplikaty — świece o powtórzonym znaczniku czasu, powstające przy ponownym "
            "pobraniu tego samego miesiąca.",
            "Spójność wewnętrzna — sprawdzenie, że minimum nie przekracza maksimum oraz że "
            "otwarcie i zamknięcie mieszczą się w tym zakresie.",
            "Wartości nieujemne — wolumen i liczba transakcji nie mogą być ujemne.",
            "Ciągłość — wykrywanie skoków czasu większych niż jeden interwał.",
        ],
    )

    add_heading(doc, "4.5. Charakterystyka zbioru", 2)
    stats = dataset_summary("1d")
    if stats:
        add_table(
            doc,
            ["Symbol", "Pierwsza świeca", "Ostatnia świeca", "Liczba świec", "Lat"],
            [[s.symbol, s.first, s.last, f"{s.bars:,}".replace(",", " "), f"{s.years:.1f}"]
             for s in stats],
            [3.4, 3.4, 3.4, 3.2, 2.6],
        )
        add_caption(
            doc, "Tabela",
            "Zakres danych dziennych w jeziorze danych, zmierzony w chwili budowania dokumentu. "
            "Symbole o krótszej historii ograniczają horyzont roczny (rozdział 8.3).",
        )
    else:
        add_body(doc, "Jezioro danych jest puste — uruchom etap pobrania.", italic=True)
    page_break(doc)


# ── 5 ────────────────────────────────────────────────────────────────────────

def _chapter_5_features(doc: Document) -> None:
    add_heading(doc, "5. Cechy", 1)

    add_heading(doc, "5.1. Zasada: niezmienniczość względem skali", 2)
    add_body(
        doc,
        "Zbiór cech treningowych jest listą włączającą, nie wykluczającą. Wcześniejsza wersja "
        "przekazywała modelowi każdą kolumnę poza kilkoma nazwanymi wyjątkami, przez co do "
        "wejścia trafiały bezwzględne poziomy cen. Przy jednym instrumencie było to nieszkodliwe, "
        "ponieważ normalizacja usuwa poziom; przy uczeniu na kilku instrumentach naraz stawało "
        "się etykietą instrumentu — zmierzone różnice sięgały 553-krotności dla średniej "
        "kroczącej i 387-krotności dla histogramu MACD.",
    )
    add_body(
        doc,
        "Lista włączająca wymusza świadomą decyzję przy każdej nowej kolumnie. Lista wykluczająca "
        "przyjęłaby każdą nową cechę względną razem z jej bezwzględnym odpowiednikiem, co jest "
        "dokładnie tym błędem, którego należało uniknąć. Kolumny bezwzględne pozostają w pliku "
        "cech, ponieważ korzysta z nich panel przeglądarkowy, ale nie trafiają do modelu.",
    )

    add_heading(doc, "5.2. Katalog cech treningowych", 2)
    catalogue = feature_catalogue()
    by_group: dict[str, list] = defaultdict(list)
    for feature in catalogue:
        by_group[feature.group].append(feature)
    add_body(
        doc,
        f"Model odczytuje {len(catalogue)} kolumn, zgrupowanych poniżej według rodzaju "
        "informacji. Nazwy pochodzą wprost z definicji w kodzie, więc katalog nie może "
        "rozminąć się z tym, co modele faktycznie czytają.",
    )
    for group in sorted(by_group):
        add_heading(doc, f"5.2.{sorted(by_group).index(group) + 1}. {group}", 3)
        add_table(
            doc,
            ["Kolumna", "Opis"],
            [[f.name, f.description] for f in by_group[group]],
            [4.5, 11.5],
        )
    add_caption(
        doc, "Tabela",
        f"Pełny katalog {len(catalogue)} cech treningowych w {len(by_group)} grupach.",
    )
    page_break(doc)


# ── 6 ────────────────────────────────────────────────────────────────────────

def _chapter_6_labels(doc: Document) -> None:
    add_heading(doc, "6. Etykiety", 1)

    add_heading(doc, "6.1. Kierunek", 2)
    add_body(
        doc,
        "Etykieta kierunku jest znakiem zwrotu w przód na zadanym horyzoncie i przyjmuje dwie "
        "wartości. Nie istnieje klasa pośrednia ani próg oddzielający ruch istotny od "
        "nieistotnego: taki próg byłby dodatkowym parametrem badania, wpływającym na raportowaną "
        "trafność, a jego uzasadnienie musiałoby odwoływać się do kosztów transakcyjnych, czyli "
        "do warstwy decyzyjnej, a nie do danych. Zwrot dokładnie zerowy nie ma strony i jest "
        "wykluczany z oceny.",
    )

    add_heading(doc, "6.2. Głębokość", 2)
    _, labels = parse_depth_bins(DEFAULT_DEPTH_BINS_STR)
    add_body(
        doc,
        f"Wielkość ruchu prognozowana jest osobno, jako rozkład na {len(labels)} kubełkach o "
        "granicach zaczerpniętych z ciągu Fibonacciego. Granice rosną w przybliżeniu "
        "wykładniczo, co odpowiada rozkładowi wielkości ruchów: ruchy małe są częste, wielkie "
        "rzadkie, a podział równomierny umieściłby niemal wszystkie obserwacje w pierwszym "
        "przedziale.",
    )
    add_table(
        doc,
        ["Kubełek", "Zakres ruchu [%]"],
        [[label, label.replace("-", " – ").replace("+", " i więcej")] for label in labels],
        [4.0, 6.0],
    )
    add_caption(doc, "Tabela", "Kubełki głębokości ruchu.")
    add_body(
        doc,
        "Każdy artefakt zapisuje własny schemat kubełków. Przebieg zapisany pod wcześniejszym, "
        "ośmioprzedziałowym schematem pozostaje czytelny, a narzędzie zakładające bieżący "
        "schemat pominęłoby ostatni kubełek i wraz z nim znaczną część masy prawdopodobieństwa "
        "na horyzontach długich.",
    )

    add_heading(doc, "6.3. Horyzonty", 2)
    add_body(
        doc,
        "Horyzont zdefiniowany jest jako rozpiętość kalendarzowa, nie liczba świec, i "
        "przeliczany na kroki osobno dla każdego interwału. Dzięki temu ten sam horyzont oznacza "
        "tę samą rozpiętość czasu w przebiegu godzinowym i dziennym, a wyniki obu pozostają "
        "porównywalne.",
    )
    add_table(
        doc,
        ["Nazwa", "Dni", "Tygodni", "Uzasadnienie"],
        [
            [h.name, str(h.forward_days), str(h.forward_days // 7), reason]
            for h, reason in zip(
                DEFAULT_HORIZONS,
                [
                    "najkrótszy horyzont zachowujący pełną moc statystyczną przy dostępnej "
                    "historii",
                    "cztery tygodnie; podstawowy horyzont decyzyjny",
                    "trzynaście tygodni; zastąpił półrocze, ponieważ okno dwuletnie zawiera 7,0 "
                    "obserwacji niezależnych dla kwartału wobec 3,0 dla półrocza",
                    "cztery kwartały dokładnie; 364 dni zamiast 365, ponieważ 365 nie dzieli się "
                    "przez 7 i zaokrąglało się odmiennie przy różnych interwałach",
                ],
                strict=False,
            )
        ],
        [2.2, 1.6, 1.8, 10.4],
    )
    add_caption(doc, "Tabela", "Zbiór horyzontów prognostycznych.")
    add_body(
        doc,
        "Nazwy nie są zapisywane na stałe w żadnym module poza definicją domenową. Przebieg "
        "zawierający horyzont nieznany bieżącej wersji systemu powoduje błąd oceny, zamiast "
        "zostać ocenionym na zerowej liczbie wierszy i wyglądać jak model, który niczego nie "
        "prognozował.",
    )
    page_break(doc)


# ── 7 ────────────────────────────────────────────────────────────────────────

def _chapter_7_models(doc: Document) -> None:
    add_heading(doc, "7. Architektury modeli", 1)

    add_body(
        doc,
        "Porównywanych jest pięć rodzin o rosnącej pojemności i rosnącym koszcie obliczeniowym. "
        "Wszystkie realizują ten sam protokół kroczący (7.6) i wystawiają identyczny interfejs, "
        "dzięki czemu różnice w wynikach można przypisać architekturze, a nie sposobowi jej "
        "uruchomienia.",
    )
    add_table(
        doc,
        ["Model", "Rodzaj", "Pojemność", "Koszt (1 symbol, 4 horyzonty)"],
        [
            ["baseline", "częstości empiryczne z wygładzaniem Laplace'a", "brak parametrów", "~8 s"],
            ["ml", "gradient boosting na histogramach", "50 iteracji, 15 liści", "~190 s"],
            ["xgboost", "gradient boosting", "100 drzew, głębokość 6", "~30 s"],
            ["gru", "rekurencyjna sieć bramkowa", "2 warstwy, 64 jednostki", "~600 s"],
            ["tft", "Temporal Fusion Transformer", "32 jednostki, 2 głowice uwagi", "~5900 s"],
        ],
        [2.4, 6.0, 4.0, 3.6],
    )
    add_caption(doc, "Tabela", "Porównywane rodziny modeli. Czasy zmierzone na 16 rdzeniach CPU.")

    add_heading(doc, "7.1. Model odniesienia", 2)
    add_body(
        doc,
        "Model odniesienia zlicza częstości kierunków w oknie i wygładza je regułą Laplace'a. "
        "Nie zawiera uczenia, więc stanowi granicę, poniżej której nie ma sensu schodzić: model "
        "uczący przegrywający z częstością bazową nie wnosi nic. Okno zlicza wyłącznie wyniki "
        "już zrealizowane w chwili prognozy — zliczanie ostatnich świec oznaczałoby korzystanie "
        "ze zwrotów, które jeszcze nie nastąpiły, czyli przewagę, której punkt odniesienia mieć "
        "nie może.",
    )

    add_heading(doc, "7.2. Modele gradientowe", 2)
    add_body(
        doc,
        "Dwie implementacje wzmacniania gradientowego — z biblioteki scikit-learn i z XGBoost — "
        "reprezentują podejście tabelaryczne, w którym każda świeca jest niezależnym wektorem "
        "cech. Nie modelują zależności czasowej wprost; informacja o przeszłości wchodzi "
        "wyłącznie przez cechy liczone na oknach. Są tanie obliczeniowo, co pozwala uruchamiać "
        "je wielokrotnie i traktować jako punkt odniesienia dla modeli sekwencyjnych.",
    )

    add_heading(doc, "7.3. Sieć rekurencyjna GRU", 2)
    add_body(
        doc,
        "Sieć rekurencyjna z bramkami przetwarza sekwencję kolejnych świec, utrzymując stan "
        "ukryty. Bramki aktualizacji i resetu regulują, jaka część stanu przechodzi dalej, co "
        "łagodzi problem zanikającego gradientu występujący w prostych sieciach rekurencyjnych. "
        "W odróżnieniu od modeli tabelarycznych zależność czasowa jest tu modelowana wprost, "
        "kosztem znacznie większej liczby parametrów przy tej samej liczbie obserwacji.",
    )

    add_heading(doc, "7.4. Temporal Fusion Transformer", 2)
    add_body(
        doc,
        "Architektura łącząca kodowanie rekurencyjne z mechanizmem uwagi po osi czasu oraz "
        "sieciami wyboru zmiennych, które uczą się wagi poszczególnych cech. Jest to model o "
        "największej pojemności w zestawieniu i zarazem najdroższy — jeden przebieg na jednym "
        "symbolu kosztuje około godziny i pół, czyli o rząd wielkości więcej niż pozostałe "
        "łącznie.",
    )
    add_body(
        doc,
        "Prognoza całego miesiąca douczania powstaje w jednym przebiegu wsadowym, z historią "
        "celu zamaskowaną na granicy miesiąca. Konstrukcja osobnego zbioru dla każdej świecy "
        "była pierwotną przyczyną wielogodzinnego czasu wykonania i została zastąpiona.",
    )

    add_heading(doc, "7.5. Trening wielosymbolowy", 2)
    add_body(
        doc,
        "Modele tabelaryczne przyjmują opcjonalną listę dodatkowych symboli, których wiersze "
        "wchodzą do zbioru treningowego; prognoza pozostaje wyznaczana dla jednego symbolu. "
        "Wiersze wybierane są po znaczniku czasu, nigdy po numerze wiersza — instrumenty "
        "notowane od różnych dat nie mają wspólnej numeracji, a okno oparte na indeksie "
        "wprowadziłoby do zbioru treningowego świece, które jeszcze nie nastąpiły.",
    )
    add_body(
        doc,
        "Zysk z takiego łączenia jest wyraźnie mniejszy, niż sugeruje liczba wierszy. Zwroty "
        "kryptowalut są silnie skorelowane — zmierzona średnia korelacja par wynosi około 0,60 "
        "na horyzoncie tygodniowym i 0,76 na rocznym — a liczba symboli efektywnych dana wzorem "
        "n / (1 + (n−1)ρ) dąży do 1/ρ. Trzy symbole odpowiadają zatem 1,19–1,36 symbolu "
        "niezależnego, a żadna ich liczba nie podniesie horyzontu rocznego powyżej około 1,3. "
        "Jest to argument za prognozą przekrojową, a nie za łączeniem większej liczby "
        "instrumentów.",
    )

    add_heading(doc, "7.6. Protokół kroczący", 2)
    add_body(
        doc,
        "Model douczany jest raz na miesiąc kalendarzowy, na oknie kończącym się przed "
        "prognozowaną świecą, po czym prognozuje kolejne świece tego miesiąca. Zbiór treningowy "
        "zawiera wyłącznie etykiety zrealizowane w chwili prognozy: etykieta świecy k wymaga "
        "zamknięcia świecy k+F, więc nadaje się do uczenia przy prognozowaniu świecy i dopiero "
        "wtedy, gdy k+F < i.",
    )

    add_heading(doc, "7.7. Okno treningowe wyprowadzane z horyzontu", 2)
    add_body(
        doc,
        "Szerokość okna nie jest parametrem konfiguracyjnym, lecz wynika z horyzontu i "
        "dostępnej historii. Kolidują tu dwa wymagania. Model potrzebuje wierszy — kilkuset, aby "
        "dopasowanie kilkudziesięciu cech nie było rysowaniem krzywej przez punkty. Statystyka "
        "potrzebuje obserwacji niezależnych, których okno o szerokości W zawiera (W − F) / F. "
        "Pierwsze wymaganie wiąże na horyzontach krótkich, drugie na długich.",
    )
    add_body(
        doc,
        "Okno wynosi zatem F + max(minimalna liczba wierszy, 3F) i jest przycinane przez długość "
        "historii. Poszerzanie okna dla wszystkich horyzontów nie jest rozwiązaniem: historia "
        "jest skończona, więc każda świeca oddana treningowi to świeca, której nie da się "
        "ocenić. Dla najdłuższego dostępnego szeregu maksimum tego, co można osiągnąć po obu "
        "stronach jednocześnie, wynosi około 3,45 obserwacji — przy oknie równym połowie "
        "historii.",
    )

    add_heading(doc, "7.8. Kalibracja i kryterium zatrzymania", 2)
    add_body(
        doc,
        "Prawdopodobieństwa kalibrowane są skalowaniem temperatury na wycinku wydzielonym "
        "wewnątrz okna treningowego — nigdy na świecach późniejszych, ponieważ kalibrator "
        "widziałby wtedy etykiety niedostępne prognozie.",
    )
    add_body(
        doc,
        "Modele głębokie zatrzymują uczenie na stracie mierzonej po kalibracji, a nie na "
        "surowej entropii krzyżowej. Surowa strata karze za nadmierną pewność, sieć staje się "
        "nadmiernie pewna na długo przed utratą zdolności rozróżniania, a kalibrator usuwa tę "
        "nadmierną pewność krok później — zatrzymywanie na surowej wartości zwracało model po "
        "jednej epoce, podczas gdy trafność wciąż rosła.",
    )
    page_break(doc)


# ── 8 ────────────────────────────────────────────────────────────────────────

def _chapter_8_methodology(doc: Document) -> None:
    add_heading(doc, "8. Metodyka pomiaru", 1)
    add_body(
        doc,
        "Rozdział opisuje warstwę, która czyni porównanie architektur rozstrzygalnym. Bez niej "
        "wynik „architektury nie różnią się” pozostaje nieodróżnialny od „pomiar był zbyt "
        "zaszumiony”, a każdy pojedynczy wynik dodatni — nieodróżnialny od artefaktu.",
        italic=True,
    )

    add_heading(doc, "8.1. Miary bez wolnych parametrów", 2)
    add_body(
        doc,
        "Model oceniany jest trafnością, wynikiem Briera, stratą logarytmiczną i oczekiwanym "
        "błędem kalibracji. Żadna z tych miar nie ma parametru, który dałoby się dobrać tak, aby "
        "wynik wyglądał lepiej. Jest to własność celowa i odróżnia je od symulacji handlowej, "
        "której wynik zależy od kilkunastu decyzji wykonawczych.",
    )

    add_heading(doc, "8.2. Przedziały ufności przy nakładających się etykietach", 2)
    add_body(
        doc,
        "Każda trafność jest średnią po świecach o nakładających się etykietach. Naiwny przedział "
        "ufności traktuje je jako niezależne i wychodzi kilkukrotnie za wąski, przez co wynik "
        "nieodróżnialny od losowego wyglądałby na istotny. System stosuje bootstrap blokowy o "
        "długości bloku równej rozpiętości horyzontu.",
    )
    add_body(
        doc,
        "Poniżej dziesięciu bloków przedział nie jest wyznaczany, lecz odmawiany wraz z podaniem "
        "przyczyny. Próg ten wynika z obserwacji, a nie z konwencji: przy trzech blokach dwa z "
        "pięćdziesięciu pięciu wyznaczonych przedziałów nie obejmowały własnej oceny punktowej, "
        "ponieważ cztery bloki losowane z szeregu zawierającego trzy nie są w stanie go "
        "odtworzyć.",
    )

    add_heading(doc, "8.3. Moc statystyczna jako ograniczenie", 2)
    add_body(
        doc,
        "Liczba obserwacji niezależnych to liczba świec podzielona przez rozpiętość horyzontu. "
        "System ostrzega przed uruchomieniem przebiegu o horyzontach, dla których liczba ta jest "
        "zbyt mała — osobno po stronie treningu i po stronie oceny. Obie strony czerpią z tej "
        "samej skończonej historii, więc poszerzanie okna przenosi problem, zamiast go usuwać, a "
        "ostrzeżenie o jednej tylko stronie byłoby zaproszeniem do poszerzania okna aż do "
        "zaniku możliwości oceny.",
    )

    add_heading(doc, "8.4. Stabilność w czasie", 2)
    add_body(
        doc,
        "Miary jakości wyliczane są ponadto osobno dla każdego miesiąca kalendarzowego. "
        "Pojedyncza średnia z całego okresu maskuje sytuację, w której model działa w jednym "
        "reżimie rynkowym i zawodzi w innym; rozbicie na miesiące ujawnia to i pozwala stosować "
        "testy nieparametryczne, w których jednostką jest miesiąc, a nie świeca — co jest "
        "istotne, gdy etykiety wewnątrz miesiąca nakładają się niemal całkowicie.",
    )

    add_heading(doc, "8.5. Symulacja handlowa i cztery hipotezy zerowe", 2)
    add_body(
        doc,
        "Symulacja nie jest oceną prognozy. Ten sam przebieg odtworzony pod pięcioma politykami "
        "ekspozycji daje wyniki od głębokiej straty do kilkusetprocentowego zysku, co oznacza, że "
        "raportowana stopa zwrotu mówi więcej o przyjętej polityce niż o prognozie. Wynik "
        "podawany jest wyłącznie jako kontekst ekonomiczny i zawsze wraz z czterema punktami "
        "odniesienia.",
    )
    add_table(
        doc,
        ["Punkt odniesienia", "Co usuwa", "Siła"],
        [
            ["Kup i trzymaj", "nic — punkt odniesienia rynkowy", "najsłabszy"],
            ["Stała ekspozycja równa średniej strategii",
             "efekt samego poziomu zaangażowania", "średni"],
            ["Ta sama ścieżka ekspozycji przesunięta cyklicznie",
             "efekt wyczucia momentu — test właściwy", "mocny"],
            ["Reguła momentum bez modelu",
             "przewagę wynikającą z trendu, nie z prognozy", "wiążący"],
        ],
        [5.6, 6.4, 4.0],
    )
    add_caption(doc, "Tabela", "Hipotezy zerowe towarzyszące każdej symulacji.")
    add_body(
        doc,
        "Trzeci punkt jest testem właściwym: utrzymuje sekwencję ekspozycji bez zmian i zmienia "
        "wyłącznie jej ułożenie w czasie. Jeżeli rzeczywiste ułożenie nie wypada lepiej niż "
        "dowolne przesunięte, prognoza nie wnosi wyczucia momentu, a osiągnięty wynik pochodzi z "
        "poziomu zaangażowania.",
    )

    add_heading(doc, "8.6. Odtwarzalność i ziarnowanie", 2)
    add_body(
        doc,
        "Każdy przebieg zapisuje ziarno, tryb deterministyczny, wyprowadzone okna treningowe i "
        "liczbę faktycznie wykonanych epok. Losowość każdego okna douczania wyprowadzana jest z "
        "jego własnej tożsamości — ziarna bazowego, modelu, horyzontu, miesiąca i symbolu — a nie "
        "z jednego wspólnego strumienia.",
    )
    add_body(
        doc,
        "Wspólny strumień powodowałby, że liczba kroków uczenia wykonanych gdziekolwiek przesuwa "
        "inicjalizację wszystkiego, co następuje po niej, czyniąc eksperyment z jedną zmienną "
        "niewykonalnym. Pominięcie symbolu w tej tożsamości spowodowało z kolei, że wszystkie "
        "instrumenty startowały z identycznych wag — konsekwencje opisano w rozdziale 10.5.",
    )
    add_body(
        doc,
        "Odtwarzalność zweryfikowano pomiarowo: powtórzenie tej samej konfiguracji daje plik "
        "predykcji identyczny co do bajtu dla każdego modelu korzystającego z losowości.",
    )
    page_break(doc)


# ── 9 ────────────────────────────────────────────────────────────────────────

def _chapter_9_results(doc: Document, runs: dict) -> None:
    add_heading(doc, "9. Wyniki: porównanie architektur", 1)

    total = sum(len(v) for v in runs.values())
    seeds = sorted({r["seed"] for v in runs.values() for r in v if r["seed"] is not None})
    add_body(
        doc,
        f"Podstawą tabel jest {total} przebiegów odczytanych z artefaktów projektu"
        + (f", pod ziarnami: {', '.join(str(s) for s in seeds)}." if seeds else ".")
        + " Każdą konfigurację uruchomiono wielokrotnie, ponieważ pojedynczy przebieg niesie "
        "szum inicjalizacji, którego przedział ufności liczony wewnątrz tego przebiegu nie "
        "obejmuje.",
    )

    if not runs:
        add_body(
            doc,
            "UWAGA: w katalogu artefaktów nie znaleziono ocenionych przebiegów. Tabele wyników "
            "wypełnią się po zakończeniu przeliczeń; dokument należy wtedy wygenerować ponownie.",
            italic=True,
        )
        page_break(doc)
        return

    add_heading(doc, "9.1. Ranking architektur wobec podłogi szumu", 2)
    add_body(
        doc,
        "Tabele poniżej odpowiadają wprost na pytanie badawcze. Wielkością rozstrzygającą jest "
        "porównanie rozstępu między architekturami z rozrzutem tej samej architektury pod "
        "różnymi ziarnami. Jeżeli rozstęp nie przekracza rozrzutu, ranking architektur nie ma "
        "podstaw.",
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

        add_body(doc, f"Horyzont „{horizon}”:")
        add_table(
            doc,
            ["Poz.", "Architektura", "Przebiegów", "Trafność średnia"],
            [[str(rank), model, str(len(per_model[model])), fmt(means[model])]
             for rank, model in enumerate(order, start=1)],
            [1.6, 5.0, 2.6, 4.0],
            font=9,
        )
        if noise is not None:
            verdict = (
                "rozstęp mieści się w rozrzucie ziarna — ranking architektur nie ma podstaw"
                if spread <= noise
                else "rozstęp przekracza rozrzut ziarna"
            )
            add_caption(
                doc, "Tabela",
                f"Horyzont „{horizon}”. Rozstęp między architekturą najlepszą a najgorszą: "
                f"{spread:.3f}. Typowy rozrzut tej samej architektury pod innym ziarnem: "
                f"{noise:.3f}. Wniosek: {verdict}.",
            )
        else:
            add_caption(
                doc, "Tabela",
                f"Horyzont „{horizon}”. Rozstęp: {spread:.3f}. Brak powtórzeń pod różnymi "
                "ziarnami — rozrzutu inicjalizacji nie zmierzono.",
            )

    add_heading(doc, "9.2. Wyniki szczegółowe", 2)
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
            [2.2, 2.4, 1.5, 2.2, 1.8, 3.2, 2.4],
        )
        add_caption(
            doc, "Tabela",
            f"Horyzont „{horizon}”. Kolumna „Odch.” to odchylenie standardowe trafności między "
            "przebiegami różniącymi się wyłącznie ziarnem — miara szumu inicjalizacji, "
            "niewidoczna w przedziale pojedynczego przebiegu.",
        )

    _results_stability(doc)
    page_break(doc)


def _results_stability(doc: Document) -> None:
    """Per-month quality, read from the stability artifacts."""
    reports = sorted(glob.glob(f"{REPORT_ROOT}/eval/*/*/stability_summary.json"))
    if not reports:
        return

    add_heading(doc, "9.3. Stabilność w czasie", 2)
    add_body(
        doc,
        "Średnia z całego okresu maskuje sytuację, w której model działa w jednym reżimie "
        "rynkowym i zawodzi w innym. Poniższe zestawienie rozbija jakość na miesiące "
        "kalendarzowe i podaje, w jakim odsetku miesięcy trafność przekroczyła poziom losowy — "
        "wielkość odporną na pojedyncze wartości skrajne.",
    )

    rows = []
    for path in reports:
        parts = path.replace("\\", "/").split("/")
        symbol, slug = parts[-3], parts[-2]
        model = slug.split("_")[0]
        if symbol not in SYMBOLS or model not in MODELS:
            continue
        try:
            payload = json.load(open(path, encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for horizon in HORIZON_NAMES:
            values = [
                cell["accuracy"]
                for month in payload.get("months", [])
                if (cell := (month.get("horizons") or {}).get(horizon))
                and not cell.get("sparse")
                and cell.get("accuracy") is not None
                and (cell.get("samples") or 0) >= 20
            ]
            if len(values) < 6:
                continue
            arr = np.array(values)
            above = int((arr > 0.5).sum())
            rows.append([
                symbol.replace("USDT", ""), model, horizon, str(len(arr)),
                fmt(float(arr.mean())), fmt(float(np.median(arr))),
                f"{above}/{len(arr)}", f"{above / len(arr):.0%}",
            ])
    if not rows:
        return
    add_table(
        doc,
        ["Symbol", "Model", "Horyzont", "Mies.", "Średnia", "Mediana", "> 0,5", "Udział"],
        sorted(rows, key=lambda r: (r[2], r[0], r[1])),
        [1.9, 2.1, 2.1, 1.5, 1.9, 1.9, 1.8, 1.8],
        font=8,
    )
    add_caption(
        doc, "Tabela",
        "Jakość prognozy miesiąc po miesiącu. Kolumna „Udział” podaje odsetek miesięcy, w "
        "których trafność przekroczyła poziom losowy; wartość bliska połowie oznacza brak "
        "trwałej przewagi niezależnie od średniej z całego okresu.",
    )


# ── 10 ───────────────────────────────────────────────────────────────────────

def _chapter_10_verification(doc: Document) -> None:
    add_heading(doc, "10. Weryfikacja poprawności implementacji", 1)
    add_body(
        doc,
        "Rozdział dokumentuje defekty wykryte w trakcie budowy systemu i stanowi materiał "
        "dowodowy dla ostatniego zdania tezy. Wspólną cechą wszystkich pozycji jest to, że "
        "defekt dawał wynik wyglądający na poprawny i nie ujawniał się w samej trafności "
        "prognozy — porównanie architektur przeprowadzone bez warstwy weryfikacji porównywałoby "
        "zatem artefakty.",
    )
    add_table(
        doc,
        ["Defekt", "Objaw bez weryfikacji", "Wykryty przez"],
        [
            ["Odczyt klas kierunku po pozycji przy enkoderze sortującym etykiety alfabetycznie",
             "trafność równa 1 − trafność; przy wyniku bliskim losowego nieodróżnialne od szumu",
             "test na szeregu syntetycznym o znanej odpowiedzi"],
            ["Okno treningowe węższe od horyzontu",
             "stały rozkład jednostajny punktowany jako prognoza przez wiele miesięcy",
             "kontrola szerokości okna przed uruchomieniem"],
            ["Wspólny, nieweryfikowany katalog predykcji",
             "wyniki ostatnio uruchomionego modelu przypisywane wszystkim pozostałym",
             "wprowadzenie wersjonowania przebiegów"],
            ["Jednorodne rozkłady głębokości w trzech modelach",
             "najwyższa rekomendacja nieosiągalna dla części modeli",
             "porównanie rozkładów między modelami"],
            ["Wspólny strumień inicjalizacji dla wszystkich symboli",
             "cztery symbole wykazujące przewagę jednocześnie; istotność p = 0,0015 dla "
             "zjawiska nieistniejącego",
             "powtórzenie pod innym ziarnem oraz kontrola na szeregu losowym"],
            ["Puste okno normalizacji dla symbolu notowanego od daty wewnątrz zakresu",
             "przerwanie przebiegu wyjątkiem zamiast oznaczenia świecy jako nietrenowanej",
             "przeliczenie obejmujące symbol o krótszej historii"],
        ],
        [4.6, 6.2, 5.2],
    )
    add_caption(doc, "Tabela", "Defekty wykryte w trakcie budowy systemu.")

    add_heading(doc, "10.1. Zamiana klas kierunku w modelu TFT", 2)
    add_body(
        doc,
        "Enkoder etykiet biblioteki pytorch-forecasting numeruje klasy w porządku "
        "alfabetycznym, dołączając na początku klasę zastępczą dla wartości brakujących. "
        "Odczytanie kolumn wyjściowych po pozycji, po odrzuceniu kolumny zastępczej, przypisało "
        "prawdopodobieństwo klasy „spadek” do klasy „wzrost” i odwrotnie.",
    )
    add_body(
        doc,
        "Defekt był niewidoczny dla wszystkich raportowanych miar: zamiana dwóch klas nie zmienia "
        "entropii, pewności ani wyniku Briera, a trafność przekształca w swoje dopełnienie, co "
        "przy wartościach bliskich 0,5 wygląda jak szum. Wykryto go dopiero na szeregu "
        "syntetycznym, w którym kierunek jest deterministyczną funkcją poprzedniej świecy: "
        "zamiast oczekiwanej zgodności 0,987 model osiągnął 0,013.",
    )

    add_heading(doc, "10.2. Horyzonty bez danych treningowych", 2)
    add_body(
        doc,
        "Przy oknie treningowym węższym od horyzontu zbiór treningowy jest pusty, ponieważ żadna "
        "etykieta nie zdążyła się zrealizować. Wcześniejsza wersja emitowała wtedy stały rozkład "
        "jednostajny, a warstwa oceny punktowała go jak prognozę. Obecnie taka konfiguracja "
        "powoduje błąd przed uruchomieniem, a świece bez możliwości uczenia oznaczane są jako "
        "nietrenowane i wykluczane z oceny, planowania i symulacji.",
    )

    add_heading(doc, "10.3. Brak przypisania wyników do modelu", 2)
    add_body(
        doc,
        "Wszystkie modele zapisywały prognozy do wspólnego, nieopatrzonego wersją katalogu, z "
        "którego czytały warstwy pochodne. W praktyce oznaczało to, że model uruchomiony jako "
        "ostatni przejmował wyniki wszystkich pozostałych, a przebieg godzinowy nadpisywał "
        "dzienny. Rozwiązaniem jest wersjonowanie opisane w rozdziale 3.4.",
    )

    add_heading(doc, "10.4. Jednorodne rozkłady głębokości", 2)
    add_body(
        doc,
        "Trzy z pięciu modeli zwracały jednakowe prawdopodobieństwo w każdym kubełku głębokości. "
        "Ponieważ warstwa decyzyjna wyznacza ryzyko jako masę prawdopodobieństwa w kubełkach "
        "dużych ruchów, jednorodny rozkład przypinał miarę ryzyka do stałej i czynił najwyższą "
        "rekomendację nieosiągalną dla tych modeli — przy zachowaniu jej osiągalności dla "
        "pozostałych. Porównanie międzymodelowe było więc obciążone w sposób niewidoczny w "
        "miarach jakości prognozy kierunku.",
    )

    add_heading(doc, "10.5. Wspólny strumień inicjalizacji", 2)
    add_body(
        doc,
        "Najpoważniejszy z wykrytych defektów, ponieważ wyprodukował wynik pozytywny, który "
        "przeszedł replikację poza próbą.",
    )
    add_body(
        doc,
        "Jedna komórka wyników przekroczyła próg istotności. Hipotezę zweryfikowano na czterech "
        "symbolach nieuczestniczących w jej powstaniu, a zaplanowany z góry test znaków wypadł "
        "pozytywnie na poziomie p = 0,0015. Dopiero trzy dalsze kontrole ją obaliły.",
    )
    add_bullets(
        doc,
        [
            "Powtórzenie pod innym ziarnem przesuwało trafność pojedynczego symbolu o wartość "
            "większą niż deklarowany efekt, przy czym kierunek przesunięcia był różny dla "
            "różnych symboli — to, który symbol wypada „powyżej losowego”, okazało się loterią.",
            "Trzy szeregi losowe o dryfie i zmienności dopasowanych do danych rzeczywistych, w "
            "których prawdziwa odpowiedź wynosi dokładnie 0,500, dały poziom odniesienia, "
            "względem którego różnica przestawała być istotna.",
            "Analiza kodu wykazała, że tożsamość, z której wyprowadzane jest ziarno okna, nie "
            "zawierała symbolu. Cztery „niezależne” symbole startowały z identycznych wag, z "
            "tymi samymi maskami dropoutu i tą samą kolejnością porcji — a więc stanowiły jedno "
            "losowanie sekwencji inicjalizacji zastosowane do czterech zbiorów, co jest dokładnie "
            "założeniem, na którym opierał się test istotności.",
        ],
    )
    add_body(
        doc,
        "Po uzupełnieniu tożsamości o symbol i powtórzeniu pomiaru na dwunastu niezależnych "
        "przebiegach średnia trafność spadła do wartości nieodróżnialnej od poziomu odniesienia, "
        "a komórka, od której rzecz się zaczęła, straciła niemal całą deklarowaną przewagę. "
        "Wynik wycofano.",
    )
    add_body(
        doc,
        "Wniosek metodyczny, przenoszony do wszystkich raportowanych rezultatów: przedział "
        "ufności wyznaczony wewnątrz jednego przebiegu mierzy szum próbkowania i pozostaje "
        "ślepy na szum inicjalizacji o porównywalnej wielkości. Z tego powodu rozdział 9 "
        "raportuje wielokrotne przebiegi, a nie pojedyncze.",
    )

    add_heading(doc, "10.6. Puste okno normalizacji", 2)
    add_body(
        doc,
        "Zażądanie zakresu rozpoczynającego się przed pierwszym notowaniem instrumentu umieszcza "
        "jego pierwszą świecę wewnątrz zakresu, przez co pętla krocząca startuje od wiersza "
        "zerowego. Przyczynowe okno normalizacji sięga wstecz i nie znajduje nic, a statystyki "
        "pozycyjne nie są zdefiniowane dla pustego zbioru. Świeca bez historii wstecz nie ma "
        "jednak również wierszy treningowych, więc właściwym wynikiem jest oznaczenie jej jako "
        "nietrenowanej; przerwanie całego przebiegu było defektem.",
    )

    add_heading(doc, "10.7. Zakres pakietu testów", 2)
    modules = test_inventory()
    cases = sum(len(m.cases) for m in modules)
    add_body(
        doc,
        f"Pakiet liczy {cases} przypadków testowych w {len(modules)} modułach. Nie są to testy "
        "pokrycia: każdy moduł odpowiada konkretnej klasie błędów, a znaczna część przypadków "
        "opisuje w dokumentacji defekt, którego wystąpieniu zapobiega. Pełny inwentarz zawiera "
        "załącznik A.",
    )
    add_table(
        doc,
        ["Moduł", "Przypadków", "Czego pilnuje"],
        [[m.path, str(len(m.cases)), m.purpose[:160]] for m in modules[:20]],
        [4.4, 2.0, 9.6],
    )
    add_caption(
        doc, "Tabela",
        f"Pierwsze 20 z {len(modules)} modułów testowych. Pełne zestawienie w załączniku A.",
    )
    page_break(doc)


# ── 11 ───────────────────────────────────────────────────────────────────────

def _chapter_11_interfaces(doc: Document) -> None:
    add_heading(doc, "11. Interfejsy", 1)

    commands = cli_reference()
    routes = api_reference()

    add_heading(doc, "11.1. Wiersz poleceń", 2)
    add_body(
        doc,
        f"Wiersz poleceń wystawia {len(commands)} poleceń zgrupowanych tematycznie i stanowi "
        "kanoniczny interfejs systemu: warstwa HTTP nie zawiera własnej logiki, lecz uruchamia "
        "te same polecenia jako podprocesy. Pełna referencja wraz z flagami znajduje się w "
        "załączniku B.",
    )
    add_table(
        doc,
        ["Polecenie", "Opis", "Flag"],
        [[c.path, c.summary[:110], str(len(c.options))] for c in commands],
        [4.6, 9.4, 2.0],
    )
    add_caption(doc, "Tabela", "Polecenia wiersza poleceń.")

    add_heading(doc, "11.2. Interfejs HTTP", 2)
    add_body(
        doc,
        f"Serwer wystawia {len(routes)} tras. Żadna z nich nie wykonuje długotrwałych obliczeń "
        "w procesie serwera: zadania trafiają do kolejki obsługiwanej przez jeden wątek roboczy, "
        "a przeglądarka odpytuje o jej stan. Kolejka żyje po stronie serwera, więc przeładowanie "
        "strony nie gubi zadań oczekujących.",
    )
    add_table(
        doc,
        ["Metoda", "Ścieżka", "Opis"],
        [[r.methods, r.path, r.summary[:90]] for r in routes],
        [2.0, 6.4, 7.6],
    )
    add_caption(doc, "Tabela", "Trasy interfejsu HTTP.")

    add_heading(doc, "11.3. Panel przeglądarkowy", 2)
    add_body(
        doc,
        "Panel jest aplikacją jednostronicową zbudowaną na natywnych modułach ES, bez kroku "
        "budowania. Zachowanie elementów deklarowane jest atrybutem w znaczniku i obsługiwane "
        "przez jeden nasłuch delegowany, co eliminuje wstawianie kodu bezpośrednio w atrybutach "
        "— wcześniejsza wersja interpolowała nazwy symboli prosto do treści atrybutów zdarzeń.",
    )
    add_bullets(
        doc,
        [
            "Menedżer danych — inwentarz jeziora danych, pobieranie, agregacja, kontrola jakości.",
            "Modele — konfiguracja i uruchamianie treningu, kolejka zadań, inwentarz przebiegów.",
            "Ocena — miary jakości, przedziały ufności, wykres stabilności w czasie.",
            "Symulacja — parametry wykonawcze, wynik wraz z czterema hipotezami zerowymi.",
        ],
    )
    add_body(
        doc,
        "Formularz treningu pobiera wartości domyślne z interfejsu HTTP, a nie ze znaczników "
        "strony. Wcześniejsza wersja zawierała je wpisane wprost w znaczniku, przez co ten sam "
        "model uruchamiany z panelu wykonywał inną liczbę epok niż uruchamiany z wiersza "
        "poleceń — wartość domyślna obowiązuje bowiem wyłącznie przy nieobecności flagi.",
    )

    add_heading(doc, "11.4. Konfiguracja", 2)
    fields = settings_reference()
    add_body(
        doc,
        f"Konfiguracja obejmuje {len(fields)} pól walidowanych typami. Wszystkie ścieżki "
        "wyprowadzane są z jednego korzenia jeziora danych, dzięki czemu uruchomienie na innym "
        "zbiorze wymaga zmiany jednej wartości. Pełne zestawienie w załączniku D.",
    )
    page_break(doc)


# ── 12–13 ────────────────────────────────────────────────────────────────────

def _chapter_12_limits(doc: Document) -> None:
    add_heading(doc, "12. Ograniczenia i kierunki dalszych prac", 1)

    add_heading(doc, "12.1. Zakres wniosków", 2)
    add_bullets(
        doc,
        [
            "Zbiór informacyjny obejmuje wyłącznie cechy wyprowadzone z ceny i wolumenu tego "
            "samego instrumentu. Nie uwzględniono danych on-chain, stawek finansowania, "
            "głębokości arkusza zleceń, danych międzyrynkowych ani sentymentu. Jest to granica "
            "przyjęta świadomie — praca testuje słabą formę hipotezy efektywności — ale wniosków "
            "nie wolno rozciągać poza nią.",
            "Prognozowany jest znak zwrotu, czyli sformułowanie najtrudniejsze. Model może nie "
            "mieć przewagi kierunkowej i jednocześnie poprawnie prognozować zmienność, czego "
            "praca nie bada.",
            "Nie badano prognozy przekrojowej. Pomiar korelacji zwrotów wskazuje, że wspólny "
            "czynnik rynkowy odpowiada za większość wariancji, a jego odjęcie jest jedyną drogą "
            "powyżej granicy wynikającej ze wzoru na liczbę symboli efektywnych.",
            "Zastosowano modele o niewielkiej pojemności. Przy zmierzonej liczbie obserwacji "
            "niezależnych pojemność nie jest jednak wiążącym ograniczeniem, więc rozszerzenie "
            "modeli bez rozszerzenia danych nie zmieniłoby wyniku.",
            "Badanie obejmuje jeden rynek i jeden reżim regulacyjny. Wyniki nie przenoszą się "
            "automatycznie na rynki o nieciągłym obrocie.",
        ],
    )

    add_heading(doc, "12.2. Kierunki dalszych prac", 2)
    add_table(
        doc,
        ["Kierunek", "Uzasadnienie", "Oczekiwana wartość"],
        [
            ["Prognoza przekrojowa", "Odjęcie wspólnego czynnika rynkowego, którego obecność "
                                     "ogranicza wartość łączenia instrumentów.", "najwyższa"],
            ["Rozszerzenie zbioru informacyjnego",
             "Dane on-chain, stawki finansowania i arkusz zleceń wykraczają poza słabą formę "
             "hipotezy.", "wysoka"],
            ["Zmiana wielkości prognozowanej",
             "Zmienność jest prognozowalna tam, gdzie kierunek nie jest.", "średnia"],
            ["Rozszerzenie na inne rynki",
             "Pierwotny zamiar pracy; wymaga obsługi nieciągłości i zdarzeń korporacyjnych.",
             "średnia"],
            ["Zrównoleglenie pętli kroczącej",
             "Możliwe po uniezależnieniu ziaren okien od kolejności wykonania; skraca czas, nie "
             "zmienia wyników.", "wygoda"],
        ],
        [4.2, 8.4, 3.4],
    )
    add_caption(doc, "Tabela", "Kierunki dalszych prac w kolejności oczekiwanej wartości.")

    add_heading(doc, "13. Podsumowanie", 1)
    add_body(
        doc,
        "Porównano pięć rodzin architektur uczenia maszynowego w zadaniu prognozy kierunku "
        "zmiany ceny na podstawie wyłącznie historii notowań. Żadna z nich nie osiąga trafności "
        "odróżnialnej od losowej na horyzoncie o wystarczającej mocy statystycznej, a różnice "
        "między nimi są mniejsze niż rozrzut wynikający z losowej inicjalizacji tej samej "
        "architektury — co czyni ranking oparty na pojedynczym przebiegu pozbawionym podstaw.",
    )
    add_body(
        doc,
        "Jest to wynik zgodny ze słabą formą hipotezy efektywności rynku, uzyskany na rynku "
        "wybranym tak, aby mierzyć zachowanie modelu, a nie artefakty systemu obrotu. "
        "Rozstrzygnięcie tego pytania było możliwe wyłącznie dzięki warstwie weryfikacji, której "
        "skuteczność dokumentuje sześć wykrytych defektów — z których każdy produkował wynik "
        "pozornie pozytywny, a jeden przeszedł replikację poza próbą, zanim został obalony.",
    )
    add_body(
        doc,
        "Osobnym, pozytywnym wynikiem inżynierskim jest kalibracja: system poprawnie raportuje "
        "własną niepewność tam, gdzie nie ma zdolności rozróżniania. Prognoza, która wie, że nie "
        "wie, jest użyteczna; prognoza pewna siebie i błędna nie jest.",
    )
    page_break(doc)


def _chapter_13_bibliography(doc: Document) -> None:
    add_heading(doc, "14. Bibliografia", 1)
    add_bullets(
        doc,
        [
            "Fama E. F., Efficient Capital Markets: A Review of Theory and Empirical Work, "
            "Journal of Finance, 25(2), 1970.",
            "Brier G. W., Verification of Forecasts Expressed in Terms of Probability, "
            "Monthly Weather Review, 78(1), 1950.",
            "Gneiting T., Raftery A. E., Strictly Proper Scoring Rules, Prediction, and "
            "Estimation, Journal of the American Statistical Association, 102(477), 2007.",
            "Guo C., Pleiss G., Sun Y., Weinberger K. Q., On Calibration of Modern Neural "
            "Networks, ICML, 2017.",
            "Künsch H. R., The Jackknife and the Bootstrap for General Stationary Observations, "
            "Annals of Statistics, 17(3), 1989.",
            "Politis D. N., Romano J. P., The Stationary Bootstrap, Journal of the American "
            "Statistical Association, 89(428), 1994.",
            "Cho K. i in., Learning Phrase Representations using RNN Encoder-Decoder for "
            "Statistical Machine Translation, EMNLP, 2014.",
            "Lim B., Arık S. Ö., Loeff N., Pfister T., Temporal Fusion Transformers for "
            "Interpretable Multi-horizon Time Series Forecasting, International Journal of "
            "Forecasting, 37(4), 2021.",
            "Chen T., Guestrin C., XGBoost: A Scalable Tree Boosting System, KDD, 2016.",
            "Bailey D. H., López de Prado M., The Deflated Sharpe Ratio, Journal of Portfolio "
            "Management, 40(5), 2014.",
            "López de Prado M., Advances in Financial Machine Learning, Wiley, 2018.",
            "Cont R., Empirical Properties of Asset Returns: Stylized Facts and Statistical "
            "Issues, Quantitative Finance, 1(2), 2001.",
        ],
    )
    page_break(doc)


# ── appendices ───────────────────────────────────────────────────────────────

def _appendix_tests(doc: Document) -> None:
    add_heading(doc, "Załącznik A. Inwentarz testów", 1)
    modules = test_inventory()
    cases = sum(len(m.cases) for m in modules)
    add_body(
        doc,
        f"Zestawienie {cases} przypadków testowych w {len(modules)} modułach, odczytane wprost "
        "z kodu. Jest to specyfikacja systemu w jedynej postaci, która nie może się "
        "zdezaktualizować: każdej pozycji odpowiada asercja, która wykonuje się przy każdym "
        "uruchomieniu pakietu. Opis pochodzi z dokumentacji przypadku, a gdy jej nie ma — z jego "
        "nazwy, którą pakiet formułuje zdaniem.",
    )
    for module in modules:
        add_heading(doc, module.path, 3)
        add_body(doc, module.purpose, italic=True, size=9)
        add_table(
            doc,
            ["Przypadek", "Co gwarantuje"],
            [[c.name.removeprefix("test_"), c.intent] for c in module.cases],
            [5.6, 10.4],
            font=8,
        )
    page_break(doc)


def _appendix_cli(doc: Document) -> None:
    add_heading(doc, "Załącznik B. Referencja wiersza poleceń", 1)
    commands = cli_reference()
    add_body(
        doc,
        f"Pełna referencja {len(commands)} poleceń wraz z "
        f"{sum(len(c.options) for c in commands)} flagami, odczytana z definicji poleceń. "
        "Kolumna „domyślnie” podaje wartość obowiązującą przy nieobecności flagi.",
    )
    for command in commands:
        add_heading(doc, command.path, 3)
        add_body(doc, command.summary, italic=True, size=9)
        if not command.options:
            add_body(doc, "Polecenie nie przyjmuje opcji.", size=9)
            continue
        add_table(
            doc,
            ["Flaga", "Typ", "Wym.", "Domyślnie", "Opis"],
            [[o.flags, o.kind, "tak" if o.required else "—", o.default, o.help]
             for o in command.options],
            [4.0, 1.8, 1.2, 2.4, 6.6],
            font=8,
        )
    page_break(doc)


def _appendix_api(doc: Document) -> None:
    add_heading(doc, "Załącznik C. Referencja interfejsu HTTP", 1)
    routes = api_reference()
    add_body(doc, f"Zestawienie {len(routes)} tras odczytane z definicji serwera.")
    add_table(
        doc,
        ["Metoda", "Ścieżka", "Opis"],
        [[r.methods, r.path, r.summary] for r in routes],
        [2.0, 6.0, 8.0],
    )
    page_break(doc)


def _appendix_settings(doc: Document) -> None:
    add_heading(doc, "Załącznik D. Konfiguracja", 1)
    fields = settings_reference()
    add_body(
        doc,
        f"Zestawienie {len(fields)} pól konfiguracji odczytane z definicji modelu ustawień. "
        "Każde pole można nadpisać zmienną środowiskową o tej samej nazwie.",
    )
    add_table(
        doc,
        ["Pole", "Typ", "Domyślnie", "Opis"],
        [[f.name, f.kind, f.default, f.description] for f in fields],
        [4.6, 2.6, 3.0, 5.8],
    )


# ── 11a: worked example ──────────────────────────────────────────────────────

def _chapter_11a_worked_example(doc: Document) -> None:
    """One bar traced end to end. Nothing explains a pipeline like following a
    single row through it."""
    add_heading(doc, "11.5. Przykład przejścia jednej świecy przez potok", 2)
    add_body(
        doc,
        "Poniżej prześledzono pojedynczą świecę dzienną od pobrania po rekomendację. Przykład "
        "ilustruje, w którym miejscu zapada każda decyzja i gdzie przebiegają granice "
        "przyczynowości — czyli co system wie w chwili, gdy prognozuje.",
    )
    add_table(
        doc,
        ["Krok", "Co się dzieje", "Granica przyczynowa"],
        [
            ["1. Pobranie",
             "Archiwum miesięczne trafia do jeziora danych jako świece jednominutowe.",
             "—"],
            ["2. Agregacja",
             "Świece minutowe składają się w jedną dzienną: otwarcie pierwszej, zamknięcie "
             "ostatniej, skrajne maksimum i minimum, suma wolumenów.",
             "wyłącznie minuty należące do tego dnia"],
            ["3. Cechy",
             "Dla świecy i wyliczane są wskaźniki z okien kończących się na niej — średnie "
             "kroczące, oscylatory, miary zmienności, udział wolumenu kupujących.",
             "świece o indeksie ≤ i"],
            ["4. Etykieta",
             "Znak zwrotu z i do i+F oraz kubełek jego wielkości.",
             "wymaga świecy i+F, więc etykieta powstaje z opóźnieniem"],
            ["5. Okno treningowe",
             "Przy prognozie świecy i model uczy się na etykietach k, dla których k+F < i.",
             "ostatnia użyteczna etykieta to i−F−1"],
            ["6. Wycinek kalibracyjny",
             "Najnowsza część okna treningowego zostaje wyłączona z dopasowania i służy do "
             "wyznaczenia temperatury.",
             "wciąż wewnątrz okna, nigdy po świecy i"],
            ["7. Prognoza",
             "Model zwraca rozkład na dwóch kierunkach oraz rozkład na kubełkach głębokości, "
             "po czym kalibrator koryguje ostrość.",
             "cechy świecy i"],
            ["8. Rekomendacja",
             "Przewaga i miara ryzyka przechodzą przez bramki oraz próg kosztowy, dając jeden "
             "z siedmiu stopni.",
             "bramki ryzyka to kwantyle ryzyka już zaobserwowanego, bez świecy i"],
            ["9. Symulacja",
             "Sygnał ze świecy i realizowany jest po cenie otwarcia świecy i+1.",
             "brak realizacji po cenie, której sygnał dotyczył"],
        ],
        [3.0, 8.2, 4.8],
    )
    add_caption(
        doc, "Tabela",
        "Ścieżka jednej świecy przez potok wraz z granicami przyczynowymi. Naruszenie "
        "którejkolwiek z nich objawia się jako poprawa trafności, nie jako błąd wykonania.",
    )
    add_body(
        doc,
        "Warto zwrócić uwagę na krok piąty. Etykieta świecy i−1 nie jest jeszcze znana przy "
        "prognozie świecy i, ponieważ wymaga zamknięcia świecy i−1+F. Najświeższa informacja, "
        "jaką model może wykorzystać, ma zatem F świec opóźnienia — i to właśnie ta luka, a nie "
        "liczba wierszy, ogranicza horyzonty długie.",
    )


# ── glossary ─────────────────────────────────────────────────────────────────

def _appendix_glossary(doc: Document) -> None:
    add_heading(doc, "Załącznik E. Słownik pojęć", 1)
    add_table(
        doc,
        ["Pojęcie", "Znaczenie w tym systemie"],
        [
            ["bootstrap blokowy",
             "Metoda szacowania niepewności losująca ciągi kolejnych obserwacji zamiast "
             "pojedynczych, dzięki czemu zachowuje zależność między nakładającymi się "
             "etykietami."],
            ["ECE",
             "Oczekiwany błąd kalibracji: średnia rozbieżność między deklarowaną pewnością a "
             "rzeczywistą częstością trafień."],
            ["głębokość",
             "Wielkość ruchu ceny wyrażona jako rozkład prawdopodobieństwa na kubełkach o "
             "granicach z ciągu Fibonacciego."],
            ["horyzont",
             "Rozpiętość kalendarzowa prognozy, przeliczana na liczbę świec osobno dla każdego "
             "interwału."],
            ["kalibracja",
             "Zgodność deklarowanej pewności z rzeczywistą częstością; niezależna od zdolności "
             "rozróżniania."],
            ["liczba obserwacji efektywnych",
             "Liczba świec podzielona przez rozpiętość horyzontu — przybliżenie liczby "
             "wzajemnie niezależnych obserwacji."],
            ["okno kroczące",
             "Schemat uczenia, w którym model douczany jest cyklicznie na danych poprzedzających "
             "prognozowany okres."],
            ["przeciek z przyszłości",
             "Wykorzystanie w uczeniu informacji niedostępnej w chwili prognozy; objawia się "
             "poprawą trafności, nie błędem wykonania."],
            ["przewaga",
             "Różnica prawdopodobieństw obu kierunków; z konstrukcji ograniczona przez 0,5."],
            ["reguła właściwa",
             "Miara oceny prognozy probabilistycznej osiągająca optimum wtedy i tylko wtedy, gdy "
             "prognoza podaje prawdziwe prawdopodobieństwo."],
            ["skalowanie temperaturą",
             "Jednoparametrowa kalibracja zmieniająca ostrość rozkładu bez zmiany porządku klas."],
            ["symbol efektywny",
             "Miara wartości informacyjnej łączenia instrumentów: n / (1 + (n−1)ρ), gdzie ρ to "
             "średnia korelacja zwrotów."],
            ["ślepy nawrót",
             "Symulacja handlowa; w tym systemie traktowana jako kontekst ekonomiczny, nie jako "
             "ocena prognozy."],
            ["test przesunięcia cyklicznego",
             "Hipoteza zerowa zachowująca sekwencję ekspozycji i zmieniająca wyłącznie jej "
             "ułożenie w czasie; izoluje wyczucie momentu."],
            ["ziarno okna",
             "Liczba wyprowadzona z tożsamości okna douczania, z której inicjalizowany jest "
             "generator liczb losowych tego okna."],
        ],
        [4.2, 11.8],
        font=9,
    )
    add_caption(doc, "Tabela", "Słownik pojęć używanych w dokumencie.")


if __name__ == "__main__":
    written = build()
    print(f"written: {written} ({os.path.getsize(written) / 1024:.0f} kB)")
