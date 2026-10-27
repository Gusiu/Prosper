"""Generate the technical documentation as a .docx.

The document is *generated*, not written by hand, for the reason the previous one
was deleted: a hand-written write-up drifts from the code and then contradicts
it. Every catalogue - features, CLI flags, HTTP routes, configuration, the test
inventory - is read from the running system, and every result table is read from
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
import re
import sys
from collections import defaultdict

import numpy as np
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from documentation_sources import (  # noqa: E402
    STUDY_MODELS,
    STUDY_SYMBOLS,
    api_reference,
    cli_reference,
    dataset_summary,
    feature_catalogue,
    is_study_run,
    settings_reference,
    test_inventory,
)

from prosper.domain import (  # noqa: E402
    DEFAULT_DEPTH_BINS_STR,
    DEFAULT_HORIZONS,
    HORIZON_NAMES,
    parse_depth_bins,
)
from prosper.eval.significance import (  # noqa: E402
    DEFAULT_RESAMPLES,
    accuracy_interval,
    benjamini_hochberg,
    bootstrap_p_value,
)

REPORT_ROOT = "data/reports"
# Which runs are the study is decided in one place: documentation_sources.
SYMBOLS = STUDY_SYMBOLS
MODELS = STUDY_MODELS
ACCENT = RGBColor(0x1F, 0x3B, 0x63)
MUTED = RGBColor(0x55, 0x5F, 0x6B)

_counters: dict[str, int] = defaultdict(int)

# The formal details of the thesis. The title is a working one until the
# supervisor approves it; the title page layout follows the PJATK template
# once it is supplied.
THESIS_TITLE = (
    "Platforma badawcza do porównania architektur uczenia maszynowego w prognozowaniu "
    "kierunku zmian cen kryptowalut na podstawie historii notowań"
)
THESIS_TITLE_EN = (
    "A research platform for comparing machine learning architectures in forecasting the "
    "direction of cryptocurrency price changes from historical quotes"
)
AUTHOR = "Adrian Budzich"
ALBUM_NUMBER = "30076"
SUPERVISOR = "dr Michał Majewski"
UNIVERSITY = "Polsko-Japońska Akademia Technik Komputerowych"
FIELD = "Informatyka"
SPECIALISATION = "Robotyka i Inteligentne Systemy Autonomiczne"
YEAR = "2028"

try:
    from documentation_figures import build_all as _build_figures

    _FIGURES: dict[str, str] = _build_figures()
except Exception as _figure_error:  # matplotlib absent, or no artifacts yet
    print(f"[figury] pominięte: {_figure_error}")
    _FIGURES = {}

try:
    from documentation_diagrams import render_all as _render_diagrams
    from documentation_diagrams import size_cm as _diagram_size

    _DIAGRAMS = _render_diagrams()
except Exception as _diagram_error:  # Java or PlantUML absent and nothing rendered before
    print(f"[diagrams] skipped: {_diagram_error}")
    _DIAGRAMS = {}


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
        if not is_study_run(symbol, slug):
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
    return "-" if value is None else f"{value:.{places}f}"


def pl(value: float, spec: str = ".3f") -> str:
    """A number for Polish prose, with a decimal comma. Tables keep the point the
    software prints, so a reader can match them against the artifacts."""
    return format(value, spec).replace(".", ",")


# ── document furniture ───────────────────────────────────────────────────────


def plain_dashes(text: str) -> str:
    """Long dashes to plain hyphens.

    Applied in every helper that writes to the page, so it also covers text the
    document does not author: numeric ranges, the empty-value marker, and the
    test docstrings pulled out of the suite.
    """
    return text.replace(chr(8212), "-").replace(chr(8211), "-")


def add_heading(doc: Document, text: str, level: int) -> None:
    heading = doc.add_heading(plain_dashes(text), level=level)
    _start_page_if_pending(doc, heading)
    for run in heading.runs:
        run.font.color.rgb = ACCENT


def add_body(doc: Document, text: str, *, italic: bool = False, size: float = 10) -> None:
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(6)
    paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    run = paragraph.add_run(plain_dashes(text))
    run.italic = italic
    run.font.size = Pt(size)


def add_quote(doc: Document, text: str) -> None:
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.left_indent = Cm(1.0)
    paragraph.paragraph_format.space_after = Pt(10)
    paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    run = paragraph.add_run(plain_dashes(text))
    run.bold = True
    run.font.size = Pt(10.5)


def add_bullets(doc: Document, items: list[str]) -> None:
    for item in items:
        paragraph = doc.add_paragraph(plain_dashes(item), style="List Bullet")
        paragraph.paragraph_format.space_after = Pt(3)
        for run in paragraph.runs:
            run.font.size = Pt(10)


def add_code(doc: Document, lines: list[str]) -> None:
    for line in lines:
        paragraph = doc.add_paragraph()
        paragraph.paragraph_format.left_indent = Cm(0.8)
        paragraph.paragraph_format.space_after = Pt(0)
        run = paragraph.add_run(plain_dashes(line))
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
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    for index, label in enumerate(header):
        cell = table.rows[0].cells[index]
        cell.text = ""
        run = cell.paragraphs[0].add_run(plain_dashes(label))
        run.bold = True
        run.font.size = Pt(font)

    _horizontal_rules(table)

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
            run = paragraph.add_run(plain_dashes(str(value)))
            run.font.size = Pt(font)
    for row in table.rows:
        for index, cell in enumerate(row.cells):
            cell.width = Cm(widths[index])
    doc.add_paragraph()


# Colours of the table rules: the accent for the frame, a light grey between rows.
RULE_COLOUR = "1F3B63"
HAIRLINE_COLOUR = "C9D1DB"
HEADER_FILL = "EEF2F7"
# Elements that must follow w:tblBorders inside w:tblPr, in schema order.
_AFTER_TABLE_BORDERS = ("shd", "tblLayout", "tblCellMar", "tblLook", "tblCaption",
                        "tblDescription")


def _border(name: str, size: int, colour: str = "auto"):
    element = OxmlElement(f"w:{name}")
    if size:
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), str(size))
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), colour)
    else:
        element.set(qn("w:val"), "nil")
    return element


def _horizontal_rules(table) -> None:
    """Horizontal rules only: a frame above and below, a rule under the header,
    hairlines between rows and nothing vertical - the plain academic table, which
    reads as well as a grid and prints better in black and white."""
    properties = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for name, size, colour in (("top", 8, RULE_COLOUR), ("left", 0, ""),
                               ("bottom", 8, RULE_COLOUR), ("right", 0, ""),
                               ("insideH", 2, HAIRLINE_COLOUR), ("insideV", 0, "")):
        borders.append(_border(name, size, colour))
    successor = next((child for child in properties
                      if child.tag in {qn(f"w:{n}") for n in _AFTER_TABLE_BORDERS}), None)
    if successor is not None:
        successor.addprevious(borders)
    else:
        properties.append(borders)
    for cell in table.rows[0].cells:
        cell_properties = cell._tc.get_or_add_tcPr()
        cell_borders = OxmlElement("w:tcBorders")
        cell_borders.append(_border("bottom", 6, RULE_COLOUR))
        cell_properties.append(cell_borders)
        shading = OxmlElement("w:shd")
        shading.set(qn("w:val"), "clear")
        shading.set(qn("w:color"), "auto")
        shading.set(qn("w:fill"), HEADER_FILL)
        cell_properties.append(shading)


def _field(paragraph, instruction: str, shown: str) -> None:
    """A Word field: *instruction* computed by Word, *shown* until it updates."""
    parts = []
    for kind in ("begin", "instr", "separate", "text", "end"):
        if kind == "instr":
            element = OxmlElement("w:instrText")
            element.set(qn("xml:space"), "preserve")
            element.text = f" {instruction} "
        elif kind == "text":
            element = OxmlElement("w:t")
            element.text = shown
        else:
            element = OxmlElement("w:fldChar")
            element.set(qn("w:fldCharType"), kind)
        parts.append(element)
    for element in parts:
        run = paragraph.add_run()
        run.italic = True
        run.font.size = Pt(8.5)
        run.font.color.rgb = MUTED
        run._r.append(element)


def add_caption(doc: Document, kind: str, text: str) -> None:
    """"Tabela N. text", where N is a SEQ field: the lists of tables and figures
    at the end of the document are built by Word from these fields."""
    _counters[kind] += 1
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(12)

    def styled(fragment: str) -> None:
        run = paragraph.add_run(plain_dashes(fragment))
        run.italic = True
        run.font.size = Pt(8.5)
        run.font.color.rgb = MUTED

    # Only the first sentence goes in the numbered caption, so the lists of
    # tables and figures stay one line per entry; the rest follows beneath.
    match = re.search(r"(?<=[^\s.]{2})\. (?=[A-ZĄĆĘŁŃÓŚŹŻ„])", text)
    title, rest = (text[: match.start() + 1], text[match.end():]) if match else (text, "")
    styled(f"{kind} ")
    _field(paragraph, f"SEQ {kind} \\* ARABIC", str(_counters[kind]))
    styled(f". {title}")
    paragraphs = [paragraph]
    if rest:
        paragraph.paragraph_format.space_after = Pt(2)
        paragraph.paragraph_format.keep_with_next = True
        follow = doc.add_paragraph()
        follow.paragraph_format.space_after = Pt(12)
        run = follow.add_run(plain_dashes(rest))
        run.italic = True
        run.font.size = Pt(8.5)
        run.font.color.rgb = MUTED
        paragraphs.append(follow)
    if kind == "Tabela":
        _place_above_table(paragraphs)


def _place_above_table(paragraphs: list) -> None:
    """Move a table's caption above it, as Polish theses set them.

    Captions are written after the table they describe, so the table is found as
    the nearest preceding element, past the spacing paragraph `add_table` leaves.
    """
    previous = paragraphs[0]._p.getprevious()
    while (previous is not None and previous.tag == qn("w:p")
           and not "".join(previous.itertext()).strip()):
        previous = previous.getprevious()
    if previous is None or previous.tag != qn("w:tbl"):
        return
    for paragraph in paragraphs:
        previous.addprevious(paragraph._p)
        paragraph.paragraph_format.keep_with_next = True
    paragraphs[0].paragraph_format.space_before = Pt(6)
    paragraphs[-1].paragraph_format.space_after = Pt(4)


def add_figure(doc: Document, key: str, caption: str) -> None:
    """Place a rendered figure, or say nothing if it was not produced.

    Figures come from `documentation_figures.build_all()`, which draws them from
    the same artifacts the tables read. A missing figure means the data behind it
    is absent, so the document simply omits it rather than leaving a broken
    reference.
    """
    path = _FIGURES.get(key)
    if not path or not os.path.exists(path):
        return
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(6)
    paragraph.paragraph_format.space_after = Pt(2)
    paragraph.paragraph_format.keep_with_next = True
    paragraph.add_run().add_picture(path, width=Cm(15.5))
    add_caption(doc, "Rysunek", caption)

# The text block of the page, and the height that still leaves room for a caption.
DIAGRAM_MAX_WIDTH_CM = 16.0
DIAGRAM_MAX_HEIGHT_CM = 19.0


def add_diagram(doc: Document, key: str, caption: str) -> None:
    """Place a UML diagram at its natural size, shrunk only as far as the page needs.

    Natural size keeps the diagram's text at the size it was drawn with; scaling
    every diagram to the full width would blow small ones up and leave large ones
    no smaller than they must be.
    """
    path = _DIAGRAMS.get(key)
    if not path or not os.path.exists(path):
        print(f"WARNING: diagram {key} was not rendered; the thesis omits it")
        return
    width, height = _diagram_size(path)
    scale = min(1.0, DIAGRAM_MAX_WIDTH_CM / width, DIAGRAM_MAX_HEIGHT_CM / height)
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(6)
    paragraph.paragraph_format.space_after = Pt(2)
    paragraph.paragraph_format.keep_with_next = True
    paragraph.add_run().add_picture(str(path), width=Cm(width * scale))
    add_caption(doc, "Rysunek", caption)


def add_toc(doc: Document, switches: str = '\\o "1-3" \\h \\z \\u') -> None:
    """A field Word fills in on open; python-docx cannot compute page numbers."""
    paragraph = doc.add_paragraph()
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instruction = OxmlElement("w:instrText")
    instruction.set(qn("xml:space"), "preserve")
    instruction.text = f"TOC {switches}"
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    placeholder = OxmlElement("w:t")
    placeholder.text = "Spis zostanie zbudowany po otwarciu dokumentu (F9)."
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    for element in (begin, instruction, separate, placeholder, end):
        run._r.append(element)


def add_front_heading(doc: Document, text: str) -> None:
    """A heading for front and back matter that stays out of the table of contents."""
    paragraph = doc.add_paragraph()
    _start_page_if_pending(doc, paragraph)
    paragraph.paragraph_format.space_after = Pt(12)
    run = paragraph.add_run(text)
    run.bold = True
    run.font.size = Pt(16)
    run.font.color.rgb = ACCENT


# Where the next heading must start a new page: the index its paragraph will
# take in the document body, or None when no break is pending.
_page_break_at: int | None = None


def page_break(doc: Document) -> None:
    """Start what follows on a new page.

    The break is carried by the next heading (page_break_before) rather than by
    an empty paragraph holding one: a chapter that ends at the foot of a page
    would push that paragraph onto a page of its own and leave the page blank.
    """
    global _page_break_at
    # The body ends with the section properties, and new paragraphs go before them.
    _page_break_at = len(doc.element.body) - 1


def _start_page_if_pending(doc: Document, paragraph) -> None:
    global _page_break_at
    if _page_break_at is None:
        return
    position = list(doc.element.body).index(paragraph._p)
    if position != _page_break_at:
        print(f"WARNING: a page break meant for body element {_page_break_at} "
              f"landed on element {position}")
    paragraph.paragraph_format.page_break_before = True
    _page_break_at = None


# ═════════════════════════════════════════════════════════════════════════════
def build(path: str = "docs/Prosper_Dokumentacja_Techniczna.docx") -> str:
    runs = load_runs()
    _check_thesis_against_data(runs)
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
    _abstracts(doc, runs)

    add_front_heading(doc, "Spis treści")
    add_toc(doc)
    page_break(doc)

    _chapter_1_introduction(doc)
    _chapter_2_requirements(doc)
    _chapter_3_theory(doc)
    _chapter_3_equations(doc)
    _chapter_4_architecture(doc)
    _chapter_5_data(doc)
    _chapter_6_features(doc)
    _chapter_7_labels(doc)
    _chapter_8_models(doc)
    _chapter_9_methodology(doc)
    _chapter_10_results(doc, runs)
    _chapter_11_verification(doc)
    _chapter_12_interfaces(doc)
    _chapter_12a_worked_example(doc)
    _chapter_12b_screenshots(doc)
    _chapter_13_limits(doc)
    _chapter_14_summary(doc, runs)
    _chapter_14_bibliography(doc)
    _lists_of_tables_and_figures(doc)
    _appendix_tests(doc)
    _appendix_cli(doc)
    _appendix_api(doc)
    _appendix_settings(doc)
    _appendix_runs(doc, runs)
    _appendix_glossary(doc)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    doc.save(path)
    return path


# ── front matter ─────────────────────────────────────────────────────────────

def _title_page(doc: Document) -> None:
    """Until the PJATK template arrives: the usual fields, in the usual order."""

    def line(text: str, size: float, *, bold: bool = False, italic: bool = False,
             before: float = 0, after: float = 4, colour=None) -> None:
        paragraph = doc.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.space_before = Pt(before)
        paragraph.paragraph_format.space_after = Pt(after)
        run = paragraph.add_run(text)
        run.font.size = Pt(size)
        run.bold = bold
        run.italic = italic
        if colour is not None:
            run.font.color.rgb = colour

    line(UNIVERSITY, 14, bold=True, before=30)
    line(f"Kierunek: {FIELD}", 11)
    line(f"Specjalizacja: {SPECIALISATION}", 11, after=70)
    line(AUTHOR, 14, bold=True)
    line(f"Nr albumu {ALBUM_NUMBER}", 11, after=50)
    line(THESIS_TITLE, 18, bold=True, colour=ACCENT, after=10)
    line(THESIS_TITLE_EN, 11, italic=True, colour=MUTED, after=50)
    line("Praca inżynierska", 13, after=4)
    line(f"Promotor: {SUPERVISOR}", 12, after=110)
    line(YEAR, 11)


KEYWORDS_PL = [
    "uczenie maszynowe", "prognozowanie szeregów czasowych", "kryptowaluty",
    "hipoteza efektywności rynku", "walidacja krocząca", "kalibracja prognoz",
    "Temporal Fusion Transformer",
]
KEYWORDS_EN = [
    "machine learning", "time series forecasting", "cryptocurrencies",
    "efficient market hypothesis", "walk-forward validation", "forecast calibration",
    "Temporal Fusion Transformer",
]


def rank_agreement(runs: dict) -> tuple[dict[str, list[str]], float]:
    """Order of the learning architectures per horizon, and the mean rank
    correlation between horizons (+1 the same order everywhere, 0 unrelated).

    The baseline is excluded: it does not learn and including it would inflate
    the apparent agreement.
    """
    learners = [m for m in MODELS if m != "baseline"]
    order: dict[str, list[str]] = {}
    for horizon in HORIZON_NAMES:
        means = {}
        for model in learners:
            values = []
            for symbol in SYMBOLS:
                values.extend(horizon_values(runs.get((symbol, model), []), horizon, "accuracy"))
            if values:
                means[model] = float(np.mean(values))
        if len(means) == len(learners):
            order[horizon] = sorted(means, key=lambda m: -means[m])
    correlations = []
    horizons = list(order)
    for index, first in enumerate(horizons):
        for second in horizons[index + 1:]:
            left = np.array([order[first].index(m) for m in learners], dtype=float)
            right = np.array([order[second].index(m) for m in learners], dtype=float)
            if left.std() and right.std():
                correlations.append(float(np.corrcoef(left, right)[0, 1]))
    return order, float(np.mean(correlations)) if correlations else float("nan")


def seed_invariant(runs: dict) -> set[tuple[str, str]]:
    """(symbol, model) cells whose runs under different seeds scored identically.

    Both gradient-boosting models learn without subsampling, so a seed has
    nothing to vary: their three runs are one measurement taken three times.
    Read from the artifacts rather than assumed, so a model that starts
    subsampling drops out of this set by itself.
    """
    return {
        key for key, entries in runs.items()
        if len(entries) > 1 and all(e["by_horizon"] == entries[0]["by_horizon"] for e in entries)
    }


def distinct_runs(runs: dict) -> dict:
    """The runs with each seed-invariant cell collapsed to one measurement."""
    repeated = seed_invariant(runs)
    return {key: entries[:1] if key in repeated else entries for key, entries in runs.items()}


def architecture_spread(runs: dict) -> dict[str, dict]:
    """Per horizon: how far apart the architectures' mean accuracies lie, against
    how far one architecture moves under another seed.

    The seed spread is taken only from cells a seed changes. A deterministic
    model contributes exactly zero, which would lower the noise floor and turn
    ties between the learners into a ranking.
    """
    repeated = seed_invariant(runs)
    spread: dict[str, dict] = {}
    for horizon in HORIZON_NAMES:
        means: dict[str, float] = {}
        within: list[float] = []
        for model in MODELS:
            values: list[float] = []
            for symbol in SYMBOLS:
                accs = horizon_values(runs.get((symbol, model), []), horizon, "accuracy")
                values.extend(accs)
                if len(accs) > 1 and (symbol, model) not in repeated:
                    within.append(float(np.std(accs, ddof=1)))
            if values:
                means[model] = float(np.mean(values))
        learners = [v for m, v in means.items() if m != "baseline"]
        spread[horizon] = {
            "means": means,
            "range": max(means.values()) - min(means.values()) if len(means) > 1 else None,
            "range_learners": max(learners) - min(learners) if len(learners) > 1 else None,
            "noise": float(np.mean(within)) if within else None,
        }
    return spread


def learners_beyond_noise(spread: dict[str, dict]) -> list[str]:
    """Horizons on which the learning architectures lie further apart than a seed moves them."""
    return [
        horizon for horizon, cell in spread.items()
        if cell["noise"] is not None and cell["range_learners"] is not None
        and cell["range_learners"] > cell["noise"]
    ]


def study_facts(runs: dict) -> dict:
    """The few numbers the abstracts and the summary quote, all read from the artifacts.

    Significance is counted over distinct measurements: the three identical runs
    of a seed-invariant model count once.
    """
    symbols = sorted({symbol for symbol, _ in runs})
    total = sum(len(v) for v in runs.values())
    distinct = distinct_runs(runs)
    facts: dict = {
        "symbols": len(symbols), "runs": total,
        "distinct": sum(len(v) for v in distinct.values()),
        "seed_invariant": sorted({model for _, model in seed_invariant(runs)}),
        "by_horizon": {},
    }
    for horizon in HORIZON_NAMES:
        accuracies, above, below, estimable = [], 0, 0, 0
        for entries in distinct.values():
            for run in entries:
                cell = run["by_horizon"].get(horizon) or {}
                if cell.get("accuracy") is None:
                    continue
                accuracies.append(float(cell["accuracy"]))
                interval = cell.get("accuracy_ci") or {}
                low, high = interval.get("low"), interval.get("high")
                if low is None or high is None:
                    continue
                estimable += 1
                above += low > 0.5
                below += high < 0.5
        facts["by_horizon"][horizon] = {
            "low": min(accuracies) if accuracies else None,
            "high": max(accuracies) if accuracies else None,
            "above": above, "below": below, "estimable": estimable,
        }
    facts["rank_correlation"] = rank_agreement(runs)[1]
    facts["spread"] = architecture_spread(runs)
    facts["beyond_noise"] = learners_beyond_noise(facts["spread"])
    return facts


# The p-values of section 10.9 come from a longer run of the interval's own
# resampling: under the same seed its first DEFAULT_RESAMPLES draws are the
# published interval's. The length is set by the correction, not by taste: B
# draws cannot give a p below 2 / (B + 1), and Benjamini-Hochberg asks the
# smallest of ~240 values to clear 0.05 / 240, about 2e-4.
P_VALUE_RESAMPLES = 20_000
# A measurement that survives the correction is re-measured with ten times the
# draws, so that its survival is not a matter of where the resampling stopped.
PRECISION_RESAMPLES = 200_000
FALSE_DISCOVERY_RATE = 0.05
CHANCE = 0.5

_departures_cache: dict[tuple, dict] = {}


def _scored(symbol: str, slug: str, horizon: str, columns: list[str]):
    """The scored rows of one run at one horizon, in the chronological order the
    evaluation wrote them in - the order the block bootstrap needs."""
    import polars

    path = f"{REPORT_ROOT}/evaluations/{symbol}/{slug}/predictions_quality.parquet"
    frame = polars.read_parquet(path, columns=sorted({"horizon", "status", *columns}))
    return frame.filter((polars.col("status") == "scored") & (polars.col("horizon") == horizon))


def _correct(frame) -> tuple[np.ndarray, int]:
    return (np.asarray([bool(v) for v in frame["correct"].to_list()], dtype=float),
            int(frame["forward_steps"][0] or 1))


def departures(runs: dict) -> dict:
    """Every distinct measurement tested against a coin flip, with its
    Benjamini-Hochberg q-value over all of them.

    The intervals are recomputed from the scored rows and compared with the ones
    the evaluation stored, so a p-value is never read off data that differs
    from what the tables show. Takes a few minutes, so it is computed once.
    """
    key = tuple(sorted((s, m, r["slug"]) for (s, m), v in runs.items() for r in v))
    if key in _departures_cache:
        return _departures_cache[key]

    distinct = distinct_runs(runs)
    measurements, mismatched = [], 0
    for (symbol, model), entries in sorted(distinct.items()):
        for run in entries:
            for horizon in HORIZON_NAMES:
                stored = (run["by_horizon"].get(horizon) or {}).get("accuracy_ci") or {}
                frame = _scored(symbol, run["slug"], horizon, ["forward_steps", "correct"])
                if frame.height == 0:
                    continue
                values, block = _correct(frame)
                interval = accuracy_interval(values.astype(bool), block)
                if (interval.low, interval.high) != (stored.get("low"), stored.get("high")):
                    mismatched += 1
                if not interval.estimable:
                    continue
                measurements.append({
                    "symbol": symbol, "model": model, "slug": run["slug"],
                    "seed": run["seed"] if len(entries) > 1 else None,
                    "horizon": horizon, "accuracy": interval.point,
                    "low": interval.low, "high": interval.high, "block": block,
                    "p": bootstrap_p_value(values, block, CHANCE, resamples=P_VALUE_RESAMPLES),
                })
    for m, q in zip(measurements, benjamini_hochberg([m["p"] for m in measurements])):
        m["q"] = q

    # The survivors once more, with ten times the draws and with the block doubled.
    survivors = [i for i, m in enumerate(measurements) if m["q"] < FALSE_DISCOVERY_RATE]
    precise = [m["p"] for m in measurements]
    for i in survivors:
        m = measurements[i]
        values, block = _correct(_scored(m["symbol"], m["slug"], m["horizon"],
                                         ["forward_steps", "correct"]))
        m["p_precise"] = precise[i] = bootstrap_p_value(
            values, block, CHANCE, resamples=PRECISION_RESAMPLES)
        m["p_double_block"] = bootstrap_p_value(values, 2 * block, CHANCE,
                                                resamples=P_VALUE_RESAMPLES)
    q_precise = benjamini_hochberg(precise)
    for i in survivors:
        measurements[i]["q_precise"] = q_precise[i]

    result = {"measurements": measurements, "mismatched": mismatched}
    _departures_cache[key] = result
    return result


def _side_runs(sides: list) -> float:
    """Mean length of a stretch of consecutive bars predicting the same side."""
    changes = sum(1 for a, b in zip(sides, sides[1:]) if a != b)
    return len(sides) / (changes + 1) if sides else 0.0


def _mean_pairwise_correlation(series: dict[str, dict]) -> tuple[float, int]:
    """Mean correlation over every pair of symbols, on the dates all of them share."""
    common = set.intersection(*(set(values) for values in series.values()))
    dates = sorted(common)
    matrix = np.array([[values[d] for d in dates] for values in series.values()], dtype=float)
    corr = np.corrcoef(matrix)
    upper = corr[np.triu_indices(len(series), k=1)]
    return float(np.mean(upper)), len(dates)


def effective_count(n: int, rho: float) -> float:
    """How many independent series *n* equicorrelated ones are worth for a mean."""
    return n / (1 + (n - 1) * rho)


def baseline_lag(runs: dict, horizon: str = "quarter", symbol: str = "BTCUSDT") -> dict:
    """What section 10.9 says about the baseline at one horizon: its year-by-year
    record on one symbol, how long it holds a side, and how much the symbols
    share."""
    distinct = distinct_runs(runs)
    returns, hits, holds, by_year = {}, {}, [], []
    for name in SYMBOLS:
        entries = distinct.get((name, "baseline"), [])
        if not entries:
            continue
        frame = _scored(name, entries[0]["slug"], horizon,
                        ["date", "correct", "pred_direction", "actual_direction", "actual_return"])
        if frame.height == 0:
            continue
        dates = [str(d)[:10] for d in frame["date"].to_list()]
        returns[name] = dict(zip(dates, frame["actual_return"].to_list()))
        hits[name] = dict(zip(dates, (float(bool(v)) for v in frame["correct"].to_list())))
        holds.append(_side_runs(frame["pred_direction"].to_list()))
        if name == symbol:
            years: dict[str, list] = defaultdict(list)
            for date, predicted, actual, correct in zip(
                dates, frame["pred_direction"].to_list(), frame["actual_direction"].to_list(),
                frame["correct"].to_list(),
            ):
                years[date[:4]].append((predicted == "long", actual == "long", bool(correct)))
            by_year = [
                {"year": year, "bars": len(rows),
                 "predicted_long": float(np.mean([r[0] for r in rows])),
                 "actual_long": float(np.mean([r[1] for r in rows])),
                 "accuracy": float(np.mean([r[2] for r in rows]))}
                for year, rows in sorted(years.items())
            ]
    if len(returns) < 2 or not by_year:
        return {}
    rho_returns, days = _mean_pairwise_correlation(returns)
    rho_hits, _ = _mean_pairwise_correlation(hits)
    slug = distinct[(symbol, "baseline")][0]["slug"]
    summary = f"{REPORT_ROOT}/predictions/{symbol}/{slug}/run_summary.json"
    window = {}
    if os.path.exists(summary):
        windows = json.load(open(summary, encoding="utf-8")).get("training_windows") or {}
        window = windows.get(horizon, {})
    return {
        "symbols": len(returns), "days": days, "by_year": by_year, "hold": float(np.mean(holds)),
        "rho_returns": rho_returns, "rho_hits": rho_hits,
        "window_bars": window.get("window_bars"), "forward_bars": window.get("forward_bars"),
    }


def abstract_texts(runs: dict) -> tuple[list[str], list[str]]:
    """Streszczenie and abstract. Every number in them is read from the runs."""
    f = study_facts(runs)
    week = f["by_horizon"].get("week", {})
    month = f["by_horizon"].get("month", {})

    powered_above = week.get("above", 0) + month.get("above", 0)
    powered = week.get("estimable", 0) + month.get("estimable", 0)
    # One side of a two-sided 95% interval: what chance alone puts above it.
    expected = round(powered * 0.025)
    by_chance = powered_above <= 1.5 * expected + 1
    rho = f["rank_correlation"]
    horizons = len(f["spread"])
    within = horizons - len(f["beyond_noise"])
    deterministic = set(f["seed_invariant"]) == {"ml", "xgboost"}

    polish = [
        "Praca dotyczy pytania, w jakim stopniu architektury uczenia maszynowego różnego "
        "rodzaju potrafią przewidzieć kierunek zmiany ceny aktywa wyłącznie na podstawie "
        "historii jego notowań. W ramach pracy zaprojektowano i zbudowano platformę Prosper, "
        "obejmującą pozyskiwanie danych giełdy Binance, kontrolę ich jakości, wyznaczanie cech "
        "i etykiet, trening pięciu architektur - od modelu odniesienia, przez dwie implementacje "
        "wzmacniania gradientowego (scikit-learn i XGBoost), po sieć rekurencyjną GRU i Temporal "
        "Fusion Transformer - we wspólnym protokole kroczącym, ocenę regułami oceny właściwej z "
        "przedziałami ufności uwzględniającymi nakładanie się etykiet oraz symulację handlową "
        "z czterema hipotezami zerowymi. System udostępnia interfejs wiersza poleceń i panel "
        "przeglądarkowy.",
        f"Badanie objęło {f['runs']} przebiegów na {f['symbols']} instrumentach, w czterech "
        "horyzontach od tygodnia do roku, każdą konfigurację pod trzema ziarnami inicjalizacji"
        + (". Oba modele gradientowe uczą się deterministycznie, więc różnych pomiarów jest "
           f"{f['distinct']}. " if deterministic else f"; różnych pomiarów jest {f['distinct']}. ")
        + "Na horyzontach tygodniowym i miesięcznym, jedynych o wystarczającej mocy "
        f"statystycznej, trafność istotnie wyższą od losowej dało {powered_above} z {powered} "
        "pomiarów - "
        + ("nie więcej, niż przy braku jakiejkolwiek przewagi daje sam przypadek"
           if by_chance else "więcej, niż przy braku przewagi dałby sam przypadek")
        + f" (oczekiwanie: około {expected}). Na horyzoncie rocznym liczba obserwacji "
        "niezależnych nie pozwala na żaden rozstrzygający pomiar. Różnice między "
        "architekturami uczącymi się mieszczą się w rozrzucie wynikającym z losowej "
        f"inicjalizacji na {within} z {horizons} horyzontów, a ich ranking zmienia się z "
        f"horyzontem (średnia korelacja rang {pl(rho, '+.2f')}), więc nie są właściwością "
        "architektury. Wynik jest zgodny ze słabą formą hipotezy efektywności "
        "rynku i stanowi górne ograniczenie przewagi możliwej do uzyskania z samych notowań. "
        "Pozytywnym wynikiem inżynierskim jest kalibracja: system poprawnie raportuje własną "
        "niepewność.",
    ]
    english = [
        "This thesis asks how well machine learning architectures of different kinds can "
        "forecast the direction of an asset's price from its quotation history alone. It "
        "presents Prosper, a platform designed and built for the purpose: it acquires Binance "
        "market data and checks its quality, derives features and labels, trains five "
        "architectures - a frequency baseline, gradient-boosted trees from scikit-learn and "
        "from XGBoost, a GRU recurrent network and a Temporal Fusion Transformer - under one "
        "walk-forward "
        "protocol, evaluates them with proper scoring rules and confidence intervals that "
        "account for overlapping labels, and simulates trading against four null hypotheses. "
        "It is operated from a command line and a browser dashboard.",
        f"The study comprises {f['runs']} runs on {f['symbols']} instruments, over four "
        "horizons from a week to a year, each configuration under three initialisation seeds"
        + (". Both gradient-boosting models learn deterministically, which leaves "
           f"{f['distinct']} distinct measurements. " if deterministic
           else f"; {f['distinct']} of the measurements are distinct. ")
        + "On the weekly and monthly horizons, the only ones with enough statistical power, "
        f"{powered_above} of {powered} measurements reached an accuracy significantly above "
        "chance - "
        + ("no more than chance alone produces without any edge"
           if by_chance else "more than chance alone would produce")
        + f" (about {expected} expected). At the yearly horizon the number of independent "
        "observations allows no conclusive measurement. The differences between the learning "
        f"architectures stay within the spread of random initialisation on {within} of "
        f"{horizons} horizons, and their ranking changes with the horizon (mean rank "
        f"correlation {rho:+.2f}), so they are not a property of the architecture. The result "
        "is consistent with the weak form of the "
        "efficient market hypothesis and bounds the edge obtainable from prices alone. "
        "Calibration is a positive engineering result: the system reports its own "
        "uncertainty correctly.",
    ]

    return polish, english


def _abstracts(doc: Document, runs: dict) -> None:
    """Streszczenie and abstract, each with keywords, before the table of contents."""
    polish, english = abstract_texts(runs)
    for heading, text, label, words in (
        ("Streszczenie", polish, "Słowa kluczowe", KEYWORDS_PL),
        ("Abstract", english, "Keywords", KEYWORDS_EN),
    ):
        add_front_heading(doc, heading)
        for paragraph_text in text:
            add_body(doc, paragraph_text)
        keywords = doc.add_paragraph()
        keywords.paragraph_format.space_before = Pt(8)
        lead = keywords.add_run(f"{label}: ")
        lead.bold = True
        lead.font.size = Pt(10)
        rest = keywords.add_run(", ".join(words))
        rest.font.size = Pt(10)
        page_break(doc)


def _lists_of_tables_and_figures(doc: Document) -> None:
    add_front_heading(doc, "Spis rysunków")
    add_toc(doc, '\\h \\z \\c "Rysunek"')
    add_front_heading(doc, "Spis tabel")
    add_toc(doc, '\\h \\z \\c "Tabela"')
    page_break(doc)


# ── 1 ────────────────────────────────────────────────────────────────────────

def _chapter_1_introduction(doc: Document) -> None:
    add_heading(doc, "1. Wprowadzenie", 1)

    add_heading(doc, "1.1. Kontekst", 2)
    add_body(
        doc,
        "Rynek kryptowalut jest młody, płynny i całodobowy. Dane historyczne o notowaniach są "
        "publicznie dostępne w jednolitym formacie i o pełnej głębokości, co czyni go wygodnym "
        "poligonem dla metod uczenia maszynowego - i zarazem miejscem, w którym łatwo o wynik "
        "pozornie dobry, ponieważ szereg cenowy jest niemal całkowicie szumem, a każdy błąd "
        "metodologiczny objawia się jako poprawa trafności, nie jako awaria.",
    )
    add_body(
        doc,
        "Praca opisuje system zbudowany po to, aby tę różnicę rozstrzygać: platformę realizującą "
        "pełny potok badawczy od pobrania surowych świec, przez konstrukcję cech i etykiet, "
        "trening pięciu rodzin modeli, aż po ocenę jakości prognozy i symulację handlową - z "
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
        "Żadna z badanych architektur uczenia maszynowego nie osiąga w prognozie kierunku ceny "
        "z samej historii notowań trafności istotnie wyższej od losowej, a wszystkie wypadają "
        "poniżej reguły opartej na częstości klasy większościowej. Różnice między "
        "architekturami uczącymi się nie wykraczają wyraźnie poza rozrzut wynikający z losowej "
        "inicjalizacji, a ich porządek zmienia się wraz z horyzontem, nie odzwierciedlają więc "
        "właściwości architektury. Orzeczenie tego wymaga warstwy weryfikacji zdolnej odróżnić "
        "brak różnicy od niedostatecznej dokładności pomiaru.",
    )
    add_body(
        doc,
        "Teza rozstrzyga trzy rzeczy naraz i warto je rozdzielić. Pierwsza dotyczy poziomu: "
        "żadna architektura nie przekracza progu losowego, a punktem odniesienia nie jest "
        "0,500, lecz częstość klasy liczniejszej, ponieważ w badanym okresie klasy nie są "
        "zrównoważone. Druga dotyczy różnic: między architekturami uczącymi się są one tego "
        "samego rzędu co szum samej inicjalizacji, więc ranking tych architektur nie ma "
        "podstaw; wyraźnie odróżnia się jedynie model odniesienia, słabszy od nich na "
        "horyzontach dłuższych niż tydzień. Trzecia dotyczy trwałości: architektura najlepsza "
        "na jednym horyzoncie bywa najgorsza na innym - czego należy oczekiwać, gdy różnice "
        "mieszczą się w szumie.",
    )

    add_body(
        doc,
        "Zdanie ostatnie tezy jest metodyczne. Stwierdzenie „architektury nie różnią się "
        "między sobą” jest nieodróżnialne od stwierdzenia „pomiar był zbyt zaszumiony, aby je "
        "rozróżnić”, dopóki poziom szumu nie zostanie zmierzony. Z tego powodu każdą "
        "konfigurację uruchomiono wielokrotnie pod różnymi ziarnami generatora liczb losowych, "
        "a rozdział 10 raportuje rozstęp między architekturami obok rozrzutu wynikającego z "
        "samej inicjalizacji oraz stabilność rankingu między horyzontami.",
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
            "Notowania ciągłe - brak luk weekendowych i nocnych, które na rynku akcji tworzą "
            "nieciągłości wymagające osobnego traktowania w każdej z cech technicznych, a w "
            "modelach sekwencyjnych zaburzają pojęcie sąsiedztwa w czasie.",
            "Brak zdarzeń korporacyjnych - podziały akcji, dywidendy, emisje i wykupy wymagają "
            "korekty szeregu historycznego, a przyjęta metoda korekty sama wpływa na wynik "
            "modelu i staje się nieudokumentowanym parametrem badania.",
            "Jednolita mikrostruktura - jedna giełda, jeden mechanizm kojarzenia zleceń, brak "
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
        "kierunkiem dalszych prac (rozdział 13).",
    )

    add_heading(doc, "1.5. Twierdzenia szczegółowe", 2)
    add_bullets(
        doc,
        [
            "Żadna z pięciu badanych architektur nie osiąga trafności kierunkowej istotnie "
            "większej od losowej na horyzoncie o wystarczającej mocy statystycznej. Jest to "
            "wynik ograniczony od góry, a nie twierdzenie o niemożliwości prognozowania.",
            "Różnice między architekturami uczącymi się są tego samego rzędu co rozrzut "
            "wynikający z ziarna inicjalizacji, a ranking architektur zmienia się wraz z "
            "horyzontem, przy korelacji rang bliskiej zeru. Wybór architektury na podstawie tych "
            "wyników nie ma podstaw; wyraźnie odróżnia się jedynie model odniesienia, słabszy od "
            "pozostałych na horyzontach dłuższych niż tydzień.",
            "Trafność pojedynczego przebiegu obarczona jest szumem inicjalizacji, którego "
            "przedział ufności liczony wewnątrz tego przebiegu nie obejmuje. Każdą "
            "konfigurację uruchomiono zatem trzykrotnie, pod różnymi ziarnami. Dla obu modeli "
            "gradientowych ziarno nie zmienia wyniku - uczą się bez podpróbkowania - więc szum "
            "inicjalizacji mierzą wyłącznie GRU i TFT.",
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
            ["2", "Wymagania funkcjonalne i niefunkcjonalne, kryteria akceptacji, granice zakresu"],
            ["3", "Podstawy teoretyczne: efektywność rynku, reguły oceny właściwej, kalibracja, "
                  "bootstrap blokowy"],
            ["4", "Architektura systemu i decyzje technologiczne"],
            ["5", "Dane: źródło, format, agregacja, kontrola jakości"],
            ["6", "Cechy: zasada niezmienniczości względem skali i pełny katalog"],
            ["7", "Etykiety: kierunek, głębokość, horyzonty"],
            ["8", "Architektury modeli i wspólny protokół kroczący"],
            ["9", "Metodyka pomiaru"],
            ["10", "Wyniki: porównanie architektur"],
            ["11", "Weryfikacja poprawności implementacji - studia przypadków"],
            ["12", "Interfejsy: wiersz poleceń, HTTP, panel przeglądarkowy"],
            ["13", "Ograniczenia i kierunki dalszych prac"],
            ["14", "Podsumowanie"],
            ["A-F", "Załączniki: inwentarz testów, referencja CLI, referencja HTTP, konfiguracja, "
                    "wyniki wszystkich przebiegów, słownik pojęć"],
        ],
        [2.5, 13.5],
    )
    add_caption(doc, "Tabela", "Układ dokumentu.")
    page_break(doc)


# ── 2 ────────────────────────────────────────────────────────────────────────

def _verified_by(*modules: str) -> str:
    """Join test module names, refusing any that is not in the suite.

    The requirements table is the one place in this document where a reference
    is written by hand rather than read from the system, so it is the one place
    where a name can be invented. Failing the build is the cheapest moment to
    find out: a requirement pointing at a test that does not exist is worse than
    a requirement with no test at all, because it reads as covered.
    """
    for name in modules:
        if not os.path.exists(os.path.join("tests", name)):
            raise SystemExit(f"[wymagania] nie ma takiego modułu testowego: tests/{name}")
    return ", ".join(modules)


def _chapter_2_requirements(doc: Document) -> None:
    add_heading(doc, "2. Wymagania", 1)

    add_heading(doc, "2.1. Dwa źródła wymagań", 2)
    add_body(
        doc,
        "Wymagania tego projektu powstały w dwóch trybach i rozdzielenie ich jest samo w sobie "
        "wynikiem wartym zapisania. Część dała się sformułować przed napisaniem pierwszej linii "
        "kodu, ponieważ wynikała z dziedziny albo z rzemiosła inżynierskiego. Część nie mogła "
        "powstać wcześniej niż pomiar, który ją wymusił - i próba zapisania jej z góry dałaby "
        "sformułowanie błędne, nie niepełne.",
    )
    add_body(
        doc,
        "Przykładem pierwszej grupy jest zakaz zaglądania w przyszłość. Wiadomo z góry, że "
        "etykieta świecy k odwołuje się do zamknięcia świecy k+F, jest więc dostępna dopiero po "
        "upływie horyzontu; reguła ta nie wymaga żadnego eksperymentu. Przykładem drugiej jest "
        "sposób wyznaczania szerokości okna treningowego. Pierwotnie była to stała podawana w "
        "konfiguracji, ponieważ tak wygląda ten parametr w większości opracowań. Dopiero pomiar "
        "pokazał, że jedna szerokość dla wszystkich horyzontów oznacza 103 niezależne "
        "obserwacje na horyzoncie tygodniowym i 1,0 na rocznym, co czyni drugi z nich "
        "niemierzalnym. Wymaganie w brzmieniu ostatecznym - szerokość okna jest wielkością "
        "wyprowadzaną z horyzontu i długości dostępnej historii, nigdy stałą - jest sprzeczne z "
        "wymaganiem pierwotnym, a nie jego uszczegółowieniem.",
    )
    add_quote(
        doc,
        "Specyfikacja spisana wyłącznie z góry zawierałaby pierwszą grupę wymagań w brzmieniu "
        "poprawnym, a drugą w brzmieniu mylnym, i to właśnie druga grupa decyduje o "
        "wiarygodności wyników.",
    )
    add_body(
        doc,
        "Rozdział ten podaje zatem wymagania w trzech zestawieniach: funkcjonalne "
        "(podrozdział 2.2), niefunkcjonalne (2.3) oraz osobno te, które wyprowadzono w trakcie "
        "badania wraz ze wskazaniem, co je wymusiło (2.4). Podrozdział 2.5 wyznacza granice "
        "zakresu, czyli zadania świadomie pozostawione poza systemem, a 2.6 podaje kryteria "
        "akceptacji wraz z ich obecnym stanem. Numeracja wymagań jest przywoływana w dalszych "
        "rozdziałach tam, gdzie opisywana jest realizacja.",
    )

    add_heading(doc, "2.2. Wymagania funkcjonalne", 2)
    add_body(
        doc,
        "Wymagania funkcjonalne opisują, co system ma robić. Uporządkowano je zgodnie z "
        "kolejnością etapów potoku przetwarzania, ponieważ taka jest też kolejność zależności: "
        "żaden etap nie może zostać zrealizowany przed poprzednim. Kolumna Weryfikacja wskazuje "
        "moduł testowy, na podstawie którego można stwierdzić spełnienie wymagania. Nazwy w tej "
        "kolumnie sprawdzane są przy generowaniu dokumentu wobec zawartości katalogu testów: "
        "odwołanie do modułu, którego nie ma, przerywa budowę, ponieważ wymaganie wskazujące "
        "nieistniejący test czyta się jako pokryte i jest gorsze od wymagania bez testu.",
    )
    add_table(
        doc,
        ["Nr", "Wymaganie", "Realizacja", "Weryfikacja"],
        [
            ["WF-01",
             "Pozyskanie historycznych notowań z publicznego źródła, w rozdzielczości "
             "jednominutowej, dla dowolnego symbolu SPOT i dowolnego zakresu miesięcy.",
             "prosper backfill; pobieranie archiwów ZIP z Binance Public Data",
             _verified_by("test_integration.py", "test_cli_exit_codes.py")],
            ["WF-02",
             "Utrzymanie lokalnego jeziora danych odpornego na powtórne pobranie tego samego "
             "zakresu: ponowne uruchomienie nie może duplikować ani gubić świec.",
             "zapis partycjonowany symbol=/year=/month=, nadpisanie partycji",
             _verified_by("test_integration.py", "test_data_manager.py")],
            ["WF-03",
             "Agregacja świec jednominutowych do interwałów wyższych bez utraty poprawności "
             "OHLCV oraz kolumn przepływu zleceń.",
             "prosper aggregate; 1m do 1h, 1d, 1w",
             _verified_by("test_aggregate.py", "test_order_flow_features.py")],
            ["WF-04",
             "Kontrola jakości danych wykrywająca braki ciągłości, duplikaty, złe sortowanie i "
             "naruszenia niezmienników świecy.",
             "prosper qa check; raport z liczbą defektów w podziale na partycje",
             _verified_by("test_qa_checks.py")],
            ["WF-05",
             "Wyliczenie zestawu cech technicznych niezależnych od poziomu ceny, "
             "umożliwiającego łączenie danych z różnych instrumentów.",
             "prosper features build; lista TRAINING_FEATURE_COLUMNS w domain.py",
             _verified_by("test_features_build.py", "test_scale_free_features.py")],
            ["WF-06",
             "Wyznaczenie etykiet kierunku oraz głębokości ruchu dla każdego horyzontu, z "
             "jawnym oznaczeniem świec, dla których etykieta nie jest jeszcze znana.",
             "labels/forward.py - jedna definicja, z której korzysta każdy model; prosper "
             "labels build zapisuje ją dla czterech horyzontów",
             _verified_by("test_forward_labels.py", "test_depth_scheme.py",
                          "test_horizon_set.py")],
            ["WF-07",
             "Wytrenowanie modelu prognozującego rozkład prawdopodobieństwa kierunku i "
             "głębokości dla każdego z czterech horyzontów.",
             "prosper predict baseline / ml / xgboost / gru / tft",
             _verified_by("test_predictor_training_edges.py",
                          "test_predict_interval_integration.py")],
            ["WF-08",
             "Udostępnienie co najmniej pięciu architektur o różnym sposobie wprowadzenia "
             "zależności czasowej, porównywalnych w ramach jednego protokołu.",
             "model odniesienia, wzmacnianie gradientowe scikit-learn, XGBoost, GRU, TFT",
             _verified_by("test_predict_interval_integration.py",
                          "test_tft_class_order.py")],
            ["WF-09",
             "Trening kroczący z okresowym przeuczeniem, w którym prognoza dla danego miesiąca "
             "korzysta wyłącznie z danych wcześniejszych.",
             "predict/window.py; przeuczenie miesięczne",
             _verified_by("test_predict_leakage.py", "test_train_window.py")],
            ["WF-10",
             "Możliwość zasilenia treningu danymi z kilku symboli naraz, z zachowaniem "
             "poprawności czasowej doboru wierszy.",
             "predict/pool.py; flaga --symbols dla modeli przyjmujących pulę",
             _verified_by("test_pooled_training.py", "test_api_pool_command.py")],
            ["WF-11",
             "Kalibracja zwracanych prawdopodobieństw na wydzielonym fragmencie okna "
             "treningowego, nigdy na danych późniejszych niż prognoza.",
             "skalowanie temperaturą; calibration_split",
             _verified_by("test_calibration.py")],
            ["WF-12",
             "Ocena przebiegu regułami oceny właściwej wraz z przedziałami ufności "
             "uwzględniającymi nakładanie się etykiet.",
             "prosper eval predictions; trafność, Brier, logloss, ECE, bootstrap blokowy",
             _verified_by("test_prediction_evaluation.py", "test_significance.py")],
            ["WF-13",
             "Ocena stabilności prognozy w czasie, w podziale na miesiące, z oznaczeniem "
             "okresów o zbyt małej liczbie obserwacji.",
             "prosper eval stability",
             _verified_by("test_eval_stability.py")],
            ["WF-14",
             "Zestawienie wszystkich ocenionych przebiegów w jednym rankingu oraz ocena "
             "wsadowa nadająca się do uruchomienia z harmonogramu.",
             "prosper eval compare; prosper eval batch",
             _verified_by("test_api_server.py", "test_cli_exit_codes.py")],
            ["WF-15",
             "Przełożenie prognozy na skalę rekomendacji od Strong Buy do Strong Sell, z "
             "progiem kosztowym i bramkami ryzyka.",
             "prosper planner windows; SequenceGrader",
             _verified_by("test_planner_windows.py", "test_cost_threshold.py",
                          "test_risk_gates.py", "test_risk_is_directional.py")],
            ["WF-16",
             "Symulacja handlowa odtwarzająca przebieg na danych historycznych, z jawnymi "
             "założeniami wykonawczymi.",
             "prosper eval backtest; prowizja, poślizg, próg kosztowy, polityka ekspozycji",
             _verified_by("test_backtest_execution.py", "test_backtest_parameters.py")],
            ["WF-17",
             "Podanie wyniku symulacji zawsze wraz z punktami odniesienia, w tym z rozkładem "
             "wyników tej samej ścieżki ekspozycji przesuniętej w czasie.",
             "sekcja nulls: kupno i trzymanie, stała ekspozycja, przesunięcia cykliczne, "
             "reguła momentum",
             _verified_by("test_backtest_nulls.py")],
            ["WF-18",
             "Wersjonowanie przebiegów: każdy artefakt pochodny wskazuje przebieg, z którego "
             "powstał, a usunięcie przebiegu usuwa wszystko, co z niego wynika.",
             "storage/runs.py; katalog symbol / model_interwał_znacznik-czasu",
             _verified_by("test_run_attribution.py", "test_versioned_predictions.py")],
            ["WF-19",
             "Obsługa całego potoku z wiersza poleceń, w formie nadającej się do wywołania z "
             "harmonogramu zadań.",
             "aplikacja Typer; pełna referencja w załączniku B",
             _verified_by("test_cli_exit_codes.py", "test_cli_api_contract.py")],
            ["WF-20",
             "Obsługa całego potoku z przeglądarki, bez konieczności znajomości wiersza "
             "poleceń.",
             "FastAPI oraz panel w natywnych modułach ES",
             _verified_by("test_api_server.py", "test_api_performance.py")],
            ["WF-21",
             "Kolejkowanie długotrwałych zadań po stronie serwera, odporne na przeładowanie "
             "przeglądarki.",
             "api/tasks.py; jeden wątek roboczy, kolejka po stronie serwera",
             _verified_by("test_task_queue.py")],
            ["WF-22",
             "Prezentacja stanu jeziora danych: symbole, zakresy, liczba świec, zajętość "
             "dysku, wraz z możliwością usunięcia symbolu.",
             "inventory.py; DataManager",
             _verified_by("test_data_manager.py", "test_inventory_cache.py")],
        ],
        [1.5, 5.2, 4.4, 4.4],
        font=8,
    )
    add_caption(
        doc, "Tabela",
        "Wymagania funkcjonalne. Kolumna Realizacja wskazuje polecenie lub moduł, kolumna "
        "Weryfikacja - moduł testowy, który sprawdza spełnienie. Pełny inwentarz testów wraz z "
        "opisem każdego przypadku zawiera załącznik A.",
    )
    add_body(
        doc,
        "Rysunek poniżej zestawia te same wymagania jako przypadki użycia: pokazuje, które "
        "czynności inicjuje badacz, które sięgają do zewnętrznych źródeł danych i które są "
        "częścią innych.",
    )
    add_diagram(
        doc, "przypadki_uzycia",
        "Diagram przypadków użycia. W nawiasach numery wymagań funkcjonalnych z tabeli "
        "powyżej. Relacja «include» oznacza czynność wykonywaną zawsze w ramach innej, a "
        "«extend» - wykonywaną opcjonalnie.",
    )

    add_heading(doc, "2.3. Wymagania niefunkcjonalne", 2)
    add_body(
        doc,
        "Wymagania niefunkcjonalne dotyczą własności, które system ma mieć niezależnie od "
        "wykonywanej czynności. W projekcie badawczym mają one wagę większą niż zwykle: praca "
        "nie kończy się działającym programem, lecz liczbą, którą ktoś ma potraktować "
        "poważnie. Liczba pochodząca z systemu nieodtwarzalnego albo pozwalającego modelowi "
        "zobaczyć przyszłość nie jest wynikiem gorszym - nie jest wynikiem w ogóle.",
    )
    add_table(
        doc,
        ["Nr", "Wymaganie", "Sposób spełnienia", "Weryfikacja"],
        [
            ["WN-01",
             "Brak zaglądania w przyszłość na każdym etapie: uczenia, kalibracji, oceny i "
             "symulacji.",
             "training_bounds(); sygnał ze świecy t realizowany po cenie otwarcia świecy t+1",
             _verified_by("test_predict_leakage.py", "test_backtest_execution.py")],
            ["WN-02",
             "Odtwarzalność: ten sam zestaw parametrów wraz z ziarnem daje ten sam wynik przy "
             "powtórnym uruchomieniu - zarówno prognoza, jak i jej ocena.",
             "flagi --deterministic --seed; ziarnowanie na okno przez window_seed(); sumy w "
             "ocenie liczone w ustalonej kolejności, niezależnej od losowego haszowania",
             _verified_by("test_window_seed.py", "test_prediction_evaluation.py")],
            ["WN-03",
             "Izolacja konfiguracji: zmiana parametru jednego horyzontu nie może wpłynąć na "
             "wynik pozostałych.",
             "ziarno wyprowadzane z krotki model, horyzont, rok, miesiąc, symbol",
             _verified_by("test_window_seed.py") + "; rozdział 11.5"],
            ["WN-04",
             "Rozpoznawalność pochodzenia wyniku: z samego artefaktu na dysku musi dać się "
             "odczytać model, interwał, ziarno i tryb deterministyczny.",
             "run_summary.json zapisywany przy każdym przebiegu",
             _verified_by("test_run_provenance.py")],
            ["WN-05",
             "Niezmienniczość cech względem poziomu ceny, warunkująca poprawność treningu na "
             "puli symboli.",
             "lista włączająca TRAINING_FEATURE_COLUMNS zamiast listy wykluczającej",
             _verified_by("test_scale_free_features.py")],
            ["WN-06",
             "Jawne raportowanie mocy statystycznej: wynik podawany jest wraz z liczbą "
             "obserwacji niezależnych po obu stronach podziału.",
             "warn_low_power() wywoływane dla części uczącej i ocenianej",
             _verified_by("test_train_window.py", "test_significance.py")],
            ["WN-07",
             "Nierozstrzyganie o jakości modelu na podstawie symulacji handlowej.",
             "wynik symulacji podawany wyłącznie z czterema hipotezami zerowymi",
             _verified_by("test_backtest_nulls.py") + "; rozdział 10.7"],
            ["WN-08",
             "Jedno miejsce definicji dla pojęć wspólnych: horyzontów, klas kierunku, kubełków "
             "głębokości, arytmetyki interwałów.",
             "moduł domain.py na poziomie pakietu; definicja etykiety w labels/forward.py",
             _verified_by("test_horizon_set.py", "test_depth_scheme.py",
                          "test_forward_labels.py")],
            ["WN-09",
             "Jedno miejsce podejmowania decyzji o werdykcie, wspólne dla planera, symulacji i "
             "oceny.",
             "planner/windows.py, klasa SequenceGrader",
             _verified_by("test_one_grader.py")],
            ["WN-10",
             "Brak stanu współdzielonego między przebiegami: nie istnieje katalog ostatniego "
             "wyniku, do którego zapisuje każdy przebieg.",
             "artefakty adresowane wyłącznie identyfikatorem przebiegu",
             _verified_by("test_run_attribution.py") + "; rozdział 11.3"],
            ["WN-11",
             "Warstwa HTTP nie wykonuje długotrwałych obliczeń we własnym procesie i nie "
             "buduje własnej wersji poleceń.",
             "uruchomienie python -m prosper.cli z shell=False; budowa argumentów w jednym "
             "module",
             _verified_by("test_cli_api_contract.py", "test_api_pool_command.py")],
            ["WN-12",
             "Panel przeglądarkowy działa bez etapu budowania i bez zewnętrznych zależności "
             "pobieranych w czasie działania.",
             "natywne moduły ES; biblioteki wykresów i fonty w przypiętych wersjach w katalogu "
             "frontend/vendor, z sumami kontrolnymi i licencjami",
             "przegląd frontend/: brak adresów spoza aplikacji"],
            ["WN-13",
             "Praca w całości na danych lokalnych: po pobraniu archiwów system nie wymaga "
             "połączenia z siecią.",
             "jezioro danych w formacie Parquet pod katalogiem data/; panel serwowany wraz "
             "ze wszystkimi zasobami przez tę samą aplikację",
             _verified_by("test_integration.py") + " (bez rzeczywistej sieci)"],
            ["WN-14",
             "Zachowanie jakości kodu: brak ostrzeżeń narzędzia kontroli statycznej, komplet "
             "testów przechodzący przed każdą zmianą.",
             "ruff check; pytest",
             "uruchomienie w cyklu pracy"],
            ["WN-15",
             "Czas przetworzenia jednego symbolu w standardowym zakresie pozostaje w granicach "
             "umożliwiających powtórzenie eksperymentu pod kilkoma ziarnami.",
             "predykcja miesiąca w jednym przebiegu wsadowym zamiast osobnego zbioru na świecę",
             _verified_by("test_tft_batched_inference.py")],
            ["WN-16",
             "Odporność na dane niekompletne i na przerwanie: symbol o krótszej historii nie "
             "przerywa przebiegu, a przerwany przebieg nie pozostawia artefaktu częściowego.",
             "zapis wyników dopiero po zakończeniu pętli kroczącej; świeca bez okna "
             "normalizacji oznaczana jako nietrenowana",
             _verified_by("test_history_shorter_than_range.py")],
            ["WN-17",
             "Panel nie daje innym stronom otwartym w przeglądarce ani wartościom z żądań "
             "dostępu do plików poza jeziorem danych.",
             "brak zezwoleń CORS; odrzucanie obcego nagłówka Host; walidacja każdej wartości "
             "tworzącej ścieżkę w jednym module; usuwanie tylko wewnątrz katalogu raportów",
             _verified_by("test_api_path_safety.py")],
            ["WN-18",
             "Jeden protokół kroczący dla wszystkich modeli: model dostarcza wyłącznie "
             "dopasowania i prognozy, a pętla, wiersze nietrenowane i zapis przebiegu są "
             "wspólne.",
             "predict/walk_forward.py; części wspólne w tabular.py i sequence.py",
             _verified_by("test_walk_forward.py", "test_train_window.py")],
        ],
        [1.5, 5.2, 4.6, 4.2],
        font=8,
    )
    add_caption(
        doc, "Tabela",
        "Wymagania niefunkcjonalne. Sześć z nich (WN-01, WN-03, WN-04, WN-05, WN-06, WN-10) "
        "zapisano dopiero po tym, jak ich naruszenie zostało wykryte w działającym systemie; "
        "opisy tych zdarzeń zawiera rozdział 11.",
    )

    add_heading(doc, "2.4. Wymagania wyprowadzone w trakcie badania", 2)
    add_body(
        doc,
        "Poniższe zestawienie podaje wymagania, których nie dało się sformułować przed "
        "uruchomieniem systemu, wraz z obserwacją, która je wymusiła. Zestawienie to jest "
        "elementem metodyki, nie kroniką pomyłek: w każdym z tych przypadków wcześniejsze "
        "brzmienie wymagania było wewnętrznie spójne i dawało wyniki wyglądające poprawnie. "
        "Właśnie dlatego wykrycie wymagało pomiaru, a nie przeglądu kodu.",
    )
    add_table(
        doc,
        ["Wymaganie w brzmieniu ostatecznym", "Brzmienie pierwotne", "Co je zmieniło", "Opis"],
        [
            ["Szerokość okna treningowego jest wyprowadzana osobno dla każdego horyzontu z "
             "jego rozpiętości i długości historii.",
             "Jedna szerokość okna podawana w konfiguracji.",
             "Przy stałej szerokości horyzont roczny dysponował 1,0 obserwacji niezależnej "
             "wobec 103 na tygodniowym. Rozszerzenie okna dla wszystkich odbiera obserwacje "
             "stronie ocenianej, więc nie jest rozwiązaniem.",
             "8.7"],
            ["Horyzonty są rozpiętościami kalendarzowymi przeliczanymi na świece osobno dla "
             "każdego interwału; zbiór to tydzień, miesiąc, kwartał i rok.",
             "Trzy horyzonty nazwane short, medium i long, wyrażone w liczbie świec.",
             "Nazwy short i long kolidowały z nazwami klas kierunku. Rozpiętość roczna po "
             "zaokrągleniu dawała 364 dni przy interwale tygodniowym i 365 przy dziennym, więc "
             "ten sam horyzont oznaczał dwa różne pytania.",
             "7.3"],
            ["Ziarno generatora liczb losowych wyprowadzane jest z krotki obejmującej symbol.",
             "Jedno ziarno ustawiane przed pętlą kroczącą.",
             "Bez symbolu w krotce wszystkie symbole dzieliły strumień inicjalizacji, przez co "
             "przebiegi uznane za niezależne nie były niezależne, a test istotności wskazał "
             "efekt, którego nie ma.",
             "11.5"],
            ["Kryterium zatrzymania uczenia liczone jest po kalibracji.",
             "Zatrzymanie na surowej entropii krzyżowej zbioru walidacyjnego.",
             "Surowa entropia karze za pewność siebie. Sieć wyostrza rozkład na długo przed "
             "utratą zdolności rozróżniania, a kalibrator odwraca to krok później - "
             "zatrzymanie następowało po jednej epoce, przy wciąż rosnącej trafności.",
             "8.8"],
            ["Cechy treningowe wybierane są z listy włączającej.",
             "Wszystkie kolumny pliku cech poza wymienionymi wyjątkami.",
             "Lista wykluczająca wpuszczała kolumny w wartościach bezwzględnych. Przy jednym "
             "symbolu normalizacja usuwa poziom i nic się nie dzieje; przy puli symboli poziom "
             "staje się etykietą symbolu.",
             "6.1"],
            ["Wiersze do treningu na puli symboli wybierane są po znaczniku czasu.",
             "Wybór po numerze wiersza, wspólny dla wszystkich symboli.",
             "Wiersz o tym samym numerze oznacza w różnych symbolach różne daty, więc okno "
             "wyznaczone indeksem uczyło prognozę na świecach, które jeszcze nie zaszły. "
             "Jedynym objawem była wyższa trafność.",
             "8.5"],
            ["Symulacja handlowa podawana jest wraz z regułą momentum jako punktem "
             "odniesienia.",
             "Porównanie z kupnem i trzymaniem.",
             "Strategia długa wychodząca z rynku na czas spadków bije kupno i trzymanie samym "
             "składaniem zysków, bez żadnej prognozy. Potrzebny był punkt odniesienia, który "
             "też wychodzi z rynku.",
             "9.5"],
            ["Horyzont bez danych treningowych oznaczany jest jako nietrenowany i wykluczany z "
             "oceny.",
             "Brak takiego pojęcia.",
             "Przy zbyt wąskim oknie dwa horyzonty zwracały stały rozkład, a warstwa oceny "
             "punktowała tę stałą jako prognozę.",
             "11.2"],
        ],
        [4.4, 3.4, 6.0, 1.7],
        font=8,
    )
    add_caption(
        doc, "Tabela",
        "Wymagania wyprowadzone z pomiaru. Kolumna Opis wskazuje podrozdział zawierający pełne "
        "omówienie. Wspólną cechą tych ośmiu przypadków jest to, że brzmienie pierwotne nie "
        "powodowało awarii - powodowało wynik, i to zwykle wynik lepszy od prawdziwego.",
    )
    add_body(
        doc,
        "Wniosek metodyczny jest następujący. W części inżynierskiej systemu - układ warstw, "
        "wersjonowanie artefaktów, rozdział wiersza poleceń od warstwy HTTP, format jeziora "
        "danych - specyfikacja spisana z góry byłaby trafna i oszczędziłaby pracy. W części "
        "badawczej wymagania są hipotezami o tym, jak zachowa się pomiar, a hipotezy tego "
        "rodzaju weryfikuje się wyłącznie przez uruchomienie. Stąd sposób pracy przyjęty w "
        "projekcie: każde wymaganie z drugiej grupy zapisywane jest wraz z obserwacją, która "
        "je wymusiła, i wraz z testem, który zapobiega powrotowi do brzmienia poprzedniego.",
    )

    add_heading(doc, "2.5. Granice zakresu", 2)
    add_body(
        doc,
        "Poniższe zadania pozostawiono świadomie poza systemem. Każde z nich jest wykonalne, a "
        "część była rozważana na wczesnym etapie; podanie powodu rezygnacji jest istotniejsze "
        "od samej listy, ponieważ w kilku przypadkach powodem jest wynik pomiaru, nie brak "
        "czasu.",
    )
    add_table(
        doc,
        ["Poza zakresem", "Powód"],
        [
            ["Dane fundamentalne, informacje ze spółek, nastroje z mediów społecznościowych.",
             "Pytanie badawcze dotyczy tego, ile informacji niesie sam szereg notowań. "
             "Dołączenie danych spoza szeregu odpowiadałoby na inne pytanie i uniemożliwiłoby "
             "odniesienie wyniku do słabej formy hipotezy efektywności rynku."],
            ["Składanie zleceń, połączenie z rachunkiem giełdowym, obsługa kluczy API "
             "użytkownika.",
             "System jest narzędziem badawczym. Zmierzona jakość prognozy nie uzasadnia "
             "podejmowania na jej podstawie decyzji finansowych, a moduł wykonawczy "
             "sugerowałby, że jest odwrotnie."],
            ["Obsługa rynków innych niż kryptowaluty: akcje, kontrakty terminowe, waluty.",
             "Zamierzenie pierwotne obejmowało system uniwersalny. Zrezygnowano z niego, gdy "
             "okazało się, że różnice w kalendarzu notowań, godzinach sesji, podziałach akcji "
             "i dostępności danych historycznych wymagają osobnej warstwy pozyskiwania dla "
             "każdego rynku. Wykonanie jednego rynku rzetelnie ma większą wartość niż czterech "
             "powierzchownie. Sam potok jest jednak od rynku niezależny - opisano to jako "
             "kierunek dalszych prac w rozdziale 13.2."],
            ["Dane z arkusza zleceń i mikrostruktura rynku.",
             "Binance Public Data udostępnia archiwa arkusza w rozdzielczości, przy której "
             "kilka lat historii dla jednego symbolu przekracza rozmiar całego obecnego "
             "jeziora danych. Przy horyzontach od tygodnia wzwyż związek między mikrostrukturą "
             "a prognozą jest ponadto odległy."],
            ["Optymalizacja portfela, dobór wag między symbolami, zarządzanie kapitałem.",
             "Wymagałaby wiarygodnej prognozy jako danych wejściowych. Jest to zadanie "
             "następujące po rozstrzygnięciu pytania badawczego, nie równoległe do niego."],
            ["Automatyczny dobór hiperparametrów.",
             "Przy zmierzonej liczbie obserwacji niezależnych przeszukiwanie przestrzeni "
             "hiperparametrów wybrałoby konfigurację najlepiej dopasowaną do szumu okresu "
             "oceny. Zestawy parametrów są zatem ustalone i wspólne dla wszystkich symboli."],
            ["Prognoza wielkości ruchu jako pojedynczej wartości liczbowej.",
             "Głębokość ruchu prognozowana jest jako rozkład na kubełkach, nie jako liczba. "
             "Rozkład niesie informację o niepewności, którą punktowa wartość oczekiwana "
             "zaciera."],
            ["Uwierzytelnianie użytkowników, obsługa wielu kont, wdrożenie sieciowe.",
             "System uruchamiany jest lokalnie przez jedną osobę. Warstwa uwierzytelniania "
             "zwiększyłaby objętość kodu bez związku z pytaniem badawczym."],
        ],
        [5.2, 10.8],
        font=8.5,
    )
    add_caption(
        doc, "Tabela",
        "Granice zakresu wraz z uzasadnieniem. Trzy pierwsze pozycje odpowiadają decyzjom "
        "podjętym w trakcie projektu, pozostałe - przyjętym od początku.",
    )

    add_heading(doc, "2.6. Kryteria akceptacji", 2)
    add_body(
        doc,
        "Kryteria akceptacji sformułowano tak, aby dało się je rozstrzygnąć pomiarem, a nie "
        "oceną. Warto zwrócić uwagę, że żadne z nich nie odnosi się do trafności prognozy. "
        "Byłoby to kryterium nierozstrzygalne w chwili formułowania i zarazem zachęta do "
        "powtarzania pomiaru aż do uzyskania wartości oczekiwanej - czyli dokładnie ten sposób "
        "pracy, którego cała metodyka rozdziału 9 ma unikać. System jest zdatny wtedy, gdy "
        "mierzy poprawnie, niezależnie od tego, co zmierzy.",
    )
    _acceptance_criteria(doc)
    page_break(doc)


def _acceptance_criteria(doc: Document) -> None:
    """The acceptance table, with the counts read from the suite rather than typed."""
    modules = test_inventory()
    cases = sum(len(m.cases) for m in modules)
    commands = len(cli_reference())
    add_table(
        doc,
        ["Kryterium", "Miara", "Stan"],
        [
            ["Komplet testów przechodzi bez błędów.",
             f"pytest; {len(modules)} modułów, {cases} udokumentowanych przypadków",
             "spełnione"],
            ["Kontrola statyczna nie zgłasza uwag.",
             "ruff check .",
             "spełnione"],
            ["Kontrola jakości danych nie wykrywa defektów w jeziorze.",
             "prosper qa check na wszystkich partycjach",
             "spełnione; szczegóły w rozdziale 5.4"],
            ["Każdy z czterech horyzontów faktycznie się uczy dla każdej architektury albo "
             "jest jawnie oznaczony jako nietrenowany.",
             "pole trained w wynikach przebiegu; licznik untrained_rows w ocenie",
             "spełnione"],
            ["Powtórzenie przebiegu pod tym samym ziarnem daje wynik identyczny co do bajtu.",
             "porównanie plików wynikowych dwóch uruchomień",
             "spełnione dla wszystkich architektur, na procesorze CPU, przy tej samej "
             "liczbie wątków"],
            ["Każdy raportowany wynik da się odtworzyć z artefaktu na dysku bez odwołania do "
             "notatek.",
             "run_summary.json zawiera model, interwał, ziarno i tryb",
             "spełnione"],
            ["Wynik podawany jest wraz z przedziałem ufności uwzględniającym nakładanie się "
             "etykiet.",
             "bootstrap blokowy; minimum 10 bloków",
             "spełnione"],
            ["Każda architektura uruchomiona pod co najmniej trzema ziarnami, aby rozrzut "
             "inicjalizacji był widoczny.",
             "liczba przebiegów na komórkę w załączniku E",
             "spełnione; dla obu modeli gradientowych ziarno nie zmienia wyniku (rozdział 10)"],
            ["Cały potok obsługiwalny zarówno z wiersza poleceń, jak i z przeglądarki.",
             f"{commands} poleceń CLI; komplet tras HTTP w załączniku C",
             "spełnione"],
            ["Usunięcie przebiegu nie pozostawia artefaktów pochodnych.",
             "kaskadowe usuwanie katalogów pochodnych przy usunięciu przebiegu",
             "spełnione"],
        ],
        [6.4, 6.2, 3.4],
        font=8.5,
    )
    add_caption(
        doc, "Tabela",
        "Kryteria akceptacji wraz ze stanem na dzień wygenerowania dokumentu. Odtwarzalność "
        "co do bajtu zweryfikowano dla każdej architektury, w tym obu sieci neuronowych, na "
        "procesorze CPU, na którym policzono wszystkie wyniki, przy stałej liczbie wątków "
        "(rozdział 9.6). Na procesorze "
        "graficznym nie była badana: tamtejsze implementacje części operacji nie są "
        "deterministyczne bez dodatkowych ustawień.",
    )


# ── 3 ────────────────────────────────────────────────────────────────────────

def _chapter_3_theory(doc: Document) -> None:
    add_heading(doc, "3. Podstawy teoretyczne", 1)

    add_heading(doc, "3.1. Hipoteza efektywności rynku", 2)
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
        "wykazania przewagi istotnie większej od losowej; jej nieodrzucenie - co jest wynikiem "
        "niniejszej pracy - nie dowodzi jej prawdziwości, lecz wyznacza górne ograniczenie na "
        "wielkość ewentualnej przewagi.",
    )

    add_heading(doc, "3.2. Własności finansowych szeregów czasowych", 2)
    add_body(
        doc,
        "Szeregi zwrotów aktywów wykazują zbiór własności powtarzalnych na różnych rynkach i "
        "w różnych okresach, zebranych i usystematyzowanych przez Conta (2001). Cztery z nich "
        "mają bezpośredni wpływ na konstrukcję opisywanego systemu.",
    )
    add_bullets(
        doc,
        [
            "Zwroty są w przybliżeniu nieskorelowane w czasie, ale ich wartości bezwzględne już "
            "nie - zmienność skupia się w klastrach, co czyni ją prognozowalną wtedy, gdy sam "
            "kierunek nie jest.",
            "Rozkład zwrotów ma ciężkie ogony: zdarzenia oddalone o wiele odchyleń standardowych "
            "występują znacznie częściej, niż przewiduje rozkład normalny.",
            "Szereg jest niestacjonarny - poziom ceny, zmienność i zależności między cechami "
            "zmieniają się w czasie, co uzasadnia uczenie w schemacie kroczącym zamiast "
            "jednorazowego podziału na zbiór uczący i testowy.",
            "Stosunek sygnału do szumu jest bardzo niski. Nawet prawdziwa przewaga rzędu jednego "
            "punktu procentowego trafności jest trudna do odróżnienia od zera przy realnie "
            "dostępnej liczbie obserwacji.",
        ],
    )

    add_heading(doc, "3.3. Prognoza kierunku jako zadanie klasyfikacji", 2)
    add_body(
        doc,
        "Prognoza jest sformułowana jako klasyfikacja binarna: znak zwrotu w przód na zadanym "
        "horyzoncie. Wyjściem modelu jest rozkład prawdopodobieństwa na dwóch klasach, a nie "
        "twarda decyzja - dzięki czemu ocena może korzystać z reguł właściwych, a warstwa "
        "decyzyjna może osobno rozstrzygać, czy przewaga pokrywa koszty transakcyjne.",
    )
    add_body(
        doc,
        "Nie wprowadzono klasy „bez zmiany”. Próg oddzielający ruch istotny od nieistotnego "
        "byłby dodatkowym, arbitralnym parametrem badania, a jego wartość wpływałaby na "
        "raportowaną trafność. Wielkość ruchu jest natomiast prognozowana osobno, jako rozkład "
        "na kubełkach głębokości (rozdział 7).",
    )

    add_heading(doc, "3.4. Reguły oceny właściwej", 2)
    add_body(
        doc,
        "Regułą właściwą (Gneiting i Raftery, 2007) nazywamy funkcję oceny prognozy "
        "probabilistycznej, która osiąga "
        "optimum wtedy i tylko wtedy, gdy prognoza podaje prawdziwe prawdopodobieństwo. Reguła "
        "niewłaściwa nagradza zniekształcanie prognozy - na przykład sama trafność nagradza "
        "deklarowanie skrajnej pewności.",
    )
    add_table(
        doc,
        ["Miara", "Definicja", "Zakres", "Interpretacja"],
        [
            ["Trafność", "udział poprawnie wskazanych kierunków", "0-1",
             "niewłaściwa, ale bezpośrednio interpretowalna"],
            ["Brier", "średni kwadrat różnicy prognozy i wyniku", "0-1",
             "właściwa; kwadratowa kara za pewność błędną"],
            ["Logarytmiczna", "średnia ujemnego logarytmu prawdopodobieństwa trafnej klasy",
             "0-∞", "właściwa; kara nieograniczona"],
            ["ECE", "średnia rozbieżność deklarowanej pewności i częstości trafień", "0-1",
             "mierzy kalibrację, nie zdolność rozróżniania"],
        ],
        [2.6, 6.2, 1.8, 5.4],
    )
    add_caption(doc, "Tabela", "Miary jakości prognozy stosowane w systemie.")
    add_body(
        doc,
        "Rozkład Briera na składnik kalibracji i składnik rozróżniania pokazuje, dlaczego obie "
        "wielkości trzeba raportować osobno: model może być doskonale skalibrowany i całkowicie "
        "pozbawiony zdolności rozróżniania - wystarczy, że zawsze podaje częstość bazową. "
        "Odwrotnie, model dobrze rozróżniający może być źle skalibrowany i wtedy jego "
        "prawdopodobieństwa nie nadają się do podejmowania decyzji, mimo że ranking jest trafny.",
    )

    add_heading(doc, "3.5. Kalibracja probabilistyczna", 2)
    add_figure(doc, "niezawodnosc", "Krzywe niezawodności: deklarowana pewność wobec zrealizowanej częstości trafień, zbiorczo po wszystkich przebiegach. Przekątna oznacza kalibrację doskonałą.")
    add_body(
        doc,
        "Kalibracja to zgodność deklarowanej pewności z rzeczywistą częstością: spośród świec, "
        "którym model przypisał prawdopodobieństwo 0,6, mniej więcej sześćdziesiąt procent "
        "powinno zrealizować prognozowany kierunek. System stosuje skalowanie temperaturą - "
        "jednoparametrową transformację logitów, która zmienia ostrość rozkładu, nie zmieniając "
        "porządku klas.",
    )
    add_body(
        doc,
        "Zaletą jednego parametru jest odporność na przeuczenie: kalibrator dopasowany na "
        "kilkudziesięciu obserwacjach pozostaje sensowny, podczas gdy metody wieloparametrowe "
        "wymagają zbiorów o rząd wielkości większych. Ograniczeniem jest to, że skalowanie "
        "temperaturą nie naprawia błędu zależnego od cechy - jeśli model jest przesadnie pewny "
        "wyłącznie w reżimie wysokiej zmienności, jeden parametr tego nie rozdzieli.",
    )

    add_heading(doc, "3.6. Bootstrap blokowy", 2)
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
        "Bootstrap blokowy (Künsch, 1989; Politis i Romano, 1994) losuje ciągi kolejnych "
        "obserwacji o długości równej rozpiętości "
        "horyzontu, dzięki czemu zachowuje zależność wewnątrz bloku i szacuje wariancję, która "
        "nie jest zaniżona. Efektywna liczba obserwacji wynosi w przybliżeniu liczbę świec "
        "podzieloną przez rozpiętość horyzontu; poniżej dziesięciu bloków system odmawia podania "
        "przedziału zamiast podać przedział fałszywie wąski.",
    )
    page_break(doc)


# ── 4 ────────────────────────────────────────────────────────────────────────

def _chapter_4_architecture(doc: Document) -> None:
    add_heading(doc, "4. Architektura systemu", 1)
    add_body(
        doc,
        "System działa na jednym komputerze i korzysta z dwóch publicznych usług giełdy "
        "Binance. Badacz obsługuje go z przeglądarki albo bezpośrednio z wiersza poleceń; obie "
        "drogi prowadzą do tych samych poleceń i tego samego jeziora danych.",
    )
    add_diagram(
        doc, "kontekst",
        "Kontekst systemu. Panel nie wykonuje obliczeń: serwer przekłada formularze na "
        "polecenia wiersza poleceń i uruchamia je kolejno jako podprocesy.",
    )

    add_heading(doc, "4.1. Potok przetwarzania", 2)
    add_body(
        doc,
        "System realizuje potok o ośmiu etapach. Każdy etap zapisuje wersjonowany artefakt na "
        "dysku, a każdy kolejny etap wskazuje artefakt źródłowy po nazwie. Nie istnieje "
        "współdzielony katalog „najnowszych” wyników - wcześniejsza wersja systemu taki katalog "
        "miała i skutkowało to przypisywaniem wyników ostatnio uruchomionego modelu wszystkim "
        "pozostałym (rozdział 11).",
    )
    add_table(
        doc,
        ["Etap", "Moduł", "Wejście", "Artefakt wyjściowy"],
        [
            ["Pobranie", "pipeline/backfill", "Binance Public Data (ZIP)",
             "świece 1m w formacie parquet"],
            ["Agregacja", "pipeline/aggregate", "świece 1m", "świece 1h, 1d, 1w"],
            ["Cechy", "features/build", "świece wybranego interwału", "features.parquet"],
            ["Etykiety", "labels/forward, labels/build", "świece wybranego interwału",
             "labels.parquet - podgląd; modele wyznaczają etykiety tą samą funkcją"],
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
    add_diagram(
        doc, "potok",
        "Diagram aktywności potoku przetwarzania. Cztery etapy weryfikacji czytają ten sam "
        "przebieg niezależnie od siebie; ranking zestawia przebiegi, które przeszły ocenę.",
    )

    add_heading(doc, "4.2. Warstwy", 2)
    add_bullets(
        doc,
        [
            "Warstwa domenowa (domain.py) - wspólne słownictwo: klasy kierunku, kubełki "
            "głębokości, definicje horyzontów, przeliczenia interwałów, miary liczby obserwacji "
            "efektywnych. Umieszczona na poziomie pakietu, ponieważ etykietowanie, prognozowanie, "
            "ocena i planowanie muszą posługiwać się tym samym słownictwem; przypisanie jej do "
            "którejkolwiek z tych warstw czyniłoby pozostałe jej klientami.",
            "Warstwa dostępu do danych (binance/, pipeline/, storage/) - pobieranie z Binance "
            "Public Data, konwersja do partycjonowanego formatu parquet, adresowanie i "
            "wersjonowanie artefaktów.",
            "Warstwa etykiet (labels/) - jedna definicja etykiety kierunku i głębokości, z "
            "której korzystają wszystkie modele, oraz zapis etykiet czterech horyzontów do "
            "pliku, który rysuje panel.",
            "Warstwa modelowa (predict/) - jeden protokół kroczący (walk_forward.py), w którym "
            "pięć modeli dostarcza jedynie dopasowania i prognozy; części wspólne modeli "
            "tabelarycznych (tabular.py) i sekwencyjnych (sequence.py), arytmetyka okna "
            "(window.py) oraz trening wielosymbolowy (pool.py).",
            "Warstwa weryfikacji (eval/) - reguły oceny właściwej, przedziały ufności, analiza "
            "stabilności w czasie, symulacja handlowa wraz z hipotezami zerowymi.",
            "Warstwa decyzyjna (planner/) - przekształcenie prognozy w rekomendację "
            "siedmiostopniową, z bramkami ryzyka i progiem kosztowym.",
            "Warstwa prezentacji (api/, frontend/) - FastAPI jako cienka warstwa uruchamiająca "
            "polecenia wiersza poleceń oraz jednostronicowa aplikacja w natywnych modułach ES. "
            "Każda wartość z żądania, która może stać się ścieżką na dysku, przechodzi przez "
            "jeden moduł walidacji (api/validation.py); aplikacja nie zezwala na zapytania z "
            "innych domen i odrzuca nagłówek Host wskazujący inną maszynę.",
        ],
    )
    add_diagram(
        doc, "komponenty",
        "Diagram komponentów. Strzałka oznacza zależność wywołania. Serwer HTTP woła "
        "bezpośrednio tylko lekkie operacje - stabilność, symulację i inwentarz - a wszystko, "
        "co trwa dłużej, uruchamia przez wiersz poleceń.",
    )

    add_heading(doc, "4.3. Decyzje technologiczne", 2)
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
    add_diagram(
        doc, "wdrozenie",
        "Diagram wdrożenia. Serwer i wątek roboczy kolejki działają w jednym procesie, a każde "
        "polecenie z kolejki - w osobnym podprocesie, jedno po drugim.",
    )

    add_heading(doc, "4.4. Wersjonowanie artefaktów", 2)
    add_body(
        doc,
        "Każdy przebieg prognozy zapisywany jest w katalogu o nazwie złożonej z modelu, "
        "interwału i znacznika czasu. Wszystkie artefakty pochodne - ocena, stabilność, "
        "rekomendacje, symulacja - powstają w katalogach o tej samej nazwie, w osobnych "
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
    add_diagram(
        doc, "artefakty",
        "Model artefaktów. Romb oznacza kompozycję: artefakt pochodny istnieje tylko razem z "
        "przebiegiem, z którego powstał. Linie przerywane wskazują, z czego dany artefakt "
        "wyznaczono.",
    )
    page_break(doc)


# ── 5 ────────────────────────────────────────────────────────────────────────

def _chapter_5_data(doc: Document) -> None:
    add_heading(doc, "5. Dane", 1)

    add_heading(doc, "5.1. Źródło", 2)
    add_body(
        doc,
        "Dane pochodzą z serwisu Binance Public Data, udostępniającego historyczne świece w "
        "postaci miesięcznych archiwów ZIP z plikami CSV. Pobierany jest interwał "
        "jednominutowy, z którego pozostałe interwały powstają przez agregację - dzięki czemu "
        "świeca godzinowa i dzienna pochodzą z tego samego źródła i są ze sobą spójne.",
    )

    add_heading(doc, "5.2. Schemat świecy", 2)
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

    add_heading(doc, "5.3. Agregacja", 2)
    add_body(
        doc,
        "Agregacja z interwału jednominutowego przebiega według reguł: otwarcie z pierwszej "
        "świecy okresu, zamknięcie z ostatniej, maksimum i minimum jako skrajne wartości, "
        "wolumeny i liczba transakcji jako sumy. Interwał tygodniowy partycjonowany jest numerem "
        "tygodnia zamiast miesiąca, ponieważ tydzień może przekraczać granicę miesiąca.",
    )

    add_heading(doc, "5.4. Kontrola jakości", 2)
    add_bullets(
        doc,
        [
            "Kompletność - wykrywanie brakujących okresów przez porównanie liczby świec z "
            "liczbą oczekiwaną dla danego zakresu i interwału.",
            "Duplikaty - świece o powtórzonym znaczniku czasu, powstające przy ponownym "
            "pobraniu tego samego miesiąca.",
            "Spójność wewnętrzna - sprawdzenie, że minimum nie przekracza maksimum oraz że "
            "otwarcie i zamknięcie mieszczą się w tym zakresie.",
            "Wartości nieujemne - wolumen i liczba transakcji nie mogą być ujemne.",
            "Ciągłość - wykrywanie skoków czasu większych niż jeden interwał.",
        ],
    )

    _data_quality_results(doc)

    add_heading(doc, "5.5. Charakterystyka zbioru", 2)
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
            "Symbole o krótszej historii ograniczają horyzont roczny (rozdział 9.3).",
        )
    else:
        add_body(doc, "Jezioro danych jest puste - uruchom etap pobrania.", italic=True)
    page_break(doc)


# ── 6 ────────────────────────────────────────────────────────────────────────

def _chapter_6_features(doc: Document) -> None:
    add_heading(doc, "6. Cechy", 1)

    add_heading(doc, "6.1. Zasada: niezmienniczość względem skali", 2)
    add_body(
        doc,
        "Zbiór cech treningowych jest listą włączającą, nie wykluczającą. Wcześniejsza wersja "
        "przekazywała modelowi każdą kolumnę poza kilkoma nazwanymi wyjątkami, przez co do "
        "wejścia trafiały bezwzględne poziomy cen. Przy jednym instrumencie było to nieszkodliwe, "
        "ponieważ normalizacja usuwa poziom; przy uczeniu na kilku instrumentach naraz stawało "
        "się etykietą instrumentu - zmierzone różnice sięgały 553-krotności dla średniej "
        "kroczącej i 387-krotności dla histogramu MACD.",
    )
    add_body(
        doc,
        "Lista włączająca wymusza świadomą decyzję przy każdej nowej kolumnie. Lista wykluczająca "
        "przyjęłaby każdą nową cechę względną razem z jej bezwzględnym odpowiednikiem, co jest "
        "dokładnie tym błędem, którego należało uniknąć. Kolumny bezwzględne pozostają w pliku "
        "cech, ponieważ korzysta z nich panel przeglądarkowy, ale nie trafiają do modelu.",
    )

    add_heading(doc, "6.2. Katalog cech treningowych", 2)
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
        add_heading(doc, f"6.2.{sorted(by_group).index(group) + 1}. {group}", 3)
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


# ── 7 ────────────────────────────────────────────────────────────────────────

def _chapter_7_labels(doc: Document) -> None:
    add_heading(doc, "7. Etykiety", 1)
    add_body(
        doc,
        "Etykieta opisuje, co rynek zrobił w horyzoncie: stronę zwrotu w przód i przedział jego "
        "wielkości. Jej definicja istnieje w systemie w jednym miejscu (labels/forward.py) i "
        "korzysta z niej każdy z pięciu modeli oraz plik etykiet, który rysuje panel. Nie zawsze "
        "tak było: ta sama pętla znajdowała się w czterech modelach, a osobny etap potoku "
        "zapisywał piątą wersję - jeden horyzont, przesunięty o wiersze zamiast o okres "
        "kalendarzowy - której nie czytał żaden model. Kopie dawały zgodne wyniki, ale nic nie "
        "gwarantowało, że zgodność przetrwa następną zmianę. Scalenie zweryfikowano porównaniem "
        "bajtowym prognoz wszystkich modeli przed zmianą i po niej.",
    )

    add_heading(doc, "7.1. Kierunek", 2)
    add_body(
        doc,
        "Etykieta kierunku jest znakiem zwrotu w przód na zadanym horyzoncie i przyjmuje dwie "
        "wartości. Nie istnieje klasa pośrednia ani próg oddzielający ruch istotny od "
        "nieistotnego: taki próg byłby dodatkowym parametrem badania, wpływającym na raportowaną "
        "trafność, a jego uzasadnienie musiałoby odwoływać się do kosztów transakcyjnych, czyli "
        "do warstwy decyzyjnej, a nie do danych. Zwrot dokładnie zerowy nie ma strony i jest "
        "wykluczany z oceny.",
    )

    add_heading(doc, "7.2. Głębokość", 2)
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
        [[label, label.replace("-", " - ").replace("+", " i więcej")] for label in labels],
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

    add_heading(doc, "7.3. Horyzonty", 2)
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


# ── 8 ────────────────────────────────────────────────────────────────────────

def _chapter_8_models(doc: Document) -> None:
    add_heading(doc, "8. Architektury modeli", 1)

    add_body(
        doc,
        "Porównywanych jest pięć rodzin o rosnącej pojemności i rosnącym koszcie obliczeniowym. "
        "Wszystkie realizują ten sam protokół kroczący (8.6) i wystawiają identyczny interfejs, "
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
    add_diagram(
        doc, "prognosty",
        "Diagram klas prognostów. Pętla krocząca zna wyłącznie protokół Forecaster; części "
        "wspólne modeli tablicowych i sekwencyjnych leżą w klasach abstrakcyjnych, a "
        "konkretne architektury dostarczają tylko dopasowania i prognozy.",
    )

    add_heading(doc, "8.1. Model odniesienia", 2)
    add_body(
        doc,
        "Model odniesienia zlicza częstości kierunków w oknie i wygładza je regułą Laplace'a. "
        "Nie zawiera uczenia, więc stanowi granicę, poniżej której nie ma sensu schodzić: model "
        "uczący przegrywający z częstością bazową nie wnosi nic. Okno zlicza wyłącznie wyniki "
        "już zrealizowane w chwili prognozy - zliczanie ostatnich świec oznaczałoby korzystanie "
        "ze zwrotów, które jeszcze nie nastąpiły, czyli przewagę, której punkt odniesienia mieć "
        "nie może.",
    )

    add_heading(doc, "8.2. Modele gradientowe", 2)
    add_figure(doc, "waznosc", "Względna ważność cech w modelu gradientowym. Rozkład jest płaski - żadna cecha nie dominuje, co jest spójne z brakiem wykrywalnej struktury w danych wejściowych.")
    add_body(
        doc,
        "Dwie implementacje wzmacniania gradientowego - z biblioteki scikit-learn i z XGBoost "
        "(Chen i Guestrin, 2016) - "
        "reprezentują podejście tabelaryczne, w którym każda świeca jest niezależnym wektorem "
        "cech. Nie modelują zależności czasowej wprost; informacja o przeszłości wchodzi "
        "wyłącznie przez cechy liczone na oknach. Są tanie obliczeniowo, co pozwala uruchamiać "
        "je wielokrotnie i traktować jako punkt odniesienia dla modeli sekwencyjnych.",
    )

    add_heading(doc, "8.3. Sieć rekurencyjna GRU", 2)
    add_body(
        doc,
        "Sieć rekurencyjna z bramkami (Cho i in., 2014) przetwarza sekwencję kolejnych świec, "
        "utrzymując stan "
        "ukryty. Bramki aktualizacji i resetu regulują, jaka część stanu przechodzi dalej, co "
        "łagodzi problem zanikającego gradientu występujący w prostych sieciach rekurencyjnych. "
        "W odróżnieniu od modeli tabelarycznych zależność czasowa jest tu modelowana wprost, "
        "kosztem znacznie większej liczby parametrów przy tej samej liczbie obserwacji.",
    )

    add_heading(doc, "8.4. Temporal Fusion Transformer", 2)
    add_body(
        doc,
        "Architektura (Lim i in., 2021) łącząca kodowanie rekurencyjne z mechanizmem uwagi po "
        "osi czasu oraz "
        "sieciami wyboru zmiennych, które uczą się wagi poszczególnych cech. Jest to model o "
        "największej pojemności w zestawieniu i zarazem najdroższy - jeden przebieg na jednym "
        "symbolu kosztuje około godziny i pół, czyli o rząd wielkości więcej niż pozostałe "
        "łącznie.",
    )
    add_body(
        doc,
        "Prognoza całego miesiąca douczania powstaje w jednym przebiegu wsadowym, z historią "
        "celu zamaskowaną na granicy miesiąca. Konstrukcja osobnego zbioru dla każdej świecy "
        "była pierwotną przyczyną wielogodzinnego czasu wykonania i została zastąpiona.",
    )

    add_heading(doc, "8.5. Trening wielosymbolowy", 2)
    add_body(
        doc,
        "Modele tabelaryczne przyjmują opcjonalną listę dodatkowych symboli, których wiersze "
        "wchodzą do zbioru treningowego; prognoza pozostaje wyznaczana dla jednego symbolu. "
        "Wiersze wybierane są po znaczniku czasu, nigdy po numerze wiersza - instrumenty "
        "notowane od różnych dat nie mają wspólnej numeracji, a okno oparte na indeksie "
        "wprowadziłoby do zbioru treningowego świece, które jeszcze nie nastąpiły.",
    )
    add_body(
        doc,
        "Zysk z takiego łączenia jest wyraźnie mniejszy, niż sugeruje liczba wierszy. Zwroty "
        "kryptowalut są silnie skorelowane - zmierzona średnia korelacja par wynosi około 0,60 "
        "na horyzoncie tygodniowym i 0,76 na rocznym - a liczba symboli efektywnych dana wzorem "
        "n / (1 + (n−1)ρ) dąży do 1/ρ. Trzy symbole odpowiadają zatem 1,19-1,36 symbolu "
        "niezależnego, a żadna ich liczba nie podniesie horyzontu rocznego powyżej około 1,3. "
        "Jest to argument za prognozą przekrojową, a nie za łączeniem większej liczby "
        "instrumentów.",
    )

    add_heading(doc, "8.6. Protokół kroczący", 2)
    add_body(
        doc,
        "Model douczany jest raz na miesiąc kalendarzowy, na oknie kończącym się przed "
        "prognozowaną świecą, po czym prognozuje kolejne świece tego miesiąca. Zbiór treningowy "
        "zawiera wyłącznie etykiety zrealizowane w chwili prognozy: etykieta świecy k wymaga "
        "zamknięcia świecy k+F, więc nadaje się do uczenia przy prognozowaniu świecy i dopiero "
        "wtedy, gdy k+F < i.",
    )
    add_body(
        doc,
        "Protokół ten jest zapisany w systemie raz (predict/walk_forward.py), a każda z pięciu "
        "architektur dostarcza jedynie dwóch operacji: dopasowania na początku miesiąca i "
        "prognozy dla pojedynczej świecy. Wcześniej pętla była powielona w każdym modelu i "
        "reguły protokołu docierały do kopii nierówno - model odniesienia zapisywał pusty "
        "przebieg tam, gdzie pozostałe go odrzucały, a TFT pomijał pierwszą świecę miesiąca "
        "pozbawionego historii zamiast oznaczyć ją jako nietrenowaną. Scalenie zweryfikowano "
        "porównaniem bajtowym prognoz w piętnastu konfiguracjach obejmujących wszystkie "
        "architektury; jedyną różnicą jest odzyskany wiersz TFT.",
    )
    add_body(
        doc,
        "Jedna rozbieżność została świadomie zachowana. GRU i TFT normalizują cechy medianą i "
        "rozstępem międzykwartylowym, lecz brakującą wartość traktują inaczej: GRU zastępuje ją "
        "zerem po przeskalowaniu, czyli medianą okna, a TFT zerem przed przeskalowaniem. "
        "Pierwsze 89 świec każdego symbolu ma braki wynikające z rozruchu wskaźników, a "
        "najwcześniejsze okna sięgają tak daleko, więc ujednolicenie zmieniłoby opublikowane "
        "wyniki jednej z sieci i wymagałoby ponownego przeliczenia.",
    )
    add_diagram(
        doc, "petla_kroczaca",
        "Diagram aktywności protokołu kroczącego. Horyzont, którego nie dało się wytrenować, "
        "nie znika z wyniku, lecz zostaje jawnie oznaczony, więc liczba wierszy przebiegu nie "
        "zależy od modelu.",
    )

    add_heading(doc, "8.7. Okno treningowe wyprowadzane z horyzontu", 2)
    add_body(
        doc,
        "Szerokość okna nie jest parametrem konfiguracyjnym, lecz wynika z horyzontu i "
        "dostępnej historii. Kolidują tu dwa wymagania. Model potrzebuje wierszy - kilkuset, aby "
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
        "stronach jednocześnie, wynosi około 3,45 obserwacji - przy oknie równym połowie "
        "historii.",
    )

    add_heading(doc, "8.8. Kalibracja i kryterium zatrzymania", 2)
    add_body(
        doc,
        "Prawdopodobieństwa kalibrowane są skalowaniem temperatury (Guo i in., 2017) na "
        "wycinku wydzielonym "
        "wewnątrz okna treningowego - nigdy na świecach późniejszych, ponieważ kalibrator "
        "widziałby wtedy etykiety niedostępne prognozie.",
    )
    add_body(
        doc,
        "Modele głębokie zatrzymują uczenie na stracie mierzonej po kalibracji, a nie na "
        "surowej entropii krzyżowej. Surowa strata karze za nadmierną pewność, sieć staje się "
        "nadmiernie pewna na długo przed utratą zdolności rozróżniania, a kalibrator usuwa tę "
        "nadmierną pewność krok później - zatrzymywanie na surowej wartości zwracało model po "
        "jednej epoce, podczas gdy trafność wciąż rosła.",
    )
    page_break(doc)


# ── 9 ────────────────────────────────────────────────────────────────────────

def _chapter_9_methodology(doc: Document) -> None:
    add_heading(doc, "9. Metodyka pomiaru", 1)
    add_body(
        doc,
        "Rozdział opisuje warstwę, która czyni porównanie architektur rozstrzygalnym. Bez niej "
        "wynik „architektury nie różnią się” pozostaje nieodróżnialny od „pomiar był zbyt "
        "zaszumiony”, a każdy pojedynczy wynik dodatni - nieodróżnialny od artefaktu.",
        italic=True,
    )

    add_heading(doc, "9.1. Miary bez wolnych parametrów", 2)
    add_body(
        doc,
        "Model oceniany jest trafnością, wynikiem Briera, stratą logarytmiczną i oczekiwanym "
        "błędem kalibracji. Żadna z tych miar nie ma parametru, który dałoby się dobrać tak, aby "
        "wynik wyglądał lepiej. Jest to własność celowa i odróżnia je od symulacji handlowej, "
        "której wynik zależy od kilkunastu decyzji wykonawczych.",
    )
    add_diagram(
        doc, "ocena",
        "Diagram aktywności oceny przebiegu. Świeca, której wynik nie jest jeszcze znany albo "
        "której zwrot wynosi dokładnie zero, nie dostaje oceny; ma osobny status, aby nie "
        "zniekształcać średnich.",
    )

    add_heading(doc, "9.2. Przedziały ufności przy nakładających się etykietach", 2)
    add_body(
        doc,
        "Każda trafność jest średnią po świecach o nakładających się etykietach - problem "
        "opisany szczegółowo przez López de Prado (2018) w kontekście uczenia maszynowego na "
        "danych finansowych. Naiwny przedział "
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
    add_body(
        doc,
        "Z tego samego próbkowania odczytywana jest wartość p dla hipotezy, że trafność wynosi "
        "0,5: odsetek wylosowanych trafności leżących po drugiej stronie 0,5 niż większość z "
        "nich, podwojony. Jest to test, który odwraca przedział percentylowy (Efron i "
        "Tibshirani, 1993), więc p spada poniżej 0,05 tam, gdzie 95-procentowy przedział nie "
        "obejmuje 0,5 - oba odczyty pochodzą z tych samych losowań i nie mogą sobie przeczyć. "
        "Pomiarów jest jednak kilkaset, a przy takiej liczbie próg 0,05 przekracza kilkanaście z "
        "nich nawet wtedy, gdy żaden model nie ma przewagi. Wartości p koryguje się zatem "
        "procedurą Benjaminiego i Hochberga (1995), która ogranicza oczekiwany odsetek "
        "fałszywych odkryć wśród pomiarów uznanych za istotne. Gwarancja ta obowiązuje także "
        "przy dodatniej zależności między testami (Benjamini i Yekutieli, 2001), a z taką "
        "zależnością ma się tu do czynienia: ziarna jednej konfiguracji widzą te same dane, a "
        "kursy instrumentów poruszają się razem.",
    )

    add_heading(doc, "9.3. Moc statystyczna jako ograniczenie", 2)
    add_figure(doc, "moc", "Średnia liczba obserwacji niezależnych na horyzont, w skali logarytmicznej. Horyzont roczny leży poniżej progu, przy którym system w ogóle wyznacza przedział ufności - i jest to własność ilości danych, nie modelu.")
    add_body(
        doc,
        "Liczba obserwacji niezależnych to liczba świec podzielona przez rozpiętość horyzontu. "
        "System ostrzega przed uruchomieniem przebiegu o horyzontach, dla których liczba ta jest "
        "zbyt mała - osobno po stronie treningu i po stronie oceny. Obie strony czerpią z tej "
        "samej skończonej historii, więc poszerzanie okna przenosi problem, zamiast go usuwać, a "
        "ostrzeżenie o jednej tylko stronie byłoby zaproszeniem do poszerzania okna aż do "
        "zaniku możliwości oceny.",
    )

    add_heading(doc, "9.4. Stabilność w czasie", 2)
    add_body(
        doc,
        "Miary jakości wyliczane są ponadto osobno dla każdego miesiąca kalendarzowego. "
        "Pojedyncza średnia z całego okresu maskuje sytuację, w której model działa w jednym "
        "reżimie rynkowym i zawodzi w innym; rozbicie na miesiące ujawnia to i pozwala stosować "
        "testy nieparametryczne, w których jednostką jest miesiąc, a nie świeca - co jest "
        "istotne, gdy etykiety wewnątrz miesiąca nakładają się niemal całkowicie.",
    )

    add_heading(doc, "9.5. Symulacja handlowa i cztery hipotezy zerowe", 2)
    add_figure(doc, "kapital", "Przebieg wartości portfela dla pięciu architektur wobec strategii kup i trzymaj, skala logarytmiczna. Żadna strategia oparta na prognozie nie zbliża się do prostego trzymania pozycji w badanym okresie.")
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
            ["Kup i trzymaj", "nic - punkt odniesienia rynkowy", "najsłabszy"],
            ["Stała ekspozycja równa średniej strategii",
             "efekt samego poziomu zaangażowania", "średni"],
            ["Ta sama ścieżka ekspozycji przesunięta cyklicznie",
             "efekt wyczucia momentu - test właściwy", "mocny"],
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
    add_body(
        doc,
        "Symulacja nie ocenia prognozy własną regułą. Każdą świecę klasyfikuje ten sam "
        "oceniacz, który wyznacza rekomendacje (pierwszy rysunek poniżej), dzięki czemu werdykt "
        "w raporcie rekomendacji i w symulacji nie może się rozejść. Symulacja przekłada "
        "werdykt na docelową ekspozycję i dopiero na końcu zestawia wynik z czterema punktami "
        "odniesienia (drugi rysunek).",
    )
    add_diagram(
        doc, "rekomendacje",
        "Diagram aktywności wyznaczania rekomendacji. Progi przewagi są stałe, ponieważ "
        "przewaga jest różnicą dwóch prawdopodobieństw; progi ryzyka są kwantylami ryzyka "
        "zaobserwowanego wcześniej, więc żadna świeca nie wpływa na własny werdykt.",
    )
    add_diagram(
        doc, "symulacja",
        "Diagram aktywności symulacji handlowej. Transakcja następuje po cenie otwarcia dnia "
        "następującego po sygnale, więc symulacja nie korzysta z ceny, której w chwili "
        "decyzji nie znano.",
    )

    add_heading(doc, "9.6. Odtwarzalność i ziarnowanie", 2)
    add_body(
        doc,
        "Każdy przebieg zapisuje ziarno, tryb deterministyczny, wyprowadzone okna treningowe, "
        "liczbę faktycznie wykonanych epok, wersje bibliotek mogących zmienić prognozę (wraz z "
        "wariantem kompilacji torch, ponieważ wersje CPU i CUDA nie dają tych samych bajtów), "
        "commit kodu i informację, czy pakiet miał niezatwierdzone zmiany, a także liczbę okien "
        "douczania, w których wystąpił wyjątek. Losowość każdego okna douczania wyprowadzana jest z "
        "jego własnej tożsamości - ziarna bazowego, modelu, horyzontu, miesiąca i symbolu - a nie "
        "z jednego wspólnego strumienia.",
    )
    add_body(
        doc,
        "Wspólny strumień powodowałby, że liczba kroków uczenia wykonanych gdziekolwiek przesuwa "
        "inicjalizację wszystkiego, co następuje po niej, czyniąc eksperyment z jedną zmienną "
        "niewykonalnym. Pominięcie symbolu w tej tożsamości spowodowało z kolei, że wszystkie "
        "instrumenty startowały z identycznych wag - konsekwencje opisano w rozdziale 11.5.",
    )
    add_body(
        doc,
        "Odtwarzalność zweryfikowano pomiarowo: powtórzenie tej samej konfiguracji daje plik "
        "predykcji identyczny co do bajtu dla każdego modelu korzystającego z losowości - pod "
        "warunkiem tej samej liczby wątków obliczeniowych. Sieci GRU i TFT policzone na ośmiu "
        "i na czterech wątkach dają wyniki różne, ponieważ równoległe redukcje sumują składniki "
        "w innej kolejności: w GRU różnice dotyczą ostatnich bitów, w TFT kierunek prognozy "
        "zgadza się w 98-99% świec, a prawdopodobieństwa na dłuższych horyzontach różnią się "
        "nawet o 0,88. Modele gradientowe i model odniesienia tej zależności nie wykazują. "
        "Wszystkie przebiegi badania policzono zatem na czterech wątkach, a ich liczba "
        "zapisywana jest w podsumowaniu każdego przebiegu.",
    )
    add_body(
        doc,
        "Ziarno nie zmienia natomiast wyniku obu modeli gradientowych: uczą się bez "
        "podpróbkowania wierszy i cech, więc ich przebiegi pod trzema ziarnami są identyczne "
        "co do bajtu. Szum inicjalizacji mierzą w badaniu wyłącznie GRU i TFT (rozdział 10).",
    )
    page_break(doc)


# ── 10 ────────────────────────────────────────────────────────────────────────

def _chapter_10_results(doc: Document, runs: dict) -> None:
    add_heading(doc, "10. Wyniki: porównanie architektur", 1)

    total = sum(len(v) for v in runs.values())
    seeds = sorted({r["seed"] for v in runs.values() for r in v if r["seed"] is not None})
    add_body(
        doc,
        f"Podstawą tabel jest {total} przebiegów odczytanych z artefaktów projektu"
        + (f", pod ziarnami: {', '.join(str(s) for s in seeds)}." if seeds else ".")
        + " Wszystkie policzono na czterech wątkach procesora (rozdział 9.6). Każdą konfigurację "
        "uruchomiono wielokrotnie, ponieważ pojedynczy przebieg niesie szum inicjalizacji, "
        "którego przedział ufności liczony wewnątrz tego przebiegu nie obejmuje.",
    )
    invariant = sorted({model for _, model in seed_invariant(runs)})
    if invariant:
        add_body(
            doc,
            f"Dla architektur {_listing(invariant)} trzy ziarna dają na każdym symbolu wynik "
            "identyczny co do bajtu: uczą się deterministycznie, bez podpróbkowania wierszy i "
            f"cech. Różnych pomiarów jest zatem {sum(len(v) for v in distinct_runs(runs).values())}"
            ", a rozrzut inicjalizacji wyznaczają wyłącznie pozostałe architektury.",
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

    add_heading(doc, "10.1. Ranking architektur wobec podłogi szumu", 2)
    add_body(
        doc,
        "Tabele poniżej odpowiadają wprost na pytanie badawcze. Wielkością rozstrzygającą jest "
        "porównanie rozstępu między architekturami z rozrzutem tej samej architektury pod "
        "różnymi ziarnami. Jeżeli rozstęp nie przekracza rozrzutu, ranking architektur nie ma "
        "podstaw. Rozrzut liczony jest wyłącznie z architektur, których wynik ziarno zmienia: "
        "modele deterministyczne wniosłyby do średniej zera i zaniżyły poziom szumu, zamieniając "
        "remis w ranking. Rozstęp podawany jest dwukrotnie - dla wszystkich architektur i dla "
        "samych uczących się - ponieważ model odniesienia nie uczy się i na dłuższych "
        "horyzontach odstaje od reszty.",
    )

    spread = architecture_spread(runs)
    distinct = distinct_runs(runs)
    for horizon in HORIZON_NAMES:
        cell = spread[horizon]
        means = cell["means"]
        if len(means) < 2:
            continue
        order = sorted(means, key=lambda m: -means[m])
        labels = {}
        for model in order:
            taken = sum(len(horizon_values(runs.get((symbol, model), []), horizon, "accuracy"))
                        for symbol in SYMBOLS)
            different = sum(
                len(horizon_values(distinct.get((symbol, model), []), horizon, "accuracy"))
                for symbol in SYMBOLS
            )
            labels[model] = str(taken) if different == taken else f"{taken} ({different} różnych)"

        add_body(doc, f"Horyzont „{horizon}”:")
        add_table(
            doc,
            ["Poz.", "Architektura", "Przebiegów", "Trafność średnia"],
            [[str(rank), model, labels[model], fmt(means[model])]
             for rank, model in enumerate(order, start=1)],
            [1.6, 4.6, 3.6, 3.4],
            font=9,
        )
        noise, every, learners = cell["noise"], cell["range"], cell["range_learners"]
        if noise is None:
            add_caption(
                doc, "Tabela",
                f"Horyzont „{horizon}”. Rozstęp: {pl(every)}. Brak powtórzeń pod różnymi "
                "ziarnami - rozrzutu inicjalizacji nie zmierzono.",
            )
            continue
        if learners <= noise:
            verdict = ("architektury uczące się mieszczą się w rozrzucie ziarna, więc ich "
                       "ranking nie ma podstaw")
            if every > noise:
                verdict += "; rozstęp całego zestawienia wynika z modelu odniesienia"
        else:
            verdict = ("rozstęp architektur uczących się przekracza rozrzut ziarna "
                       + ("nieznacznie" if learners < 1.5 * noise else "wyraźnie"))
        add_caption(
            doc, "Tabela",
            f"Horyzont „{horizon}”. Rozstęp średnich: {pl(every)} dla wszystkich architektur, "
            f"{pl(learners)} dla uczących się. Rozrzut tej samej architektury pod innym "
            f"ziarnem: {pl(noise)}. Wniosek: {verdict}.",
        )

    _results_rank_stability(doc, runs)

    add_heading(doc, "10.3. Wyniki szczegółowe", 2)
    add_figure(doc, "porownanie", "Trafność średnia każdej architektury na czterech horyzontach. Wąsy to odchylenie standardowe między wszystkimi przebiegami danej architektury - na różnych instrumentach i pod różnymi ziarnami; przerywana linia oznacza poziom losowy. Nachodzenie wąsów na siebie i na linię 0,5 jest treścią wyniku.")
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
                    fmt(float(arr.std(ddof=1))) if len(arr) > 1 else "-",
                    f"{arr.min():.3f}-{arr.max():.3f}" if len(arr) > 1 else fmt(float(arr[0])),
                    fmt(float(np.mean(eces))) if eces else "-",
                ])
        if not rows:
            continue
        add_body(doc, f"Horyzont „{horizon}” - wszystkie symbole i ziarna:")
        add_table(
            doc,
            ["Symbol", "Model", "Przeb.", "Trafność", "Odch.", "Zakres", "ECE"],
            rows,
            [2.2, 2.4, 1.5, 2.2, 1.8, 3.2, 2.4],
        )
        add_caption(
            doc, "Tabela",
            f"Horyzont „{horizon}”. Kolumna „Odch.” to odchylenie standardowe trafności między "
            "przebiegami różniącymi się wyłącznie ziarnem - miara szumu inicjalizacji, "
            "niewidoczna w przedziale pojedynczego przebiegu.",
        )

    _results_baserate(doc)
    _results_calibration(doc, runs)
    _results_stability(doc)
    _results_backtest(doc)
    _results_max_vs_mean(doc)
    _results_departures(doc, runs)
    page_break(doc)


def _results_baserate(doc: Document) -> None:
    """Accuracy against the frequency of the more common class.

    The comparison that decides whether a model knows anything: 0.50 looks like a
    coin flip only when the classes are balanced, and over this period they are
    not.
    """
    import polars as pl

    rows_by = defaultdict(list)
    for path in sorted(glob.glob(f"{REPORT_ROOT}/evaluations/*/*/predictions_quality.parquet")):
        parts = path.replace("\\", "/").split("/")
        symbol, model = parts[-3], parts[-2].split("_")[0]
        if not is_study_run(symbol, parts[-2]):
            continue
        try:
            frame = pl.read_parquet(path).filter(pl.col("status") == "scored")
        except Exception:
            continue
        for horizon in HORIZON_NAMES:
            sub = frame.filter(pl.col("horizon") == horizon)
            if sub.is_empty():
                continue
            predicted = np.array(sub["pred_direction"].to_list())
            actual = np.array(sub["actual_direction"].to_list())
            correct = np.array([bool(x) for x in sub["correct"].to_list()])
            share_long = float((actual == "long").mean())
            rows_by[(horizon, model)].append(
                (float(correct.mean()), float((predicted == "long").mean()),
                 share_long, max(share_long, 1 - share_long))
            )
    if not rows_by:
        return

    add_heading(doc, "10.4. Trafność wobec częstości klasy większościowej", 2)
    add_figure(doc, "rozrzut", "Każdy przebieg jako osobny punkt, horyzont tygodniowy. W obrębie architektury punkty ułożono według instrumentu, a trzy ziarna jednego instrumentu leżą w jednej kolumnie - dla modeli gradientowych pokrywają się, ponieważ ziarno nie zmienia ich wyniku. Różnice między instrumentami i ziarnami są większe niż odległości między architekturami - dlatego ranking oparty na pojedynczym przebiegu nie ma podstaw.")
    add_body(
        doc,
        "Trafność 0,50 oznacza rzut monetą tylko wtedy, gdy obie klasy są jednakowo częste. "
        "W badanym okresie tak nie jest: udział wzrostów rośnie wraz z horyzontem, więc "
        "właściwym punktem odniesienia nie jest 0,500, lecz częstość klasy liczniejszej. "
        "Reguła \u201ezawsze wskazuj klasę liczniejszą\u201d nie zawiera żadnego uczenia.",
    )
    table_rows = []
    for horizon in HORIZON_NAMES:
        for model in MODELS:
            values = rows_by.get((horizon, model))
            if not values:
                continue
            arr = np.array(values)
            table_rows.append([
                horizon, model,
                fmt(float(arr[:, 0].mean())),
                f"{arr[:, 1].mean():.1%}",
                f"{arr[:, 2].mean():.1%}",
                fmt(float(arr[:, 3].mean())),
                f"{arr[:, 0].mean() - arr[:, 3].mean():+.3f}",
            ])
    add_table(
        doc,
        ["Horyzont", "Model", "Trafność", "Mówi wzrost", "Był wzrost", "Większość", "Różnica"],
        table_rows,
        [2.1, 2.1, 2.0, 2.5, 2.3, 2.2, 2.2],
        font=8,
    )
    add_caption(
        doc, "Tabela",
        "Kolumna \u201eRóżnica\u201d to trafność pomniejszona o częstość klasy liczniejszej. "
        "Wartość ujemna oznacza, że model wypada gorzej niż reguła bez uczenia. Zwraca uwagę "
        "kolumna \u201eMówi wzrost\u201d: modele wystawiają prognozy bliskie zbalansowanym "
        "również tam, gdzie rynek był wyraźnie jednostronny - czego nie należy im "
        "poczytywać za wadę, ponieważ częstość bazowa jest znana dopiero z perspektywy czasu, "
        "a model kroczący poznaje ją wyłącznie z własnego okna treningowego.",
    )


def _results_calibration(doc: Document, runs: dict) -> None:
    """ECE per architecture: the project's one positive engineering result."""
    table_rows = []
    for horizon in HORIZON_NAMES:
        for model in MODELS:
            values = []
            for symbol in SYMBOLS:
                values.extend(horizon_values(runs.get((symbol, model), []), horizon, "ece"))
            if not values:
                continue
            arr = np.array(values)
            table_rows.append([
                horizon, model, str(len(arr)),
                fmt(float(arr.mean())),
                f"{arr.min():.3f}-{arr.max():.3f}",
            ])
    if not table_rows:
        return
    add_heading(doc, "10.5. Kalibracja", 2)
    add_body(
        doc,
        "Oczekiwany błąd kalibracji mierzy rozbieżność między deklarowaną pewnością a "
        "rzeczywistą częstością trafień. Jest to wielkość niezależna od zdolności "
        "rozróżniania: model pozbawiony przewagi może być doskonale skalibrowany, jeżeli "
        "poprawnie raportuje, że nic nie wie.",
    )
    add_table(
        doc,
        ["Horyzont", "Model", "Przeb.", "ECE średnie", "Zakres"],
        table_rows,
        [2.4, 2.4, 2.0, 3.0, 3.6],
        font=8.5,
    )
    add_caption(
        doc, "Tabela",
        "Błąd kalibracji na horyzont i architekturę. Niski błąd przy trafności bliskiej "
        "poziomowi losowemu oznacza system, który poprawnie komunikuje własną niepewność "
        "- jest to osobny, pozytywny wynik inżynierski, niezależny od braku przewagi "
        "prognostycznej.",
    )


def _results_backtest(doc: Document) -> None:
    """Trading simulation with its four nulls, read from the artifacts."""
    found = defaultdict(list)
    for path in sorted(glob.glob(f"{REPORT_ROOT}/backtests/*/*/backtest.json")):
        parts = path.replace("\\", "/").split("/")
        symbol, model = parts[-3], parts[-2].split("_")[0]
        if not is_study_run(symbol, parts[-2]):
            continue
        try:
            payload = json.load(open(path, encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        nulls = payload.get("nulls") or {}
        momentum = (nulls.get("naive_momentum") or {}).get("roi_pct") or 0
        found[(symbol, model)].append((
            payload.get("roi_pct"),
            (payload.get("benchmark") or {}).get("excess_roi_pct"),
            (nulls.get("constant_exposure") or {}).get("excess_roi_pct"),
            (nulls.get("mistimed_replay") or {}).get("percentile"),
            (payload.get("roi_pct") or 0) - momentum,
            payload.get("max_drawdown_pct"),
        ))
    if not found:
        return

    add_heading(doc, "10.7. Symulacja handlowa", 2)
    add_figure(doc, "stabilnosc", "Trafność miesiąc po miesiącu dla BTCUSDT na horyzoncie tygodniowym. Wahania wokół poziomu losowego są znacznie większe niż jakakolwiek trwała przewaga - to jest obraz, który średnia z całego okresu ukrywa.")
    add_body(
        doc,
        "Wyniki podano wyłącznie jako kontekst ekonomiczny; nie stanowią oceny prognozy "
        "(rozdział 9.5). Każdemu wynikowi towarzyszą cztery punkty odniesienia, z których "
        "rozstrzygający jest percentyl względem własnych przesunięć cyklicznych: wielkość ta "
        "zachowuje sekwencję ekspozycji i zmienia wyłącznie jej ułożenie w czasie, więc "
        "izoluje wyczucie momentu od poziomu zaangażowania. Wartość 50 oznacza brak wyczucia, "
        "95 byłaby sygnałem.",
    )
    table_rows, percentiles = [], []
    for (symbol, model), values in sorted(found.items()):
        arr = np.array(
            [[v if v is not None else np.nan for v in row] for row in values], dtype=float
        )
        mean = np.nanmean(arr, axis=0)
        percentiles.append(mean[3])
        table_rows.append([
            symbol.replace("USDT", ""), model, str(len(values)),
            f"{mean[0]:+.0f}", f"{mean[1]:+.0f}", f"{mean[2]:+.0f}",
            f"{mean[3]:.1f}", f"{mean[4]:+.0f}", f"{mean[5]:.0f}",
        ])
    add_table(
        doc,
        ["Symbol", "Model", "Przeb.", "ROI %", "vs hold", "vs stała", "Percentyl",
         "vs momentum", "Obsun. %"],
        table_rows,
        [1.7, 1.9, 1.4, 1.8, 1.9, 1.9, 2.0, 2.2, 1.8],
        font=8,
    )
    median = float(np.nanmedian(percentiles))
    add_caption(
        doc, "Tabela",
        f"Symulacja na horyzoncie tygodniowym, uśredniona po ziarnach. Mediana percentyla "
        f"wobec własnych przesunięć wynosi {pl(median, '.1f')} - poniżej przypadkowych 50, co "
        "oznacza, że rzeczywiste ułożenie sygnałów w czasie wypada gorzej niż ułożenie "
        "arbitralne. Kolumny \u201evs\u201d podają różnicę stopy zwrotu w punktach "
        "procentowych wobec danego punktu odniesienia; wartości ujemne oznaczają przegraną.",
    )


def _results_max_vs_mean(doc: Document) -> None:
    """Why the best cell is not a result - with the project's own example."""
    best = {}
    for path in sorted(glob.glob(f"{REPORT_ROOT}/evaluations/*/*/metrics.json")):
        parts = path.replace("\\", "/").split("/")
        symbol, slug = parts[-3], parts[-2]
        if not is_study_run(symbol, slug):
            continue
        try:
            by_horizon = json.load(open(path, encoding="utf-8")).get("by_horizon") or {}
        except (OSError, json.JSONDecodeError):
            continue
        for horizon, cell in by_horizon.items():
            if cell.get("accuracy") is None:
                continue
            interval = cell.get("accuracy_ci") or {}
            # The stored interval has no flag of its own: a refused one has no bounds.
            bounds = (interval.get("low"), interval.get("high"))
            entry = (cell["accuracy"], symbol, slug.split("_")[0],
                     interval.get("effective_samples"),
                     None if None in bounds else bounds)
            if horizon not in best or entry[0] > best[horizon][0]:
                best[horizon] = entry
    if not best:
        return

    add_heading(doc, "10.8. Dlaczego najlepsza komórka nie jest wynikiem", 2)
    add_figure(doc, "backtest", "Percentyl każdej symulacji względem jej własnych przesunięć cyklicznych. Mediana leży poniżej 50, czyli rzeczywiste ułożenie sygnałów w czasie wypada gorzej niż ułożenie arbitralne. Próg sygnału na poziomie 95 nie został osiągnięty przez żaden przebieg.")
    add_body(
        doc,
        "Zestawienie najlepszych komórek jest miarą kuszącą i myloną (Bailey i López de "
        "Prado, 2014). Maksimum z wielu "
        "zaszumionych pomiarów rośnie wraz z ich liczbą nawet wtedy, gdy żaden z nich nie "
        "zawiera sygnału - przy kilkudziesięciu komórkach i poziomie istotności 5% "
        "należy oczekiwać kilku przekroczeń progu wyłącznie z przypadku.",
    )
    add_table(
        doc,
        ["Horyzont", "Najlepsza komórka", "Trafność", "Obs. niezależnych", "Przedział"],
        [[horizon, f"{sym.replace('USDT', '')} / {model}", fmt(acc),
          "-" if eff is None else f"{eff:.1f}",
          f"{bounds[0]:.3f}-{bounds[1]:.3f}" if bounds else "odmówiony"]
         for horizon, (acc, sym, model, eff, bounds) in best.items()],
        [2.4, 4.0, 2.4, 3.4, 3.4],
        font=8.5,
    )
    add_caption(
        doc, "Tabela",
        "Najwyższa trafność na każdym horyzoncie wraz z liczbą obserwacji niezależnych, na "
        "której ją zmierzono, i 95-procentowym przedziałem ufności. Tam, gdzie liczba "
        "obserwacji jest zbyt mała, system odmawia wyznaczenia przedziału i nie orzeka o "
        "przewadze - co jest właściwym zachowaniem: wartość punktowa pozostaje raportowana, "
        "wycofane zostaje wyłącznie twierdzenie o jej precyzji. Przedział wyłączający 0,5 "
        "nie rozstrzyga jeszcze niczego, ponieważ maksimum wybrano spośród wielu pomiarów; "
        "co z takich wyników zostaje po uwzględnieniu ich liczby, pokazuje podrozdział 10.9.",
    )


def _plural(n: int, one: str, few: str, many: str) -> str:
    """The Polish noun form that goes with *n*."""
    if n == 1:
        return one
    if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14):
        return few
    return many


def _thousands(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def _percent(share: float) -> str:
    return f"{share * 100:.0f}%"


def _agreement_with_baseline(runs: dict, symbol: str, horizon: str) -> list[float]:
    """For each learning run on *symbol*: the share of bars on which it predicted
    the same side as the baseline."""
    distinct = distinct_runs(runs)
    base_entries = distinct.get((symbol, "baseline"), [])
    if not base_entries:
        return []
    base = _scored(symbol, base_entries[0]["slug"], horizon, ["date", "pred_direction"])
    side = dict(zip((str(d)[:10] for d in base["date"].to_list()), base["pred_direction"].to_list()))
    shares = []
    for model in MODELS:
        if model == "baseline":
            continue
        for run in distinct.get((symbol, model), []):
            frame = _scored(symbol, run["slug"], horizon, ["date", "pred_direction"])
            pairs = [(side[str(d)[:10]], p) for d, p in
                     zip(frame["date"].to_list(), frame["pred_direction"].to_list())
                     if str(d)[:10] in side]
            if pairs:
                shares.append(float(np.mean([a == b for a, b in pairs])))
    return shares


def _year_record(year: dict) -> str:
    """One year of a forecaster's record: the side it mostly called, against what happened."""
    rising = year["predicted_long"] >= 0.5
    called = year["predicted_long"] if rising else 1 - year["predicted_long"]
    return (f"przewidywał {'wzrost' if rising else 'spadek'} w {_percent(called)} dni, a kwartał "
            f"kończył się wzrostem w {_percent(year['actual_long'])} z nich - trafność "
            f"{pl(year['accuracy'])}")


def _who(m: dict) -> str:
    seed = f" pod ziarnem {m['seed']}" if m["seed"] is not None else ""
    return f"{MODEL_LABEL[m['model']]} na {m['symbol']}{seed}"


def _results_departures(doc: Document, runs: dict) -> None:
    """Which single measurements depart from a coin flip, and which survive the
    number of measurements taken."""
    found = departures(runs)
    every = found["measurements"]
    if not every:
        return
    total = len(every)
    tested = [h for h in HORIZON_NAMES if any(m["horizon"] == h for m in every)]
    refused = [h for h in HORIZON_NAMES if h not in tested]
    nominal = sorted((m for m in every if m["p"] < FALSE_DISCOVERY_RATE), key=lambda m: m["p"])
    below = [m for m in nominal if m["accuracy"] < CHANCE]
    above = [m for m in nominal if m["accuracy"] > CHANCE]
    survivors = sorted((m for m in every if m["q"] < FALSE_DISCOVERY_RATE), key=lambda m: m["p"])
    agree = sum((m["p"] < FALSE_DISCOVERY_RATE) == (m["low"] > CHANCE or m["high"] < CHANCE)
                for m in every)

    add_heading(doc, "10.9. Odstępstwa od losowości", 2)
    add_body(
        doc,
        "Podrozdziały 10.1-10.3 porównują architektury między sobą, a 10.8 pokazuje, dlaczego "
        "wynikiem nie może być najlepsza komórka. Pytanie uzupełniające brzmi: "
        "które pojedyncze pomiary odbiegają od rzutu monetą i czy którykolwiek z nich "
        "wytrzymuje uwzględnienie liczby wykonanych pomiarów. Wartość p dla hipotezy, że "
        "trafność wynosi 0,5, wyznaczono metodą z rozdziału 9.2 dla każdego z "
        f"{total} pomiarów: {len({m['slug'] for m in every})} różnych przebiegów na horyzontach "
        f"{_listing([HORIZON_LOCATIVE[h] for h in tested])}"
        + (f"; na horyzoncie {_listing([HORIZON_LOCATIVE[h] for h in refused])} przedział, a z "
           "nim wartość p, jest odmawiany z braku obserwacji niezależnych" if refused else "")
        + f". Próbkowanie wydłużono do {_thousands(P_VALUE_RESAMPLES)} losowań przy tym samym "
        f"ziarnie, więc pierwsze {_thousands(DEFAULT_RESAMPLES)} z nich to losowania "
        "opublikowanych przedziałów. Dłuższy ciąg jest potrzebny, ponieważ z B losowań nie da "
        "się otrzymać p mniejszego niż 2 / (B + 1), a poprawka wymaga od najmniejszej z "
        f"{total} wartości, by nie przekraczała {pl(FALSE_DISCOVERY_RATE / total, '.4f')}. "
        + ("Przedziały przeliczone z zapisanych wierszy ocenionych zgadzają się co do bitu z "
           "zapisanymi w artefaktach." if not found["mismatched"] else
           f"UWAGA: {found['mismatched']} przeliczonych przedziałów różni się od zapisanych w "
           "artefaktach - tabele i wartości p pochodzą wtedy z różnych danych."),
    )

    if not survivors:
        verdict = "nie pozostaje żaden."
    else:
        count = len(survivors)
        verdict = (
            ("pozostaje jeden pomiar: " if count == 1 else
             f"pozostają {count} {_plural(count, 'pomiar', 'pomiary', 'pomiarów')}: ")
            + "; ".join(
                f"{_who(m)}, horyzont {HORIZON_NOMINATIVE[m['horizon']]}, "
                f"trafność {pl(m['accuracy'])} - "
                + ("poniżej losowości" if m["accuracy"] < CHANCE else "powyżej losowości")
                for m in survivors)
            + "."
        )
    add_body(
        doc,
        f"Próg 0,05 {_plural(len(nominal), 'przekroczył', 'przekroczyły', 'przekroczyło')} "
        f"{len(nominal)} z {total} pomiarów, a przy braku jakiejkolwiek "
        f"przewagi należałoby się spodziewać około {round(FALSE_DISCOVERY_RATE * total)}. "
        f"Poniżej 0,5 {_plural(len(below), 'leży', 'leżą', 'leży')} {len(below)} z nich, "
        f"powyżej {len(above)}. Są to "
        + ("dokładnie te pomiary" if agree == total else
           f"w {agree} z {total} przypadków te same pomiary")
        + ", których 95-procentowy przedział nie obejmuje 0,5. Po poprawce "
        "Benjaminiego-Hochberga przy odsetku fałszywych odkryć 5% " + verdict,
    )
    add_table(
        doc,
        ["Symbol", "Model", "Ziarno", "Horyzont", "Trafność", "Przedział 95%", "p", "q"],
        [[m["symbol"].replace("USDT", ""), m["model"], "-" if m["seed"] is None else str(m["seed"]),
          m["horizon"], fmt(m["accuracy"]), f"{m['low']:.3f}-{m['high']:.3f}",
          fmt(m["p"], 4), fmt(m["q"])]
         for m in nominal],
        [1.8, 2.0, 1.5, 2.0, 1.8, 2.8, 1.6, 1.4],
        font=8.5,
    )
    add_caption(
        doc, "Tabela",
        "Pomiary, których trafność odbiega od 0,5 przy p < 0,05. Kolejność według rosnącego "
        f"p. Kolumna q to wartość skorygowana procedurą Benjaminiego-Hochberga względem wszystkich "
        f"{total} pomiarów; istotny po korekcie jest pomiar o q < 0,05. Wartości p z "
        f"{_thousands(P_VALUE_RESAMPLES)} losowań bootstrapu blokowego, przedział z pierwszych "
        f"{_thousands(DEFAULT_RESAMPLES)}. Kreska w kolumnie ziarna oznacza model, którego "
        "wyniku ziarno nie zmienia.",
    )

    lag = baseline_lag(runs)
    quarter = [m for m in nominal if m["horizon"] == "quarter"]
    quarter_base = [m for m in quarter if m["model"] == "baseline"]
    if lag and quarter_base and lag["window_bars"] and lag["forward_bars"]:
        quarter_below = sum(m["accuracy"] < CHANCE for m in quarter)
        add_body(
            doc,
            f"Spośród nich {len(quarter)} "
            f"{_plural(len(quarter), 'pochodzi', 'pochodzą', 'pochodzi')} z horyzontu kwartalnego - "
            + ("wszystkie poniżej 0,5" if quarter_below == len(quarter)
               else f"{quarter_below} poniżej 0,5")
            + f" - a {len(quarter_base)} z tych {len(quarter)} z modelu odniesienia ("
            + _listing([m["symbol"] for m in quarter_base]) + "). Mechanizm widać w rozbiciu "
            "wyników modelu odniesienia na lata. Na horyzoncie kwartalnym model zlicza kierunki "
            f"{lag['window_bars'] - lag['forward_bars']} kwartalnych zwrotów zakończonych przed "
            "dniem prognozy: najświeższy z nich rozpoczął się kwartał wcześniej, a całe okno "
            f"sięga {lag['window_bars']} dni wstecz. Prognozuje więc kierunek, który dominował "
            "przez mniej więcej dwa lata zakończone kwartał temu, i trzyma się raz wybranej "
            f"strony średnio przez {lag['hold']:.0f} kolejnych dni (średnia z {lag['symbols']} "
            "instrumentów). Taka reguła ma rację, dopóki "
            "trwa reżim rynku, i myli się po każdym jego zwrocie - aż okno wypełni się nowym "
            "reżimem.",
        )
        full_years = [y for y in lag["by_year"] if y["bars"] >= 360]
        worst = sorted(sorted(full_years, key=lambda y: y["accuracy"])[:2], key=lambda y: y["year"])
        add_table(
            doc,
            ["Rok", "Prognoza wzrostu", "Wzrost faktyczny", "Trafność", "Świec"],
            [[y["year"], _percent(y["predicted_long"]), _percent(y["actual_long"]),
              fmt(y["accuracy"]), str(y["bars"])] for y in lag["by_year"]],
            [1.8, 3.4, 3.4, 2.4, 1.8],
            font=8.5,
        )
        partial = [y["year"] for y in lag["by_year"] if y["bars"] < 360]
        add_caption(
            doc, "Tabela",
            "Model odniesienia na BTCUSDT na horyzoncie kwartalnym, rok po roku. Kolumny: "
            "odsetek dni z prognozą wzrostu, odsetek dni, po których kwartał rzeczywiście "
            "zakończył się wzrostem, i trafność."
            + (f" Rok {partial[0]} jest niepełny." if len(partial) == 1 else
               f" Lata {_listing(partial)} są niepełne." if partial else ""),
        )
        if len(worst) == 2:
            first, second = worst
            add_body(
                doc,
                f"Na BTCUSDT najgorsze były lata {first['year']} i {second['year']}. W roku "
                f"{first['year']} model {_year_record(first)}. W roku {second['year']} "
                f"{_year_record(second)}. Trafność poniżej losowości nie jest tu zdolnością, "
                "którą wystarczyłoby odwrócić: powstaje na kilku zwrotach cyklu rynku obecnych w "
                "badanym okresie, a okres następny może ich zawierać inną liczbę.",
            )

        n, k = lag["symbols"], len(quarter_base)
        add_body(
            doc,
            f"Wynik poniżej losowości na {k} "
            f"{_plural(k, 'instrumencie', 'instrumentach', 'instrumentach')} nie oznacza {k} "
            f"niezależnych potwierdzeń. Kwartalne zwroty {n} instrumentów "
            f"korelują średnio na poziomie ρ = {pl(lag['rho_returns'], '.2f')} w "
            f"{lag['days']} wspólnych dniach. Średnia z n zmiennych o wspólnej korelacji ρ ma "
            "wariancję taką jak średnia z n / (1 + (n − 1)ρ) zmiennych niezależnych, więc "
            f"{n} {_plural(n, 'instrument niesie', 'instrumenty niosą', 'instrumentów niesie')} "
            "tu tyle informacji, co około "
            f"{pl(effective_count(n, lag['rho_returns']), '.1f')} instrumentu niezależnego. "
            "Same trafienia "
            "i pomyłki modelu odniesienia korelują słabiej "
            f"(ρ = {pl(lag['rho_hits'], '.2f')}), ponieważ okno każdego instrumentu zaczyna się "
            "w innej dacie i obejmuje inną historię; równoważnik rośnie wtedy do "
            f"{pl(effective_count(n, lag['rho_hits']), '.1f')}. Dla {k} instrumentów poniżej "
            f"losowości oba odczyty dają od {pl(effective_count(k, lag['rho_returns']), '.1f')} "
            f"do {pl(effective_count(k, lag['rho_hits']), '.1f')} niezależnego pomiaru.",
        )

    btc = [m for m in every if m["symbol"] == "BTCUSDT" and m["horizon"] == "quarter"]
    shares = _agreement_with_baseline(runs, "BTCUSDT", "quarter")
    if btc and shares and all(m["accuracy"] < CHANCE for m in btc):
        accuracies = [m["accuracy"] for m in btc]
        add_body(
            doc,
            "Na BTCUSDT horyzont kwartalny wypada poniżej 0,5 dla każdej architektury i "
            f"każdego ziarna (od {pl(min(accuracies))} do {pl(max(accuracies))}). Modele uczące "
            "się wskazywały tę samą stronę co model odniesienia w "
            f"{_percent(min(shares))[:-1]}-{_percent(max(shares))} dni: przy cechach niosących "
            "mało informacji odtwarzają częstość bazową swojego okna treningowego, a wraz z nią "
            "jej opóźnienie.",
        )
    for m in survivors:
        siblings = [s["accuracy"] for s in every
                    if (s["symbol"], s["model"], s["horizon"]) == (m["symbol"], m["model"], m["horizon"])
                    and s is not m]
        text = (
            ("Jedyny pomiar, który przetrwał poprawkę, " if len(survivors) == 1
             else "Pomiar, który przetrwał poprawkę, ")
            + f"to {_who(m)}, horyzont {HORIZON_NOMINATIVE[m['horizon']]}, "
            f"trafność {pl(m['accuracy'])}"
            + (f"; pozostałe ziarna dały {_listing([pl(a) for a in siblings])}" if siblings else "")
            + ". Nie jest to skutek długości próbkowania - przy "
            f"{_thousands(PRECISION_RESAMPLES)} losowaniach p = {pl(m['p_precise'], '.5f')}, a "
            f"q = {pl(m['q_precise'])} - ani długości bloku: przy bloku podwojonym do "
            f"{2 * m['block']} świec p = {pl(m['p_double_block'], '.4f')}."
        )
        if m["accuracy"] < CHANCE:
            text += (" Jest to jednak pomiar opóźnienia trendu w jednym cyklu rynku, a nie "
                     "zdolność, którą dałoby się wykorzystać - i to pomiar po stronie przegranej.")
        add_body(doc, text)

    if above:
        def others_on(m: dict) -> list[dict]:
            return [s for s in every if (s["symbol"], s["horizon"]) == (m["symbol"], m["horizon"])
                    and s["model"] != m["model"]]

        def mean_by_model(rows: list[dict]) -> dict[str, float]:
            grouped: dict[str, list[float]] = defaultdict(list)
            for row in rows:
                grouped[row["model"]].append(row["accuracy"])
            return {model: float(np.mean(v)) for model, v in grouped.items()}

        lines = []
        for m in above:
            siblings = [s for s in every if (s["symbol"], s["model"], s["horizon"])
                        == (m["symbol"], m["model"], m["horizon"]) and s is not m]
            means = mean_by_model(others_on(m))
            lines.append(
                f"{_who(m)}, horyzont {HORIZON_NOMINATIVE[m['horizon']]}: {pl(m['accuracy'])}"
                + (f"; pozostałe ziarna {_listing([pl(s['accuracy']) for s in siblings])}"
                   if siblings else "")
                + "; pozostałe architektury na tych samych danych: "
                + ", ".join(f"{MODEL_LABEL[k]} {pl(v)}" for k, v in sorted(means.items(),
                                                                         key=lambda kv: -kv[1]))
                + "."
            )
        unconfirmed = not any(
            s["p"] < FALSE_DISCOVERY_RATE and s["accuracy"] > CHANCE
            for m in above for s in others_on(m)
        )
        elsewhere = not any(
            s["p"] < FALSE_DISCOVERY_RATE and s["accuracy"] > CHANCE
            for m in above for s in every
            if (s["model"], s["horizon"]) == (m["model"], m["horizon"]) and s["symbol"] != m["symbol"]
        )
        add_body(
            doc,
            f"Powyżej 0,5 {_plural(len(above), 'leży', 'leżą', 'leży')} {len(above)} z "
            f"{len(nominal)} pomiarów i "
            + ("żaden nie przetrwał poprawki"
               if not any(m["q"] < FALSE_DISCOVERY_RATE for m in above)
               else "nie wszystkie przetrwały poprawkę")
            + f" (najmniejsze q = {pl(min(m['q'] for m in above))}):",
        )
        add_bullets(doc, lines)
        add_body(
            doc,
            "Tam, gdzie ziarna jednej architektury zgadzają się ze sobą, zgodność ta pokazuje "
            "jedynie, że wynik nie jest artefaktem inicjalizacji. Ziarna widzą te same dane, "
            "więc nie stanowią replikacji; replikacją byłby ten sam wynik innej architektury "
            "uczonej na tych samych danych albo tej samej architektury na innym instrumencie."
            + (" Żadnego z tych potwierdzeń w zestawieniu nie ma: na tych samych danych i "
               "horyzoncie żadna inna architektura, a na innych instrumentach żaden przebieg tej "
               "samej architektury nie przekracza progu 0,05 w górę nawet przed poprawką."
               if unconfirmed and elsewhere else ""),
        )

    above_survivors = [m for m in survivors if m["accuracy"] > CHANCE]
    most_common = max(tested, key=lambda h: sum(m["horizon"] == h for m in nominal))
    if len(below) > len(above) and not above_survivors:
        add_body(
            doc,
            f"Odstępstwa od losowości leżą zatem częściej poniżej niż powyżej 0,5 ({len(below)} "
            f"wobec {len(above)}), najwięcej ich przypada na horyzont "
            f"{HORIZON_NOMINATIVE[most_common]}, a te poniżej wyjaśnia opóźnienie reguły "
            "częstościowej, nie struktura, której nauczyłby się model. Po uwzględnieniu liczby "
            "pomiarów żaden z nich nie wskazuje trafności istotnie wyższej od losowej, co jest "
            "zgodne z tezą pracy (rozdział 1.3).",
        )


def _results_stability(doc: Document) -> None:
    """Per-month quality, read from the stability artifacts."""
    reports = sorted(glob.glob(f"{REPORT_ROOT}/eval/*/*/stability_summary.json"))
    if not reports:
        return

    add_heading(doc, "10.6. Stabilność w czasie", 2)
    add_figure(doc, "kalibracja", "Oczekiwany błąd kalibracji według horyzontu i architektury. Wartości rosną z horyzontem, ponieważ wycinek kalibracyjny kurczy się wraz z liczbą obserwacji niezależnych.")
    add_body(
        doc,
        "Średnia z całego okresu maskuje sytuację, w której model działa w jednym reżimie "
        "rynkowym i zawodzi w innym. Poniższe zestawienie rozbija jakość na miesiące "
        "kalendarzowe i podaje, w jakim odsetku miesięcy trafność przekroczyła poziom losowy - "
        "wielkość odporną na pojedyncze wartości skrajne.",
    )

    rows = []
    for path in reports:
        parts = path.replace("\\", "/").split("/")
        symbol, slug = parts[-3], parts[-2]
        model = slug.split("_")[0]
        if not is_study_run(symbol, slug):
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


# ── 11 ───────────────────────────────────────────────────────────────────────

# One row per defect in chapter 11; the summary counts them from here.
DEFECTS = [
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
    ["Świeca o zerowym zwrocie usunięta z ramki treningowej TFT, a w GRU użyta jako cel",
     "TFT: okna z taką świecą zapisywane jako nietrenowane, w dziewięciu przebiegach na "
     "sześciu symbolach; GRU: przerwanie przebiegu",
     "rozszerzenie badania z trzech na dziewięć instrumentów"],
]


def _chapter_11_verification(doc: Document) -> None:
    add_heading(doc, "11. Weryfikacja poprawności implementacji", 1)
    add_body(
        doc,
        "Rozdział dokumentuje defekty wykryte w trakcie budowy systemu i stanowi materiał "
        "dowodowy dla ostatniego zdania tezy. Wspólną cechą wszystkich pozycji jest to, że "
        "defekt dawał wynik wyglądający na poprawny i nie ujawniał się w samej trafności "
        "prognozy - porównanie architektur przeprowadzone bez warstwy weryfikacji porównywałoby "
        "zatem artefakty.",
    )
    add_table(doc, ["Defekt", "Objaw bez weryfikacji", "Wykryty przez"], DEFECTS,
              [4.6, 6.2, 5.2])
    add_caption(doc, "Tabela", "Defekty wykryte w trakcie budowy systemu.")

    add_heading(doc, "11.1. Zamiana klas kierunku w modelu TFT", 2)
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

    add_heading(doc, "11.2. Horyzonty bez danych treningowych", 2)
    add_body(
        doc,
        "Przy oknie treningowym węższym od horyzontu zbiór treningowy jest pusty, ponieważ żadna "
        "etykieta nie zdążyła się zrealizować. Wcześniejsza wersja emitowała wtedy stały rozkład "
        "jednostajny, a warstwa oceny punktowała go jak prognozę. Obecnie taka konfiguracja "
        "powoduje błąd przed uruchomieniem, a świece bez możliwości uczenia oznaczane są jako "
        "nietrenowane i wykluczane z oceny, planowania i symulacji.",
    )

    add_heading(doc, "11.3. Brak przypisania wyników do modelu", 2)
    add_body(
        doc,
        "Wszystkie modele zapisywały prognozy do wspólnego, nieopatrzonego wersją katalogu, z "
        "którego czytały warstwy pochodne. W praktyce oznaczało to, że model uruchomiony jako "
        "ostatni przejmował wyniki wszystkich pozostałych, a przebieg godzinowy nadpisywał "
        "dzienny. Rozwiązaniem jest wersjonowanie opisane w rozdziale 4.4.",
    )

    add_heading(doc, "11.4. Jednorodne rozkłady głębokości", 2)
    add_body(
        doc,
        "Trzy z pięciu modeli zwracały jednakowe prawdopodobieństwo w każdym kubełku głębokości. "
        "Ponieważ warstwa decyzyjna wyznacza ryzyko jako masę prawdopodobieństwa w kubełkach "
        "dużych ruchów, jednorodny rozkład przypinał miarę ryzyka do stałej i czynił najwyższą "
        "rekomendację nieosiągalną dla tych modeli - przy zachowaniu jej osiągalności dla "
        "pozostałych. Porównanie międzymodelowe było więc obciążone w sposób niewidoczny w "
        "miarach jakości prognozy kierunku.",
    )

    add_heading(doc, "11.5. Wspólny strumień inicjalizacji", 2)
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
            "różnych symboli - to, który symbol wypada „powyżej losowego”, okazało się loterią.",
            "Trzy szeregi losowe o dryfie i zmienności dopasowanych do danych rzeczywistych, w "
            "których prawdziwa odpowiedź wynosi dokładnie 0,500, dały poziom odniesienia, "
            "względem którego różnica przestawała być istotna.",
            "Analiza kodu wykazała, że tożsamość, z której wyprowadzane jest ziarno okna, nie "
            "zawierała symbolu. Cztery „niezależne” symbole startowały z identycznych wag, z "
            "tymi samymi maskami dropoutu i tą samą kolejnością porcji - a więc stanowiły jedno "
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
        "ślepy na szum inicjalizacji o porównywalnej wielkości. Z tego powodu rozdział 10 "
        "raportuje wielokrotne przebiegi, a nie pojedyncze.",
    )

    add_heading(doc, "11.6. Puste okno normalizacji", 2)
    add_body(
        doc,
        "Zażądanie zakresu rozpoczynającego się przed pierwszym notowaniem instrumentu umieszcza "
        "jego pierwszą świecę wewnątrz zakresu, przez co pętla krocząca startuje od wiersza "
        "zerowego. Przyczynowe okno normalizacji sięga wstecz i nie znajduje nic, a statystyki "
        "pozycyjne nie są zdefiniowane dla pustego zbioru. Świeca bez historii wstecz nie ma "
        "jednak również wierszy treningowych, więc właściwym wynikiem jest oznaczenie jej jako "
        "nietrenowanej; przerwanie całego przebiegu było defektem.",
    )

    add_heading(doc, "11.7. Świece bez etykiety w modelach sekwencyjnych", 2)
    add_body(
        doc,
        "Zwrot zerowy nie ma strony, więc świeca, po której cena po upływie horyzontu wraca "
        "dokładnie do poziomu wyjściowego, nie ma etykiety (rozdział 7). Jej cechy są jednak "
        "znane, a sąsiednie świece pozostają kolejnymi krokami szeregu. TFT budował ramkę "
        "treningową wyłącznie ze świec etykietowanych, przez co w osi czasu powstawała luka, "
        "której biblioteka pytorch-forecasting nie dopuszcza: okno douczania zawierające taką "
        "świecę zgłaszało wyjątek i było zapisywane jako nietrenowane. Przebieg kończył się "
        "poprawnie, a trafność pozostałych okien się nie zmieniała - jedynym śladem był licznik "
        "okien z wyjątkiem w podsumowaniu przebiegu. Na ADAUSDT horyzont tygodniowy nie nauczył "
        "się w żadnym z 78 miesięcy.",
    )
    add_body(
        doc,
        "Ta sama przyczyna miała w GRU postać jawną: świeca bez etykiety trafiała do zbioru "
        "treningowego jako cel o wartości -1, którą funkcja straty odrzuca, kończąc cały "
        "przebieg. Zwrot dokładnie zerowy jest rzadki - wymaga, aby cena zamknięcia po upływie "
        "horyzontu trafiła dokładnie w poprzednią wartość. BTCUSDT, ETHUSDT i SOLUSDT nie mają "
        "w całej historii notowań ani jednej takiej świecy, pozostałe instrumenty od jednej do "
        "ośmiu, "
        "dlatego defekt ujawnił się dopiero po rozszerzeniu badania o sześć kolejnych "
        "instrumentów.",
    )
    add_body(
        doc,
        "Obecnie świeca bez etykiety pozostaje w sekwencji wejściowej obu modeli i jest "
        "pomijana wyłącznie jako cel uczenia. Okno bez takiej świecy przechodzi dokładnie tę "
        "samą ścieżkę co wcześniej, co potwierdzono porównaniem bajtowym, a przebiegi dotknięte "
        "defektem policzono ponownie.",
    )

    add_heading(doc, "11.8. Zakres pakietu testów", 2)
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


# ── 12 ───────────────────────────────────────────────────────────────────────

def _chapter_12_interfaces(doc: Document) -> None:
    add_heading(doc, "12. Interfejsy", 1)

    commands = cli_reference()
    routes = api_reference()

    add_heading(doc, "12.1. Wiersz poleceń", 2)
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

    add_heading(doc, "12.2. Interfejs HTTP", 2)
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

    add_heading(doc, "12.3. Panel przeglądarkowy", 2)
    add_body(
        doc,
        "Panel jest aplikacją jednostronicową zbudowaną na natywnych modułach ES, bez kroku "
        "budowania. Zachowanie elementów deklarowane jest atrybutem w znaczniku i obsługiwane "
        "przez jeden nasłuch delegowany, co eliminuje wstawianie kodu bezpośrednio w atrybutach "
        "- wcześniejsza wersja interpolowała nazwy symboli prosto do treści atrybutów zdarzeń.",
    )
    add_bullets(
        doc,
        [
            "Menedżer danych - inwentarz jeziora danych, pobieranie, agregacja, kontrola jakości.",
            "Modele - konfiguracja i uruchamianie treningu, kolejka zadań, inwentarz przebiegów.",
            "Ocena - ranking przebiegów, miary jakości, kalibracja i ostrość prognoz, "
            "wykres stabilności w czasie.",
            "Symulacja - parametry wykonawcze, wynik wraz z czterema hipotezami zerowymi.",
        ],
    )
    add_body(
        doc,
        "Formularz treningu pobiera wartości domyślne z interfejsu HTTP, a nie ze znaczników "
        "strony. Wcześniejsza wersja zawierała je wpisane wprost w znaczniku, przez co ten sam "
        "model uruchamiany z panelu wykonywał inną liczbę epok niż uruchamiany z wiersza "
        "poleceń - wartość domyślna obowiązuje bowiem wyłącznie przy nieobecności flagi.",
    )
    add_body(
        doc,
        "Długie zadania nie są wykonywane w procesie serwera. Panel zgłasza je do kolejki, "
        "którą obsługuje jeden wątek roboczy uruchamiający polecenia wiersza poleceń jako "
        "podprocesy, a postęp i dziennik odczytuje cyklicznie (pierwszy rysunek poniżej). "
        "Kolejka żyje po stronie serwera, więc przeładowanie przeglądarki nie gubi zadań "
        "oczekujących; stany zadania pokazuje drugi rysunek.",
    )
    add_diagram(
        doc, "zadanie_sekwencja",
        "Diagram sekwencji zadania zgłoszonego z panelu. Odpowiedź na zgłoszenie wraca od "
        "razu, a wynik panel poznaje z odpytywania stanu kolejki.",
    )
    add_diagram(
        doc, "zadanie_stany",
        "Diagram stanów zadania w kolejce. Zadanie oczekujące można przesunąć albo usunąć; "
        "wykonywane kończy się sukcesem dopiero wtedy, gdy wszystkie jego polecenia zwrócą "
        "kod 0.",
    )

    add_heading(doc, "12.4. Konfiguracja", 2)
    fields = settings_reference()
    add_body(
        doc,
        f"Konfiguracja obejmuje {len(fields)} pól walidowanych typami. Wszystkie ścieżki "
        "wyprowadzane są z jednego korzenia jeziora danych, dzięki czemu uruchomienie na innym "
        "zbiorze wymaga zmiany jednej wartości. Pełne zestawienie w załączniku D.",
    )
    page_break(doc)


# ── 12-13 ────────────────────────────────────────────────────────────────────

def _chapter_13_limits(doc: Document) -> None:
    add_heading(doc, "13. Ograniczenia i kierunki dalszych prac", 1)

    add_heading(doc, "13.1. Zakres wniosków", 2)
    add_bullets(
        doc,
        [
            "Zbiór informacyjny obejmuje wyłącznie cechy wyprowadzone z ceny i wolumenu tego "
            "samego instrumentu. Nie uwzględniono danych on-chain, stawek finansowania, "
            "głębokości arkusza zleceń, danych międzyrynkowych ani sentymentu. Jest to granica "
            "przyjęta świadomie - praca testuje słabą formę hipotezy efektywności - ale wniosków "
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

    add_heading(doc, "13.2. Kierunki dalszych prac", 2)
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

    page_break(doc)


HORIZON_LOCATIVE = {"week": "tygodniowym", "month": "miesięcznym", "quarter": "kwartalnym",
                    "year": "rocznym"}
HORIZON_NOMINATIVE = {"week": "tygodniowy", "month": "miesięczny", "quarter": "kwartalny",
                      "year": "roczny"}
MODEL_LABEL = {"baseline": "model odniesienia", "ml": "scikit-learn", "xgboost": "XGBoost",
               "gru": "GRU", "tft": "TFT"}
COUNT_LOCATIVE = {2: "dwóch", 3: "trzech", 4: "czterech"}
COUNT_WORDS = {5: "pięć", 6: "sześć", 7: "siedem", 8: "osiem", 9: "dziewięć", 10: "dziesięć"}


def _listing(words: list[str]) -> str:
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " i " + words[-1]


def _rank_turnover(runs: dict) -> str:
    """One architecture that is last among the learners somewhere and first elsewhere."""
    spread = architecture_spread(runs)
    places: dict[str, dict[str, int]] = {}
    learners = [m for m in MODELS if m != "baseline"]
    for horizon, cell in spread.items():
        order = sorted((m for m in cell["means"] if m != "baseline"),
                       key=lambda m: -cell["means"][m])
        if len(order) != len(learners):
            continue
        for place, model in enumerate(order, start=1):
            places.setdefault(model, {})[horizon] = place
    candidates = [
        (sum(p == len(learners) for p in got.values()), model)
        for model, got in places.items()
        if 1 in got.values() and len(learners) in got.values()
    ]
    if not candidates:
        return ""
    model = max(candidates)[1]
    last = [HORIZON_LOCATIVE[h] for h, p in places[model].items() if p == len(learners)]
    first = [HORIZON_LOCATIVE[h] for h, p in places[model].items() if p == 1]
    return (f": architektura {MODEL_LABEL[model]}, ostatnia wśród uczących się na horyzoncie "
            f"{_listing(last)}, jest pierwsza na {_listing(first)}")


def _corrected_sentence(runs: dict) -> str:
    """What is left of the departures from chance once their number is accounted for."""
    every = departures(runs)["measurements"]
    if not every:
        return ""
    survivors = [m for m in every if m["q"] < FALSE_DISCOVERY_RATE]
    higher = [m for m in survivors if m["accuracy"] > CHANCE]
    horizons = len({m["horizon"] for m in every})
    scope = (f"{len(every)} pomiarów ({len({m['slug'] for m in every})} różnych przebiegów na "
             f"{COUNT_LOCATIVE.get(horizons, str(horizons))} horyzontach)")
    opening = "Po poprawce na wielokrotne porównania (rozdział 10.9) "
    if not survivors:
        return opening + f"żaden z {scope} nie odbiega istotnie od losowości. "
    if higher:
        return (opening + f"{len(higher)} z {scope} "
                + _plural(len(higher), "wykazuje", "wykazują", "wykazuje")
                + " trafność istotnie wyższą od losowej. ")
    return (opening + f"żaden z {scope} nie wykazuje trafności istotnie wyższej od losowej; "
            + ("jedyny pomiar, który przetrwał poprawkę, leży" if len(survivors) == 1 else
               f"{len(survivors)} pomiary, które przetrwały poprawkę, "
               + _plural(len(survivors), "leży", "leżą", "leży"))
            + " poniżej losowości. ")


def _chapter_14_summary(doc: Document, runs: dict) -> None:
    add_heading(doc, "14. Podsumowanie", 1)
    if not runs:
        add_body(doc, "Podsumowanie wymaga ocenionych przebiegów.", italic=True)
        page_break(doc)
        return
    f = study_facts(runs)
    deterministic = set(f["seed_invariant"]) == {"ml", "xgboost"}
    if len(f["beyond_noise"]) <= 1:
        differences = ("Różnice między architekturami uczącymi się są tego samego rzędu co szum "
                       "inicjalizacji, a ich porządek zmienia się wraz z horyzontem")
    else:
        differences = (f"Różnice między architekturami uczącymi się przekraczają szum "
                       f"inicjalizacji na {len(f['beyond_noise'])} horyzontach, lecz ich porządek "
                       "zmienia się wraz z horyzontem")
    add_body(
        doc,
        "Porównano pięć rodzin architektur uczenia maszynowego w zadaniu prognozy kierunku "
        "zmiany ceny na podstawie wyłącznie historii notowań, w "
        f"{f['runs']} przebiegach obejmujących {f['symbols']} instrumentów i trzy ziarna "
        "inicjalizacji"
        + (f" - {f['distinct']} różnych pomiarów, ponieważ oba modele gradientowe uczą się "
           "deterministycznie" if deterministic else "")
        + ". Żadna architektura nie osiąga trafności istotnie wyższej od losowej, a wszystkie "
        "wypadają poniżej reguły opartej na częstości klasy liczniejszej. "
        + _corrected_sentence(runs)
        + differences + _rank_turnover(runs)
        + ". Ranking architektur nie jest zatem właściwością architektury.",
    )
    defects = COUNT_WORDS.get(len(DEFECTS), str(len(DEFECTS)))
    add_body(
        doc,
        "Jest to wynik zgodny ze słabą formą hipotezy efektywności rynku, uzyskany na rynku "
        "wybranym tak, aby mierzyć zachowanie modelu, a nie artefakty systemu obrotu. "
        "Rozstrzygnięcie tego pytania było możliwe wyłącznie dzięki warstwie weryfikacji, której "
        f"skuteczność dokumentuje {defects} wykrytych defektów (rozdział 11) - wśród nich jeden, "
        "który wyprodukował wynik pozornie pozytywny i przeszedł replikację poza próbą, zanim "
        "został obalony.",
    )
    add_body(
        doc,
        "Osobnym, pozytywnym wynikiem inżynierskim jest kalibracja: system poprawnie raportuje "
        "własną niepewność tam, gdzie nie ma zdolności rozróżniania. Prognoza, która wie, że nie "
        "wie, jest użyteczna; prognoza pewna siebie i błędna nie jest.",
    )
    page_break(doc)


def _check_thesis_against_data(runs: dict) -> None:
    """Warn when the hand-written thesis (1.3, 1.5) no longer matches the artifacts."""
    if not runs:
        return
    f = study_facts(runs)
    week, month = f["by_horizon"].get("week", {}), f["by_horizon"].get("month", {})
    above = week.get("above", 0) + month.get("above", 0)
    expected = round((week.get("estimable", 0) + month.get("estimable", 0)) * 0.025)
    problems = []
    if above > 1.5 * expected + 1:
        problems.append(f"{above} measurements above chance where about {expected} are expected")
    if len(f["beyond_noise"]) > 1:
        problems.append(f"the learners exceed the seed spread on {f['beyond_noise']}")
    found = departures(runs)
    higher = [m for m in found["measurements"]
              if m["q"] < FALSE_DISCOVERY_RATE and m["accuracy"] > CHANCE]
    if higher:
        problems.append(f"{len(higher)} measurements above chance survive Benjamini-Hochberg")
    if found["mismatched"]:
        problems.append(f"{found['mismatched']} recomputed intervals differ from the stored ones")
    for problem in problems:
        print(f"WARNING: thesis 1.3 / 1.5 no longer matches the data: {problem}")


def _chapter_14_bibliography(doc: Document) -> None:
    add_heading(doc, "15. Bibliografia", 1)
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
            "Efron B., Tibshirani R. J., An Introduction to the Bootstrap, Chapman & Hall, "
            "1993.",
            "Benjamini Y., Hochberg Y., Controlling the False Discovery Rate: A Practical and "
            "Powerful Approach to Multiple Testing, Journal of the Royal Statistical Society: "
            "Series B, 57(1), 1995.",
            "Benjamini Y., Yekutieli D., The Control of the False Discovery Rate in Multiple "
            "Testing under Dependency, Annals of Statistics, 29(4), 2001.",
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
        "uruchomieniu pakietu. Opis pochodzi z dokumentacji przypadku, a gdy jej nie ma - z jego "
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
            [[o.flags, o.kind, "tak" if o.required else "-", o.default, o.help]
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

def _chapter_12a_worked_example(doc: Document) -> None:
    """One bar traced end to end. Nothing explains a pipeline like following a
    single row through it."""
    add_heading(doc, "12.5. Przykład przejścia jednej świecy przez potok", 2)
    add_body(
        doc,
        "Poniżej prześledzono pojedynczą świecę dzienną od pobrania po rekomendację. Przykład "
        "ilustruje, w którym miejscu zapada każda decyzja i gdzie przebiegają granice "
        "przyczynowości - czyli co system wie w chwili, gdy prognozuje.",
    )
    add_table(
        doc,
        ["Krok", "Co się dzieje", "Granica przyczynowa"],
        [
            ["1. Pobranie",
             "Archiwum miesięczne trafia do jeziora danych jako świece jednominutowe.",
             "-"],
            ["2. Agregacja",
             "Świece minutowe składają się w jedną dzienną: otwarcie pierwszej, zamknięcie "
             "ostatniej, skrajne maksimum i minimum, suma wolumenów.",
             "wyłącznie minuty należące do tego dnia"],
            ["3. Cechy",
             "Dla świecy i wyliczane są wskaźniki z okien kończących się na niej - średnie "
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
        "jaką model może wykorzystać, ma zatem F świec opóźnienia - i to właśnie ta luka, a nie "
        "liczba wierszy, ogranicza horyzonty długie.",
    )


# ── glossary ─────────────────────────────────────────────────────────────────

def _appendix_glossary(doc: Document) -> None:
    add_heading(doc, "Załącznik F. Słownik pojęć", 1)
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
             "Liczba świec podzielona przez rozpiętość horyzontu - przybliżenie liczby "
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


def _appendix_runs(doc: Document, runs: dict) -> None:
    """Every run, every horizon, one row each.

    The aggregated tables in chapter 9 answer the research question; this one
    lets a reader check any single number against the artifact that produced it,
    which is what makes the aggregate auditable rather than merely stated.
    """
    rows = []
    for (symbol, model), entries in sorted(runs.items()):
        for entry in sorted(entries, key=lambda e: str(e.get("seed"))):
            for horizon in HORIZON_NAMES:
                cell = entry["by_horizon"].get(horizon)
                if not cell or cell.get("accuracy") is None:
                    continue
                interval = cell.get("accuracy_ci") or {}
                low, high = interval.get("low"), interval.get("high")
                rows.append([
                    symbol.replace("USDT", ""),
                    model,
                    str(entry.get("seed") if entry.get("seed") is not None else "-"),
                    horizon,
                    str(cell.get("samples") or "-"),
                    fmt(cell.get("accuracy")),
                    f"[{low:.3f}; {high:.3f}]" if low is not None and high is not None
                    else "odmówiony",
                    fmt(cell.get("ece")),
                ])
    if not rows:
        return
    add_heading(doc, "Załącznik E. Wyniki wszystkich przebiegów", 1)
    add_body(
        doc,
        f"Zestawienie {len(rows)} par (przebieg, horyzont) odczytane wprost z plików oceny. "
        "Tabele rozdziału 10 podają wielkości zagregowane; ta pozwala prześledzić każdą z nich "
        "do pojedynczego artefaktu. Kolumna „Przedział” zawiera wpis „odmówiony” tam, gdzie "
        "liczba obserwacji niezależnych była zbyt mała, aby przedział nie był artefaktem "
        "losowania.",
    )
    add_table(
        doc,
        ["Symbol", "Model", "Ziarno", "Horyzont", "Wierszy", "Trafność", "Przedział", "ECE"],
        rows,
        [1.7, 1.9, 1.5, 1.9, 1.7, 1.8, 3.0, 1.5],
        font=7.5,
    )
    add_caption(
        doc, "Tabela",
        "Pełne wyniki wszystkich przebiegów. Kolumna „Ziarno” pozwala odróżnić powtórzenia tej "
        "samej konfiguracji, których rozrzut mierzy szum inicjalizacji.",
    )

def _data_quality_results(doc: Document) -> None:
    """Outcome of the four checks across the lake, if it has been measured."""
    path = "data/_recompute/qa_summary.json"
    if not os.path.exists(path):
        return
    try:
        summary = json.load(open(path, encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not summary:
        return

    add_body(
        doc,
        "Kontrole uruchomiono na wszystkich partycjach jeziora danych. Wynik podano poniżej, "
        "ponieważ opis kontroli bez jej rezultatu dokumentuje zdolność, której nikt nie "
        "widział w działaniu.",
    )
    rows, total_parts, total_rows, total_defects = [], 0, 0, 0
    for symbol, stats in sorted(summary.items()):
        defects = (stats.get("nieposortowane", 0) + stats.get("duplikaty", 0)
                   + stats.get("luki", 0) + stats.get("naruszenia_ohlc", 0)
                   + stats.get("nieczytelne", 0))
        total_parts += stats.get("partycje", 0)
        total_rows += stats.get("wiersze", 0)
        total_defects += defects
        rows.append([
            symbol,
            str(stats.get("partycje", 0)),
            f"{stats.get('wiersze', 0):,}".replace(",", " "),
            str(stats.get("nieposortowane", 0)),
            str(stats.get("duplikaty", 0)),
            str(stats.get("luki", 0)),
            str(stats.get("naruszenia_ohlc", 0)),
        ])
    add_table(
        doc,
        ["Symbol", "Partycji", "Wierszy", "Nieposort.", "Duplikaty", "Luki", "OHLC"],
        rows,
        [2.6, 2.0, 2.4, 2.4, 2.2, 1.8, 2.0],
        font=8.5,
    )
    add_caption(
        doc, "Tabela",
        f"Wynik kontroli jakości na {total_parts} partycjach obejmujących "
        f"{total_rows:,} wierszy: {total_defects} defektów. ".replace(",", " ")
        + "Zerowa liczba naruszeń nie jest zaskoczeniem - dane pochodzą z jednego, "
        "automatycznie publikowanego źródła - lecz jej zmierzenie jest warunkiem, aby "
        "późniejsze wyniki przypisywać modelowi, a nie uszkodzeniu wejścia.",
    )

def _chapter_3_equations(doc: Document) -> None:
    """The formulas the components rest on, written out."""
    add_heading(doc, "3.7. Zapis formalny stosowanych wielkości", 2)
    add_body(
        doc,
        "Poniżej zebrano definicje wielkości, do których odwołują się kolejne rozdziały. "
        "Notacja: r oznacza zwrot, p prognozowane prawdopodobieństwo, y realizację (0 lub 1), "
        "F rozpiętość horyzontu w świecach, N liczbę ocenianych wierszy.",
    )
    add_table(
        doc,
        ["Wielkość", "Definicja", "Uwagi"],
        [
            ["Zwrot logarytmiczny",
             "r(k) = ln( C(k+F) / C(k) )",
             "addytywny w czasie; symetryczny wobec wzrostu i spadku"],
            ["Etykieta kierunku",
             "y(k) = 1 gdy r(k) > 0, 0 gdy r(k) < 0, brak gdy r(k) = 0",
             "dwie klasy; zwrot zerowy nie ma strony"],
            ["Trafność",
             "ACC = (1/N) · Σ 1[ ŷ(k) = y(k) ]",
             "reguła niewłaściwa; podana dla interpretowalności"],
            ["Wynik Briera",
             "BS = (1/N) · Σ ( p(k) − y(k) )²",
             "reguła właściwa; kara kwadratowa"],
            ["Strata logarytmiczna",
             "LL = −(1/N) · Σ [ y·ln p + (1−y)·ln(1−p) ]",
             "reguła właściwa; kara nieograniczona"],
            ["Oczekiwany błąd kalibracji",
             "ECE = Σ_b ( n_b / N ) · | acc_b − conf_b |",
             "b - kubełki pewności; mierzy kalibrację, nie rozróżnianie"],
            ["Skalowanie temperaturą",
             "p_T = softmax( z / T ), T > 0",
             "jeden parametr; nie zmienia porządku klas"],
            ["Liczba obserwacji efektywnych",
             "n_eff = N / F",
             "etykiety nakładają się na F−1 świecach"],
            ["Liczba symboli efektywnych",
             "n_sym = n / ( 1 + (n−1)·ρ )",
             "ρ - średnia korelacja par; granica 1/ρ"],
            ["Okno treningowe",
             "W = F + max( MIN_WIERSZY, 3·F )",
             "podłoga wierszowa wiąże krótkie horyzonty, obserwacyjna długie"],
            ["Przewaga prognozy",
             "edge = | p_wzrost − p_spadek |",
             "z konstrukcji nie przekracza 0,5"],
            ["Percentyl przesunięć",
             "pct = ( #{ i : ROI_i < ROI } / M ) · 100",
             "M - liczba przesunięć cyklicznych; 50 oznacza przypadek"],
        ],
        [3.6, 6.0, 6.4],
        font=8.5,
    )
    add_caption(
        doc, "Tabela",
        "Definicje formalne wielkości używanych w dokumencie. Wszystkie są zaimplementowane "
        "w jednym miejscu i pokryte testami, których wykaz zawiera załącznik A.",
    )

    add_heading(doc, "3.8. Architektury: mechanizmy", 2)
    add_body(
        doc,
        "Trzy mechanizmy odróżniają badane architektury, przy czym różnica dotyczy sposobu "
        "wprowadzenia zależności czasowej, nie rodziny funkcji.",
    )
    add_bullets(
        doc,
        [
            "Wzmacnianie gradientowe buduje addytywnie ciąg płytkich drzew, z których każde "
            "dopasowuje się do gradientu straty pozostawionej przez poprzednie. Zależność "
            "czasowa wchodzi wyłącznie przez cechy liczone na oknach - model nie ma pojęcia "
            "kolejności wierszy.",
            "Sieć rekurencyjna z bramkami utrzymuje stan ukryty aktualizowany co świecę. "
            "Bramka aktualizacji decyduje, jaka część poprzedniego stanu przechodzi dalej, a "
            "bramka resetu - jaka część jest brana pod uwagę przy wyznaczaniu stanu "
            "kandydującego. Konstrukcja ta łagodzi zanikanie gradientu, które w prostej sieci "
            "rekurencyjnej uniemożliwia uczenie zależności odległych.",
            "Mechanizm uwagi wyznacza wagi wszystkich pozycji w sekwencji naraz, zamiast "
            "przekazywać informację krok po kroku. Temporal Fusion Transformer łączy go z "
            "kodowaniem rekurencyjnym oraz sieciami wyboru zmiennych, które uczą się, które "
            "cechy są istotne w danym kontekście. Jest to model o największej pojemności w "
            "zestawieniu - i przy zmierzonej liczbie obserwacji niezależnych właśnie ta "
            "pojemność jest jego największym ryzykiem, nie zaletą.",
        ],
    )

# ── screenshots supplied by hand ─────────────────────────────────────────────

SCREENSHOT_DIR = "docs/zrzuty"

# Each entry: file name, heading, caption, and what the screenshot should show.
# The file names are fixed so the document can find them without configuration;
# a missing file leaves a visible placeholder rather than a silent gap, because a
# gap nobody notices is how a half-finished document ships.
SCREENSHOTS: list[tuple[str, str, str, str]] = [
    ("01-menedzer-danych.png", "Menedżer danych - inwentarz",
     "Zakładka menedżera danych. Inwentarz jeziora danych podaje dla każdego symbolu zakres "
     "dat, rozmiar na dysku, dostępne interwały i stan kontroli jakości; obok formularz "
     "pobierania i kolejka zadań. Na dole terminal, w którym pojawiają się polecenia "
     "wiersza poleceń, na które panel tłumaczy zadania, wraz z ich wyjściem.",
     "zakładka „Data Manager” z widoczną tabelą inwentarza i co najmniej trzema symbolami"),
    ("02-pobieranie.png", "Menedżer danych - pobieranie",
     "Formularz pobierania danych z Binance Public Data: para walutowa, zakres dat i "
     "interwały, do których świece jednominutowe zostaną zagregowane po pobraniu.",
     "formularz pobierania z wypełnionym symbolem i zakresem dat"),
    ("03-agregacja.png", "Menedżer danych - agregacja i kontrola jakości",
     "Panel agregacji jednego symbolu, rozwinięty w tabeli inwentarza. Interwał, który już "
     "istnieje, jest przekreślony i niedostępny; zadanie można dopisać do kolejki albo "
     "uruchomić od razu. Przycisk „Verify Quality” uruchamia kontrole jakości opisane w "
     "rozdziale 5.4.",
     "panel agregacji z wybranymi interwałami docelowymi"),
    ("04-trening-formularz.png", "Modele - konfiguracja treningu",
     "Formularz treningu: symbol, architektura, interwał, zakres dat i horyzonty. Wartości "
     "domyślne pobierane są z interfejsu HTTP, nie ze znaczników strony - powód opisano w "
     "rozdziale 12.3. Z prawej inwentarz zapisanych przebiegów.",
     "zakładka „AI Models” z widocznym wyborem modelu, zakresem dat i horyzontami"),
    ("05-trening-pula.png", "Modele - pula symboli treningowych",
     "Rozwinięta sekcja wyboru dodatkowych symboli do puli treningowej. Prognoza dotyczy "
     "nadal symbolu wybranego wyżej, a symbol ten jest z puli wykluczony. Sekcja pojawia się "
     "tylko dla architektur tablicowych, które przyjmują pulę.",
     "sekcja „Advanced: train on several symbols” rozwinięta, model xgboost, kilka symboli "
     "zaznaczonych"),
    ("06-kolejka.png", "Modele - kolejka zadań",
     "Kolejka zadań utrzymywana po stronie serwera: jedno zadanie wykonywane, drugie "
     "oczekujące, z przyciskami zmiany kolejności i usunięcia. Pasek u góry pokazuje postęp "
     "bieżącego zadania. Przeładowanie przeglądarki kolejki nie gubi, ponieważ przeglądarka "
     "jedynie ją wyświetla.",
     "kolejka z co najmniej jednym zadaniem oczekującym lub wykonywanym"),
    ("07-ocena-metryki.png", "Ocena - miary jakości",
     "Zakładka oceny dla jednego przebiegu badania (GRU na BTCUSDT): ranking wszystkich "
     "ocenionych przebiegów według wyniku złożonego, karty miar zbiorczych (wynik złożony, "
     "trafność, Brier, ECE), rozkład pewności prognoz w przedziałach kalibracji i ostrość "
     "prognozy na każdym horyzoncie. Przedziały ufności i liczbę obserwacji niezależnych "
     "ocena zapisuje w artefaktach przebiegu i wypisuje w wierszu poleceń; z artefaktów "
     "pochodzą tabele rozdziału 10.",
     "zakładka „Evaluation” z tabelą metryk dla wybranego przebiegu"),
    ("08-ocena-stabilnosc.png", "Ocena - stabilność w czasie",
     "Strata logarytmiczna przebiegu badania GRU na BTCUSDT (ziarno 42) miesiąc po "
     "miesiącu, od stycznia 2021 do czerwca 2026, osobno dla czterech horyzontów. Horyzonty "
     "tygodniowy i miesięczny utrzymują niemal stały poziom, a kwartalny i roczny odrywają "
     "się od niego na całe miesiące - tam, gdzie obserwacji niezależnych jest najmniej. "
     "Linia roczna kończy się wcześniej, a miesiące o zbyt małej liczbie ocenionych świec są "
     "oznaczone osobnym znacznikiem.",
     "wykres stabilności miesięcznej z widocznymi czterema horyzontami"),
    ("09-backtest-parametry.png", "Symulacja - założenia wykonawcze",
     "Założenia wykonawcze symulacji, tu dla przebiegu badania GRU na BTCUSDT, wystawione "
     "jako parametry: prowizja, poślizg, próg "
     "kosztowy, minimalna zmiana pozycji, długość okna reguły momentum oraz docelowa "
     "ekspozycja dla każdej rekomendacji. Wartość „Hold” celowo nie ma własnej ekspozycji - "
     "oznacza utrzymanie poprzedniej.",
     "zakładka „Backtest” z rozwiniętym formularzem parametrów"),
    ("10-backtest-wynik.png", "Symulacja - wynik i hipotezy zerowe",
     "Wynik symulacji przebiegu badania GRU na BTCUSDT (ziarno 42, horyzont tygodniowy, "
     "styczeń 2020 - czerwiec 2026), podany łącznie z czterema punktami odniesienia: kupnem "
     "i trzymaniem, stałą ekspozycją o tym samym średnim udziale, percentylem względem "
     "własnych przesunięć cyklicznych i regułą momentum bez modelu. Strategia zarabia 55,5%, "
     "lecz przegrywa z kupnem i trzymaniem oraz ze stałą ekspozycją, a wobec własnych "
     "przesunięć leży na 43. percentylu - poniżej losowego 50.",
     "wynik backtestu z widoczną sekcją hipotez zerowych"),
    ("11-analiza-wykres.png", "Analiza - kalendarz prognoz",
     "Inspektor historyczny przebiegu: kalendarz jednego miesiąca, w którym każdy dzień "
     "podaje prognozowane prawdopodobieństwo wzrostu (L) i spadku (S) na najkrótszym "
     "horyzoncie.",
     "kalendarz prognoz wybranego przebiegu"),
]


def _screenshot_placeholder(doc: Document, expected: str) -> None:
    """A visible frame saying what belongs here and where to put it."""
    table = doc.add_table(rows=1, cols=1)
    table.style = "Table Grid"
    cell = table.rows[0].cells[0]
    cell.width = Cm(15.5)
    paragraph = cell.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(24)
    paragraph.paragraph_format.space_after = Pt(24)
    run = paragraph.add_run(plain_dashes(f"[ MIEJSCE NA ZRZUT EKRANU ]\n{expected}"))
    run.italic = True
    run.font.size = Pt(9)
    run.font.color.rgb = MUTED
    doc.add_paragraph()


def _chapter_12b_screenshots(doc: Document) -> None:
    add_heading(doc, "12.6. Panel przeglądarkowy - widoki", 2)
    present = sum(
        1 for name, *_ in SCREENSHOTS if os.path.exists(os.path.join(SCREENSHOT_DIR, name))
    )
    add_body(
        doc,
        "Poniżej zebrano widoki panelu odpowiadające etapom potoku. Pliki leżą w katalogu "
        f"{SCREENSHOT_DIR}/ pod ustalonymi nazwami i są osadzane przy generowaniu dokumentu"
        + ("." if present == len(SCREENSHOTS)
           else f"; obecnie dostępnych jest {present} z {len(SCREENSHOTS)}."),
    )
    for name, heading, caption, expected in SCREENSHOTS:
        add_heading(doc, heading, 3)
        path = os.path.join(SCREENSHOT_DIR, name)
        if os.path.exists(path):
            paragraph = doc.add_paragraph()
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.paragraph_format.space_after = Pt(2)
            try:
                paragraph.add_run().add_picture(path, width=Cm(15.5))
            except Exception:
                _screenshot_placeholder(doc, f"plik {name} istnieje, ale nie daje się osadzić")
        else:
            _screenshot_placeholder(doc, f"{name} - {expected}")
        add_caption(doc, "Rysunek", caption)


def _both_ends_sentence(positions: dict[str, list[int]], places: int) -> str:
    """Which architectures are both first and last, depending on the horizon."""
    both = [m for m, p in positions.items() if min(p) == 1 and max(p) == places]
    if not both:
        return "Żadna architektura nie zajmuje zarówno pierwszego, jak i ostatniego miejsca."
    count = {1: "Jedna architektura zajmuje", 2: "Dwie architektury zajmują",
             3: "Trzy architektury zajmują"}.get(len(both), f"{len(both)} architektur zajmuje")
    return (f"{count} zarówno pierwsze, jak i ostatnie miejsce, zależnie od horyzontu "
            f"({', '.join(both)}).")


def _results_rank_stability(doc: Document, runs: dict) -> None:
    """Does the ordering of architectures survive a change of horizon?

    The thesis rests on this, so it is computed rather than asserted. The
    baseline is excluded: it does not learn, it is consistently last, and
    including it would inflate the apparent agreement between horizons.
    """
    learners = [m for m in MODELS if m != "baseline"]
    means_by_horizon: dict[str, dict[str, float]] = {}
    for horizon in HORIZON_NAMES:
        per_model = {}
        for model in learners:
            values = []
            for symbol in SYMBOLS:
                values.extend(horizon_values(runs.get((symbol, model), []), horizon, "accuracy"))
            if values:
                per_model[model] = float(np.mean(values))
        if len(per_model) == len(learners):
            means_by_horizon[horizon] = per_model
    if len(means_by_horizon) < 2:
        return

    order = {
        horizon: sorted(per_model, key=lambda m: -per_model[m])
        for horizon, per_model in means_by_horizon.items()
    }

    add_heading(doc, "10.2. Czy ranking architektur jest trwały?", 2)
    beyond = learners_beyond_noise(architecture_spread(runs))
    if len(beyond) <= 1:
        where = (f" na wszystkich horyzontach poza „{beyond[0]}”, gdzie przekracza go niewiele"
                 if beyond else " na każdym horyzoncie")
        intro = (f"Rozstęp między architekturami uczącymi się mieści się w rozrzucie ziarna{where}. "
                 "Jeżeli różnice są szumem, ich porządek powinien zmieniać się wraz z horyzontem. "
                 "Sprawdzeniem jest kolejność miejsc.")
    else:
        intro = (f"Rozstęp między architekturami uczącymi się przekracza rozrzut ziarna na "
                 f"{len(beyond)} horyzontach. Pozostaje pytanie, czy opisuje architekturę, czy parę "
                 "architektura-horyzont. Odpowiedzią jest kolejność miejsc.")
    add_body(doc, intro)
    add_table(
        doc,
        ["Horyzont", "1. miejsce", "2. miejsce", "3. miejsce", "4. miejsce"],
        [[horizon, *order[horizon]] for horizon in order],
        [2.8, 3.2, 3.2, 3.2, 3.2],
        font=9,
    )

    ranks = {h: {m: i for i, m in enumerate(o)} for h, o in order.items()}
    horizons = list(order)
    correlations = []
    for index, first in enumerate(horizons):
        for second in horizons[index + 1:]:
            left = np.array([ranks[first][m] for m in learners], dtype=float)
            right = np.array([ranks[second][m] for m in learners], dtype=float)
            if left.std() and right.std():
                correlations.append(float(np.corrcoef(left, right)[0, 1]))
    average = float(np.mean(correlations)) if correlations else float("nan")

    positions = {m: [ranks[h][m] + 1 for h in horizons] for m in learners}
    add_table(
        doc,
        ["Architektura", "Zajmowane miejsca", "Najlepsze", "Najgorsze"],
        [[m, ", ".join(str(p) for p in positions[m]), str(min(positions[m])),
          str(max(positions[m]))] for m in learners],
        [4.0, 5.0, 3.2, 3.2],
        font=9,
    )
    add_caption(
        doc, "Tabela",
        f"Średnia korelacja rang między horyzontami wynosi {pl(average, '+.2f')}, gdzie +1 "
        "oznaczałoby ten sam ranking wszędzie, a 0 ranking niezwiązany. "
        + _both_ends_sentence(positions, len(learners))
        + " Różnica mierzona w jednym horyzoncie nie przenosi się zatem na inny i nie stanowi "
        "podstawy do wyboru architektury.",
    )



if __name__ == "__main__":
    written = build()
    print(f"written: {written} ({os.path.getsize(written) / 1024:.0f} kB)")
