"""Pull the reference chapters out of the code itself.

Every catalogue in the documentation - features, CLI flags, HTTP routes,
configuration fields, the test inventory - is read from the running system rather
than transcribed. A transcribed reference is wrong the first time someone adds a
flag and nobody notices; a generated one cannot be.

The functions here return plain data. `build_documentation.py` decides how it
looks on the page.
"""

from __future__ import annotations

import ast
import functools
import inspect
import json
import pathlib
import re
from dataclasses import dataclass, field

# ── which runs are the study ─────────────────────────────────────────────────

# Ordered roughly by market capitalisation, so the tables read the same way.
STUDY_SYMBOLS = (
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT",
    "LINKUSDT", "LTCUSDT",
)
STUDY_MODELS = ("baseline", "ml", "xgboost", "gru", "tft")
STUDY_SEEDS = (7, 42, 123)
STUDY_PROTOCOL = {"interval": "1d", "start": "2020-01-01", "end": "2026-06-30",
                  "deterministic": True}


@functools.cache
def study_runs(report_root: str = "data/reports") -> frozenset[tuple[str, str]]:
    """(symbol, run slug) of every run that belongs to the study.

    Selected by what each run says about itself in run_summary.json - interval,
    period, deterministic mode, seed - rather than by which folders exist. A run
    trained from the dashboard while testing it, on another period or seed, is
    then never counted, however long it stays in the lake.
    """
    selected = set()
    for summary in pathlib.Path(report_root, "predictions").glob("*/*/run_summary.json"):
        symbol, slug = summary.parent.parent.name, summary.parent.name
        model = slug.split("_")[0]
        if symbol not in STUDY_SYMBOLS or model not in STUDY_MODELS:
            continue
        try:
            record = json.loads(summary.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if any(record.get(key) != value for key, value in STUDY_PROTOCOL.items()):
            continue
        if record.get("seed") not in STUDY_SEEDS:
            continue
        selected.add((symbol, slug))
    return frozenset(selected)


def is_study_run(symbol: str, slug: str) -> bool:
    return (symbol, slug) in study_runs()


# ── features ─────────────────────────────────────────────────────────────────

@dataclass
class Feature:
    name: str
    group: str
    description: str


# What each training column means, keyed by its exact name and checked against
# the formulas in `features/build.py`. Written here rather than parsed out of
# comments because a feature's *meaning* is domain knowledge, not an
# implementation detail - but the list of names is still read from the code, so
# a column that exists without an entry shows up as such.
FEATURE_NOTES: dict[str, tuple[str, str]] = {
    "log_return_1b": ("Zwroty", "Logarytmiczny zwrot z ostatniej świecy. Zwroty logarytmiczne "
                                "sumują się w czasie i są symetryczne względem wzrostów i "
                                "spadków."),
    "log_return_3b": ("Zwroty", "Logarytmiczny zwrot z trzech ostatnich świec."),
    "log_return_7b": ("Zwroty", "Logarytmiczny zwrot z siedmiu ostatnich świec."),
    "log_return_14b": ("Zwroty", "Logarytmiczny zwrot z czternastu ostatnich świec."),
    "rolling_vol_7": ("Zmienność", "Odchylenie standardowe zwrotów logarytmicznych z 7 "
                                   "ostatnich świec."),
    "rolling_vol_14": ("Zmienność", "To samo z 14 ostatnich świec."),
    "rolling_vol_30": ("Zmienność", "To samo z 30 ostatnich świec."),
    "realized_vol_1d": ("Zmienność", "Suma kwadratów zwrotów jednominutowych w ciągu dnia - "
                                     "zmienność zrealizowana. Liczona tylko dla interwału "
                                     "dziennego i tylko gdy w jeziorze są dane jednominutowe."),
    "atr_14_rel": ("Zmienność", "Średni rzeczywisty zakres (wykładnicza średnia z 14 świec) "
                                "jako ułamek zamknięcia."),
    "bb_mid_rel": ("Zmienność", "Odległość środka wstęgi Bollingera (średniej z 20 świec) od "
                                "zamknięcia, jako ułamek ceny."),
    "bb_upper_rel": ("Zmienność", "Odległość górnej wstęgi Bollingera (środek plus dwa "
                                  "odchylenia standardowe) od zamknięcia, jako ułamek ceny."),
    "bb_lower_rel": ("Zmienność", "Odległość dolnej wstęgi Bollingera (środek minus dwa "
                                  "odchylenia standardowe) od zamknięcia, jako ułamek ceny."),
    "rolling_drawdown_30": ("Ryzyko", "Spadek zamknięcia poniżej najwyższego zamknięcia z "
                                      "30 ostatnich świec, jako ułamek tego maksimum."),
    "rolling_drawdown_90": ("Ryzyko", "To samo dla 90 ostatnich świec."),
    "max_intraday_drawdown": ("Ryzyko", "Największy spadek ceny poniżej dotychczasowego "
                                        "maksimum dnia, liczony z danych jednominutowych."),
    "jump_count": ("Ryzyko", "Liczba minut w ciągu dnia, w których cena zmieniła się o "
                             "więcej niż 2% - miara nieciągłości."),
    "candle_range": ("Kształt świecy", "Rozpiętość świecy (maksimum minus minimum) jako ułamek "
                                       "zamknięcia."),
    "open_rel": ("Kształt świecy", "Otwarcie świecy względem jej zamknięcia, jako ułamek ceny. "
                                   "Ta i dwie następne cechy zachowują kierunek, który "
                                   "rozpiętość świecy gubi."),
    "high_rel": ("Kształt świecy", "Maksimum świecy względem jej zamknięcia, jako ułamek ceny."),
    "low_rel": ("Kształt świecy", "Minimum świecy względem jej zamknięcia, jako ułamek ceny."),
    "ma_10_rel": ("Trend", "Odległość średniej kroczącej z 10 świec od zamknięcia, jako ułamek "
                           "ceny."),
    "ma_30_rel": ("Trend", "To samo dla średniej z 30 świec."),
    "ma_diff_rel": ("Trend", "Różnica średnich z 10 i 30 świec jako ułamek ceny - jej znak "
                             "wskazuje kierunek trendu."),
    "ma_slope_rel": ("Trend", "Zmiana średniej z 10 świec względem poprzedniej świecy, jako "
                              "ułamek ceny - nachylenie trendu."),
    "rsi_14": ("Oscylatory", "Relative Strength Index z 14 świec: stosunek średniego wzrostu do "
                             "średniego spadku, przeskalowany do 0-100."),
    "macd_hist_rel": ("Oscylatory", "Histogram MACD - różnica średnich wykładniczych z 12 i 26 "
                                    "świec pomniejszona o jej 9-świecową średnią sygnałową - "
                                    "jako ułamek ceny."),
    "taker_buy_ratio": ("Przepływ zleceń", "Udział wolumenu inicjowanego przez kupujących w "
                                           "wolumenie świecy; 0,5 oznacza równowagę."),
    "taker_buy_ratio_14": ("Przepływ zleceń", "Średnia tego udziału z 14 świec."),
    "flow_imbalance": ("Przepływ zleceń", "Udział kupujących pomniejszony o 0,5 - nierównowaga "
                                          "przepływu ze znakiem."),
    "trade_count_rel_30": ("Przepływ zleceń", "Liczba transakcji w świecy jako wielokrotność "
                                              "jej średniej z 30 świec."),
    "avg_trade_size_rel_30": ("Przepływ zleceń", "Średnia wielkość transakcji jako "
                                                 "wielokrotność jej średniej z 30 świec: ten "
                                                 "sam wolumen z kilku dużych transakcji niesie "
                                                 "inną informację niż z wielu małych."),
    "volume_rel_30": ("Wolumen", "Wolumen świecy jako wielokrotność jego średniej z 30 świec."),
    "obv_change_rel": ("Wolumen", "Zmiana On-Balance Volume (wolumenu skumulowanego ze znakiem "
                                  "zwrotu) odniesiona do średniego wolumenu z 30 świec. Sam "
                                  "poziom OBV jest dowolną sumą bieżącą, jego zmiana nie."),
}


def feature_catalogue() -> list[Feature]:
    """Every training feature, grouped and described.

    The names come from `domain.TRAINING_FEATURE_COLUMNS`, so the catalogue can
    never drift from what the models actually read; a name without a note lands
    in its own group with no description.
    """
    from prosper.domain import TRAINING_FEATURE_COLUMNS

    return [
        Feature(name, *FEATURE_NOTES.get(name, ("Bez opisu", "")))
        for name in TRAINING_FEATURE_COLUMNS
    ]


# ── command line ─────────────────────────────────────────────────────────────

@dataclass
class Option:
    flags: str
    kind: str
    required: bool
    default: str
    help: str


@dataclass
class Command:
    path: str
    summary: str
    options: list[Option] = field(default_factory=list)


def _describe_default(value: object) -> str:
    if value is None or value is ...:
        return "-"
    if isinstance(value, bool):
        return "tak" if value else "nie"
    if isinstance(value, list | tuple):
        return ", ".join(str(item) for item in value)
    return str(value)


def _options_of(callback) -> list[Option]:
    import click
    import typer
    from typer.main import get_click_param

    out: list[Option] = []
    for parameter in inspect.signature(callback).parameters.values():
        default = parameter.default
        if not isinstance(default, typer.models.OptionInfo | typer.models.ArgumentInfo):
            continue
        try:
            click_param, _ = get_click_param(
                typer.models.ParamMeta(
                    name=parameter.name, default=default, annotation=parameter.annotation
                )
            )
        except Exception:
            continue
        flags = ", ".join(click_param.opts) or parameter.name
        kind = getattr(click_param.type, "name", "text")
        if isinstance(click_param, click.Option) and click_param.is_flag:
            kind = "flaga"
        out.append(
            Option(
                flags=flags,
                kind=kind,
                required=bool(getattr(click_param, "required", False)),
                default=_describe_default(default.default),
                help=(default.help or "").strip() or "-",
            )
        )
    return out


def cli_reference() -> list[Command]:
    """Every command the CLI exposes, with its flags, read from the Typer app."""
    from prosper.cli import app

    commands: list[Command] = []

    def collect(typer_app, prefix: str) -> None:
        for registered in typer_app.registered_commands:
            name = registered.name or registered.callback.__name__.replace("_", "-")
            summary = (inspect.getdoc(registered.callback) or "").strip().split("\n")[0] or "-"
            commands.append(
                Command(
                    path=f"{prefix} {name}".strip(),
                    summary=summary,
                    options=_options_of(registered.callback),
                )
            )
        for group in typer_app.registered_groups:
            collect(group.typer_instance, f"{prefix} {group.name}".strip())

    collect(app, "prosper")
    return sorted(commands, key=lambda c: c.path)


# ── HTTP ─────────────────────────────────────────────────────────────────────

@dataclass
class Route:
    methods: str
    path: str
    summary: str


def api_reference() -> list[Route]:
    from prosper.api.server import app

    out: list[Route] = []
    for route in app.routes:
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", "")
        if not methods or not path.startswith("/api"):
            continue
        endpoint = getattr(route, "endpoint", None)
        summary = (inspect.getdoc(endpoint) or "").strip().split("\n")[0] if endpoint else ""
        out.append(
            Route(
                methods=", ".join(sorted(methods - {"HEAD", "OPTIONS"})),
                path=path,
                summary=summary or "-",
            )
        )
    return sorted(out, key=lambda r: r.path)


# ── configuration ────────────────────────────────────────────────────────────

@dataclass
class SettingField:
    name: str
    kind: str
    default: str
    description: str


def settings_reference() -> list[SettingField]:
    from prosper.config import Settings

    out: list[SettingField] = []
    for name, info in Settings.model_fields.items():
        annotation = getattr(info.annotation, "__name__", str(info.annotation))
        # A field with a default factory has no `default` of its own; the factory
        # takes no arguments here, so calling it shows the value a user gets.
        default = info.get_default(call_default_factory=True)
        out.append(
            SettingField(
                name=name,
                kind=str(annotation).replace("typing.", ""),
                default=_describe_default(default),
                description=(info.description or "-"),
            )
        )
    return sorted(out, key=lambda f: f.name)


# ── tests ────────────────────────────────────────────────────────────────────

@dataclass
class TestCase:
    name: str
    # Empty when the test has no docstring: the suite names its tests as
    # sentences, so the name alone then states the guarantee.
    intent: str


@dataclass
class TestModule:
    path: str
    purpose: str
    cases: list[TestCase] = field(default_factory=list)


def _first_sentence(text: str | None, limit: int = 400) -> str:
    if not text:
        return "-"
    collapsed = re.sub(r"\s+", " ", text.strip())
    match = re.match(r"(.+?[.!?])(\s|$)", collapsed)
    sentence = match.group(1) if match else collapsed
    return sentence if len(sentence) <= limit else sentence[: limit - 1] + "…"


def _collapse(text: str | None) -> str:
    """Whitespace-normalised docstring, kept whole."""
    if not text:
        return "-"
    return re.sub(r"\s+", " ", text.strip())


def test_inventory(root: str = "tests") -> list[TestModule]:
    """Every test in the suite, with what it pins.

    This is the system's specification in the only form that cannot go stale:
    each entry corresponds to an assertion that runs. Modules are read with `ast`
    rather than imported, so collecting the inventory never executes test code.
    """
    modules: list[TestModule] = []
    for path in sorted(pathlib.Path(root).glob("test_*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        cases = [
            TestCase(
                name=node.name,
                # The whole rationale, not its opening line. A test docstring in
                # this suite explains which defect the assertion prevents, and
                # that explanation is the part worth documenting - the first
                # sentence usually only states what is asserted.
                intent=_collapse(ast.get_docstring(node)) if ast.get_docstring(node) else "",
            )
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            and node.name.startswith("test_")
        ]
        if not cases:
            continue
        modules.append(
            TestModule(
                path=path.name,
                purpose=_first_sentence(ast.get_docstring(tree), limit=600),
                cases=sorted(cases, key=lambda c: c.name),
            )
        )
    return modules


# ── dataset ──────────────────────────────────────────────────────────────────

@dataclass
class SymbolStats:
    symbol: str
    first: str
    last: str
    bars: int
    years: float


def dataset_summary(interval: str = "1d") -> list[SymbolStats]:
    """Extent of every symbol in the lake, measured rather than declared."""
    import polars as pl

    from prosper.config import get_settings

    settings = get_settings()
    base = settings.processed_binance_spot_klines_dir / interval
    out: list[SymbolStats] = []
    if not base.exists():
        return out
    for symbol_dir in sorted(base.glob("symbol=*")):
        files = [p for p in symbol_dir.glob("**/*.parquet") if p.stat().st_size]
        if not files:
            continue
        frame = pl.concat([pl.scan_parquet(str(p)).select("open_time") for p in files]).collect()
        if frame.is_empty():
            continue
        first, last = frame["open_time"].min(), frame["open_time"].max()
        span = (last - first).days
        out.append(
            SymbolStats(
                symbol=symbol_dir.name.split("=", 1)[1],
                first=str(first.date()),
                last=str(last.date()),
                bars=len(frame),
                years=round(span / 365.25, 1),
            )
        )
    return out
