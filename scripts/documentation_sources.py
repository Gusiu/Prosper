"""Pull the reference chapters out of the code itself.

Every catalogue in the documentation — features, CLI flags, HTTP routes,
configuration fields, the test inventory — is read from the running system rather
than transcribed. A transcribed reference is wrong the first time someone adds a
flag and nobody notices; a generated one cannot be.

The functions here return plain data. `build_documentation.py` decides how it
looks on the page.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import re
from dataclasses import dataclass, field

# ── features ─────────────────────────────────────────────────────────────────

@dataclass
class Feature:
    name: str
    group: str
    description: str


# What each column means, keyed by the stem the builder uses. Written here rather
# than parsed out of comments because a feature's *meaning* is domain knowledge,
# not an implementation detail — but the list of names is still read from the
# code, so a column that exists without an entry shows up as such.
FEATURE_NOTES: dict[str, tuple[str, str]] = {
    "log_return": ("Zwroty", "Logarytmiczny zwrot z n świec wstecz. Addytywny w czasie i "
                             "symetryczny względem wzrostów i spadków."),
    "rsi": ("Oscylatory", "Relative Strength Index: stosunek średniego wzrostu do średniego "
                          "spadku w oknie, przeskalowany do 0-100."),
    "macd": ("Oscylatory", "Różnica dwóch wykładniczych średnich kroczących oraz jej odległość "
                           "od własnej średniej sygnałowej."),
    "stoch": ("Oscylatory", "Położenie zamknięcia względem zakresu maks.-min. w oknie."),
    "williams": ("Oscylatory", "Odwrócony wskaźnik położenia zamknięcia w zakresie okna."),
    "cci": ("Oscylatory", "Odchylenie ceny typowej od jej średniej, skalowane odchyleniem "
                          "bezwzględnym."),
    "roc": ("Momentum", "Tempo zmiany: procentowa zmiana ceny względem n świec wstecz."),
    "momentum": ("Momentum", "Bezwzględna różnica ceny względem n świec wstecz, wyrażona "
                             "względnie."),
    "ma": ("Trend", "Odległość ceny od średniej kroczącej, wyrażona jako ułamek tej średniej - "
                    "dzięki czemu wielkość nie zależy od poziomu ceny."),
    "ema": ("Trend", "Jak wyżej, dla średniej wykładniczej."),
    "bb": ("Zmienność", "Położenie ceny wewnątrz wstęgi Bollingera oraz szerokość wstęgi "
                        "względem jej środka."),
    "atr": ("Zmienność", "Średni rzeczywisty zakres, wyrażony jako ułamek ceny."),
    "rolling_vol": ("Zmienność", "Odchylenie standardowe zwrotów logarytmicznych w oknie."),
    "realized_vol": ("Zmienność", "Zmienność zrealizowana liczona z danych o wyższej "
                                  "częstotliwości niż interwał modelu."),
    "volume": ("Wolumen", "Wolumen odniesiony do własnej średniej kroczącej, co usuwa zależność "
                          "od skali instrumentu."),
    "obv": ("Wolumen", "On-Balance Volume: skumulowany wolumen ze znakiem zwrotu, raportowany "
                       "jako zmiana względna."),
    "taker_buy": ("Przepływ zleceń", "Udział wolumenu inicjowanego przez kupujących w wolumenie "
                                     "całkowitym - dostępny tylko dla danych zawierających "
                                     "kolumny przepływu."),
    "trade": ("Przepływ zleceń", "Charakterystyki liczby i wielkości transakcji w świecy."),
    "candle": ("Kształt świecy", "Proporcje korpusu i cieni względem pełnego zakresu świecy."),
    "gap": ("Kształt świecy", "Odległość otwarcia od poprzedniego zamknięcia."),
    "jump": ("Kształt świecy", "Liczba zwrotów przekraczających wielokrotność bieżącej "
                               "zmienności - miara nieciągłości."),
    "drawdown": ("Ryzyko", "Maksymalne obsunięcie wewnątrz okna."),
    "intraday": ("Ryzyko", "Charakterystyki wyliczone z danych o wyższej częstotliwości."),
    "day_of": ("Kalendarz", "Zmienne kalendarzowe kodujące porę tygodnia lub miesiąca."),
    "hour": ("Kalendarz", "Pora doby, istotna wyłącznie dla interwałów krótszych niż dzień."),
}


def feature_catalogue() -> list[Feature]:
    """Every training feature, grouped and described.

    The names come from `domain.TRAINING_FEATURE_COLUMNS`, so the catalogue can
    never drift from what the models actually read.
    """
    from prosper.domain import TRAINING_FEATURE_COLUMNS

    out: list[Feature] = []
    for name in TRAINING_FEATURE_COLUMNS:
        group, description = "Pozostałe", "—"
        for stem, (candidate_group, candidate_text) in FEATURE_NOTES.items():
            if name.startswith(stem):
                group, description = candidate_group, candidate_text
                break
        if name.endswith("_rel"):
            description += " Wariant względny, niezależny od poziomu ceny."
        out.append(Feature(name=name, group=group, description=description))
    return out


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
        return "—"
    if isinstance(value, bool):
        return "tak" if value else "nie"
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
                help=(default.help or "").strip() or "—",
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
            summary = (inspect.getdoc(registered.callback) or "").strip().split("\n")[0] or "—"
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
                summary=summary or "—",
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
        default = info.default
        if default is None and info.default_factory is not None:
            default = "(wyliczane)"
        out.append(
            SettingField(
                name=name,
                kind=str(annotation).replace("typing.", ""),
                default=_describe_default(default),
                description=(info.description or "—"),
            )
        )
    return sorted(out, key=lambda f: f.name)


# ── tests ────────────────────────────────────────────────────────────────────

@dataclass
class TestCase:
    name: str
    intent: str


@dataclass
class TestModule:
    path: str
    purpose: str
    cases: list[TestCase] = field(default_factory=list)


def _first_sentence(text: str | None, limit: int = 400) -> str:
    if not text:
        return "—"
    collapsed = re.sub(r"\s+", " ", text.strip())
    match = re.match(r"(.+?[.!?])(\s|$)", collapsed)
    sentence = match.group(1) if match else collapsed
    return sentence if len(sentence) <= limit else sentence[: limit - 1] + "…"


def _collapse(text: str | None) -> str:
    """Whitespace-normalised docstring, kept whole."""
    if not text:
        return "—"
    return re.sub(r"\s+", " ", text.strip())

def _humanise(name: str) -> str:
    """`test_the_window_reports_power` -> `the window reports power`.

    The suite names its tests as sentences, so the name alone states the
    guarantee even where no docstring was written.
    """
    return name.removeprefix("test_").replace("_", " ")


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
                # that explanation is the part worth documenting — the first
                # sentence usually only states what is asserted.
                intent=_collapse(ast.get_docstring(node)) if ast.get_docstring(node)
                else _humanise(node.name),
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
