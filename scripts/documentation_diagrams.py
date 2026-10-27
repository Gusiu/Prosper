"""UML diagrams of the system, drawn by PlantUML from docs/diagramy/*.puml.

The sources are part of the repository; the pictures are derived files and go
beside the figures, re-rendered only when a source or the shared style is newer
than its picture. Rendering needs Java and the PlantUML jar named by the
PLANTUML_JAR environment variable. Without them the document uses the pictures
rendered before and says which ones are out of date.

A diagram whose content comes from data, such as the map of training features,
is written out as a source by the document builder and rendered the same way.
"""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
from pathlib import Path

SOURCE_DIR = Path("docs/diagramy")
OUTPUT_DIR = Path("data/_figures/diagramy")
GENERATED_DIR = OUTPUT_DIR / "generated"
STYLE = "_styl.iuml"
# The resolution the shared style renders at, to convert pixels to centimetres.
RENDER_DPI = 200


def render_all() -> dict[str, Path]:
    """Every diagram as a PNG, keyed by its source's name without the suffix."""
    sources = sorted(SOURCE_DIR.glob("*.puml"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    style = SOURCE_DIR / STYLE
    style_time = style.stat().st_mtime if style.exists() else 0.0
    stale = [source for source in sources if _stale(source, OUTPUT_DIR, style_time)]
    _render(stale, OUTPUT_DIR)
    return {
        source.stem: OUTPUT_DIR / f"{source.stem}.png"
        for source in sources
        if (OUTPUT_DIR / f"{source.stem}.png").exists()
    }


def render_source(name: str, source: str) -> Path | None:
    """Render a diagram whose PlantUML source the caller built; the PNG, if any.

    The source is rewritten only when it changed, so an unchanged diagram is not
    redrawn on every build.
    """
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    path = GENERATED_DIR / f"{name}.puml"
    if not path.exists() or path.read_text(encoding="utf-8") != source:
        path.write_text(source, encoding="utf-8")
    if _stale(path, GENERATED_DIR, 0.0):
        _render([path], GENERATED_DIR)
    picture = GENERATED_DIR / f"{name}.png"
    return picture if picture.exists() else None


def _stale(source: Path, output: Path, style_time: float) -> bool:
    picture = output / f"{source.stem}.png"
    return not picture.exists() or picture.stat().st_mtime < max(source.stat().st_mtime,
                                                                  style_time)


def _render(sources: list[Path], output: Path) -> None:
    if not sources:
        return
    jar = os.environ.get("PLANTUML_JAR")
    java = shutil.which("java")
    if not (jar and java and Path(jar).is_file()):
        print(f"[diagrams] {len(sources)} out of date and no PLANTUML_JAR or java to redraw "
              f"them: {', '.join(s.stem for s in sources)}")
        return
    result = subprocess.run(
        [java, "-Djava.awt.headless=true", "-jar", jar, "-tpng", "-charset", "UTF-8",
         "-failfast2", "-o", str(output.resolve()), *(str(s) for s in sources)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if result.returncode != 0:
        print(f"[diagrams] PlantUML failed: {(result.stdout + result.stderr).strip()}")


def size_cm(path: Path) -> tuple[float, float]:
    """Natural width and height of a rendered diagram, read from its PNG header."""
    with open(path, "rb") as handle:
        header = handle.read(24)
    width, height = struct.unpack(">II", header[16:24])
    return width / RENDER_DPI * 2.54, height / RENDER_DPI * 2.54
