"""Deduplication must not be allowed to reorder rows that are read by position.

Features are windows over consecutive rows, and evaluation reads an outcome k
rows ahead. Every loader sorts and then drops duplicate timestamps, and polars'
`unique` keeps the sorted order only when asked to. The stored files were all in
order, so it had been holding by luck rather than by contract.
"""

from __future__ import annotations

import pathlib
import re

SOURCE = pathlib.Path(__file__).resolve().parent.parent / "src" / "prosper"
CALL = re.compile(r"\.unique\(\s*subset=\[\"open_time\"\][^)]*\)")


def test_every_deduplication_keeps_order_or_sorts_again() -> None:
    offenders = []
    for path in SOURCE.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for match in CALL.finditer(text):
            after = text[match.end():match.end() + 40].lstrip()
            if "maintain_order=True" not in match.group(0) and not after.startswith(".sort("):
                line = text[: match.start()].count("\n") + 1
                offenders.append(f"{path.relative_to(SOURCE)}:{line}")
    assert not offenders, "order not guaranteed after unique:\n  " + "\n  ".join(offenders)
