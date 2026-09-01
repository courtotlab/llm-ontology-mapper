"""Generic (non-plotting, non-matplotlib) helpers shared by figure builders:
CSV/markdown table writers and a couple of tiny string/number coercions.
Pure I/O over already-in-memory data -- no filesystem scanning, no network.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def to_float(value: Any) -> float | None:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def write_csv(rows: list[dict[str, Any]], fieldnames: Sequence[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _fmt_cell(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return "" if value is None else str(value)


def write_markdown_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    path: Path,
    *,
    title: str | None = None,
) -> None:
    lines: list[str] = []
    if title:
        lines.append(f"# {title}")
        lines.append("")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(_fmt_cell(v) for v in row) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
