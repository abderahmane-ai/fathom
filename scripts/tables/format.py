"""Markdown table formatting helpers.

All table scripts use ``markdown_table(rows, headers, caption=...)`` to
produce a single markdown table string.  ``bold_winner`` finds the row
with the best (or worst) value in a column and wraps it in **...** so the
winner pops out.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from tabulate import tabulate


def markdown_table(
    rows: Sequence[Mapping[str, Any]],
    headers: Sequence[str] | Mapping[str, str],
    *,
    caption: str | None = None,
    floatfmt: str | None = None,
    tablefmt: str = "github",
) -> str:
    """Format rows as a markdown table.

    Args:
        rows: List of dicts, one per row.
        headers: Either a list of keys (in order) or a dict of {key: display_name}.
        caption: Optional caption prepended as ``**Caption**\n``.
        floatfmt: Passed to ``tabulate`` (e.g. ``".4f"`` for 4 decimal places).
        tablefmt: Markdown dialect.  Default ``"github"`` is the GitHub-flavored variant.

    Returns:
        The formatted markdown table as a string.
    """
    if isinstance(headers, dict):
        # Use the display names, but key rows by the original keys.
        header_display = list(headers.values())
        header_keys = list(headers.keys())
    else:
        header_display = list(headers)
        header_keys = list(headers)

    body = []
    for row in rows:
        body.append([row.get(k, "-") for k in header_keys])
    tabulate_kwargs: dict[str, Any] = {"headers": header_display, "tablefmt": tablefmt}
    if floatfmt is not None:
        tabulate_kwargs["floatfmt"] = floatfmt
    table = tabulate(body, **tabulate_kwargs)
    if caption:
        return f"**{caption}**\n\n{table}\n"
    return table + "\n"


def bold_winner(
    rows: list[dict[str, Any]],
    column: str,
    *,
    lower_is_better: bool = True,
) -> list[dict[str, Any]]:
    """Return a copy of ``rows`` with the winning value in ``column`` wrapped in ``**...**``.

    If multiple rows tie for the best value, all of them are bolded.  Rows
    with None / NaN values are skipped.

    Args:
        rows: The rows (list of dicts).  Not modified in place; returns a new list.
        column: The key to look at.
        lower_is_better: If True, the minimum value wins.  If False, the maximum wins.

    Returns:
        A new list of dicts with the same keys plus a ``_bolded`` boolean per row.
    """
    valid = [r for r in rows if r.get(column) is not None]
    if not valid:
        return [dict(r, _bolded=False) for r in rows]
    values = [r[column] for r in valid]
    target = min(values) if lower_is_better else max(values)
    return [
        dict(r, _bolded=(r.get(column) is not None and r[column] == target))
        for r in rows
    ]


def apply_bold(rows: list[dict[str, Any]], value_columns: Iterable[str]) -> list[dict[str, Any]]:
    """Format a list of rows so that columns whose ``_bolded`` flag is True get ``**...**``.

    Args:
        rows: Rows with ``_bolded`` boolean set (from ``bold_winner``).
        value_columns: Column names that should be wrapped in ``**...**`` if bolded.

    Returns:
        New list of dicts with values possibly wrapped in markdown bold.
    """
    new_rows = []
    for row in rows:
        new_row = dict(row)
        if new_row.pop("_bolded", False):
            for col in value_columns:
                val = new_row.get(col)
                if val is not None:
                    new_row[col] = f"**{val}**"
        new_rows.append(new_row)
    return new_rows


def write_markdown_table(
    out_path: str | "Path",
    title: str,
    rows: list[dict[str, Any]],
    headers: list[str] | dict[str, str],
    *,
    caption: str | None = None,
    floatfmt: str | None = None,
    preamble: str | None = None,
) -> None:
    """Write a markdown table to a file.  Prepends a title and optional preamble.

    Args:
        out_path: Output file.
        title: H1 title.
        rows: Rows of data.
        headers: Column names (list or dict for display labels).
        caption: Optional table caption.
        floatfmt: Passed to tabulate.
        preamble: Optional text between the title and the table.
    """
    from pathlib import Path

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = [f"# {title}\n"]
    if preamble:
        body.append(preamble + "\n")
    body.append(markdown_table(rows, headers, caption=caption, floatfmt=floatfmt))
    path.write_text("\n".join(body), encoding="utf-8")
