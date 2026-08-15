"""Output paths and markdown helpers shared by the example scripts.

Each example script regenerates its own docs page and plot images, so the
documentation stays in sync with the code by rerunning the script.
"""

import pandas as pd

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
IMAGES = REPO / "docs" / "images"
PAGES = REPO / "docs" / "examples"


def image_path(name: str) -> str:
    IMAGES.mkdir(parents=True, exist_ok=True)
    return str(IMAGES / name)


def save_page(slug: str, markdown: str) -> Path:
    PAGES.mkdir(parents=True, exist_ok=True)
    page = PAGES / f"{slug}.md"
    page.write_text(markdown, encoding="utf-8", newline="\n")
    return page


def md_table(table: pd.DataFrame, n_rows: int = 10) -> str:
    """Plain markdown table, no tabulate dependency."""
    rows = table.head(n_rows)
    lines = [
        "| " + " | ".join(str(column) for column in rows.columns) + " |",
        "|" + "|".join("---" for _ in rows.columns) + "|",
    ]
    for _, row in rows.iterrows():
        cells = [f"{v:.4g}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
