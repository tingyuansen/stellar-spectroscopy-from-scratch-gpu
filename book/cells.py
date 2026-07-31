"""Small helpers for canonical, generated Jupyter notebooks."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import textwrap
from typing import Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def markdown(text: str) -> dict:
    """Create one markdown cell from readable indented source."""

    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": textwrap.dedent(text).strip() + "\n",
    }


def code(source: str, *, tags: Iterable[str] = ()) -> dict:
    """Create one unexecuted Python cell."""

    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": list(tags)} if tags else {},
        "outputs": [],
        "source": textwrap.dedent(source).strip() + "\n",
    }


def definition_code(
    relative_path: str,
    names: Iterable[str],
    *,
    maximum_lines: int = 35,
    tags: Iterable[str] = (),
) -> dict:
    """Create an executable cell from small canonical top-level definitions.

    The chapter reader sees ordinary code, not a generated Markdown source
    dump. The line limit prevents a production-sized routine from being
    inserted before it has been divided into teachable stages. Callers choose
    tags so textbook-defined teaching helpers remain distinct from exact
    Payne Zero source.
    """

    path = REPOSITORY_ROOT / relative_path
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    requested = list(names)
    definitions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    missing = [name for name in requested if name not in definitions]
    if missing:
        raise ValueError(f"{path}: definitions not found: {missing}")
    lines = source.splitlines()
    blocks = [
        "\n".join(lines[definitions[name].lineno - 1 : definitions[name].end_lineno])
        for name in requested
    ]
    source_block = "\n\n".join(blocks)
    line_count = source_block.count("\n") + 1
    if line_count > maximum_lines:
        raise ValueError(
            f"{relative_path}: requested source is {line_count} lines; "
            f"the chapter limit is {maximum_lines}"
        )
    return code(source_block, tags=tags)


def source_code(
    relative_path: str,
    names: Iterable[str],
    *,
    maximum_lines: int = 35,
) -> dict:
    """Create a source-verified exact Payne Zero definition cell."""

    return definition_code(
        relative_path,
        names,
        maximum_lines=maximum_lines,
        tags=("exact-payne-zero-source",),
    )


def notebook(cells: list[dict]) -> dict:
    """Create a deterministic notebook document."""

    identified_cells = []
    for index, cell in enumerate(cells):
        identified = dict(cell)
        source_hash = hashlib.sha1(cell["source"].encode("utf-8")).hexdigest()[:8]
        identified["id"] = f"cell-{index:03d}-{source_hash}"
        identified_cells.append(identified)
    return {
        "cells": identified_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def setup_cell() -> dict:
    """Create the shared executable setup, hidden in the rendered textbook."""

    return code(
        """
        from pathlib import Path
        import sys

        repository_root = Path.cwd()
        if repository_root.name == "content":
            repository_root = repository_root.parent
        if str(repository_root) not in sys.path:
            sys.path.insert(0, str(repository_root))
        source_root = repository_root / "src"
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))

        from book.plot_style import apply_book_plot_style

        apply_book_plot_style()
        """,
        tags=("book-setup", "hide-input"),
    )
