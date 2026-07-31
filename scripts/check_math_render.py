#!/usr/bin/env python3
"""Fail if any chapter contains math that KaTeX could not render.

A KaTeX parse failure is invisible to the test suite: the notebook still
executes, every test still passes, and the only symptom is that the rendered
page shows raw LaTeX source in red where an equation should be. It is therefore
found by eye, or not at all.

Two checks run here.

1. Rendered check -- scan `content/ChapterNN.html` for KaTeX's own error class.
   This catches every kind of failure, not just the known ones, but requires the
   book to have been built.
2. Source lint -- flag `\\texttt{...}` containing an unescaped underscore. KaTeX
   reads `_` as a subscript operator even inside `\\texttt`, so
   `\\texttt{species_code}` fails while `\\texttt{species\\_code}` renders. This
   caught eight expressions in Chapter 8 and is easy to reintroduce, because the
   same string is legal in a code cell.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
CONTENT = REPOSITORY_ROOT / "content"
CHAPTERS = REPOSITORY_ROOT / "book" / "chapters"

TEXTTT = re.compile(r"\\texttt\{([^}]*)\}")
UNESCAPED_UNDERSCORE = re.compile(r"(?<!\\)_")


def main() -> int:
    failures: list[str] = []

    rendered = sorted(CONTENT.glob("Chapter*.html"))
    if not rendered:
        print("no rendered chapters found; run scripts/build_book.py first",
              file=sys.stderr)
        return 1
    for path in rendered:
        count = path.read_text(errors="replace").count("katex-error")
        if count:
            failures.append(f"{path.name}: {count} KaTeX render error(s)")

    for path in sorted(CHAPTERS.glob("chapter_*.py")):
        source = path.read_text()
        for match in TEXTTT.finditer(source):
            if UNESCAPED_UNDERSCORE.search(match.group(1)):
                failures.append(
                    f"{path.name}: unescaped underscore in {match.group(0)}"
                )

    if failures:
        for line in failures:
            print(line, file=sys.stderr)
        return 1

    print(f"all math renders ({len(rendered)} chapters checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
