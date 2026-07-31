#!/usr/bin/env python3
"""Fail if any §N.M or "Section N.M" cross-reference names a section that does
not exist. Chapter renumbering is common during pedagogical revision, and a
dangling reference is invisible in the rendered book."""
from __future__ import annotations

import re
import sys
from pathlib import Path

CHAPTERS = Path(__file__).resolve().parent.parent / "book" / "chapters"
HEADING = re.compile(r"^\s*#{2,3} (\d+\.\d+) ", re.M)
REFERENCE = re.compile(r"(?:§ ?|Section )(\d+\.\d+)")


def main() -> int:
    sources = sorted(CHAPTERS.glob("chapter_*.py"))
    defined: set[str] = set()
    for path in sources:
        defined |= set(HEADING.findall(path.read_text()))

    dangling: list[tuple[str, str]] = []
    for path in sources:
        for match in REFERENCE.finditer(path.read_text()):
            if match.group(1) not in defined:
                dangling.append((path.name, match.group(1)))

    if dangling:
        for name, section in dangling:
            print(f"dangling reference: {name} -> section {section}")
        return 1

    print(f"all cross-references resolve ({len(defined)} sections defined)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
