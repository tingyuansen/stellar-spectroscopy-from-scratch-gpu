#!/usr/bin/env python3
"""Restore the filesystem modes that the Chapter 6 publication gates require.

Two published artifacts must be single-link mode-0600 files. Git records only
the executable bit, so a fresh clone -- or any `git checkout` that rewrites the
working tree -- leaves them at the umask default (typically 0644) and the
publication-authority tests fail with

    PublicationStateError: synthesis target is not the exact single-link
    mode-0600 artifact

The contents are unaffected; only the permission bit is lost. Run this after
cloning or after any checkout that touches `data/`.
"""
from __future__ import annotations

import stat
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

REQUIRED_MODE = 0o600
PUBLISHED_ARTIFACTS = (
    "data/fixtures/chapter06_atmosphere_one_line_inputs.npz",
    "data/golden/payne_zero/chapter06/synthesis/"
    "chapter06_synthesis_one_line_cpu_float64_work_float32_accumulation.npz",
)


def main() -> int:
    missing: list[str] = []
    changed = 0

    for relative in PUBLISHED_ARTIFACTS:
        path = REPOSITORY_ROOT / relative
        if not path.exists():
            missing.append(relative)
            continue
        current = stat.S_IMODE(path.stat().st_mode)
        if current == REQUIRED_MODE:
            print(f"ok      {relative}")
            continue
        path.chmod(REQUIRED_MODE)
        changed += 1
        print(f"chmod   {relative}  {current:04o} -> {REQUIRED_MODE:04o}")

    if missing:
        for relative in missing:
            print(f"MISSING {relative}", file=sys.stderr)
        return 1

    print(f"\n{changed} mode(s) restored; publication gates can now verify.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
