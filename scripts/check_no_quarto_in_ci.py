#!/usr/bin/env python3
"""CI guard: fail if any GitHub Actions workflow renders or installs Quarto.

Rendering the cookbook is local-only (see CONTRIBUTING.md). GitHub Actions
must run pytest and serve the committed ``docs/`` to Pages, never execute
Quarto. This guard runs in CI and fails the build if a workflow reintroduces
a render step or a Quarto installation.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

WORKFLOWS = Path(__file__).resolve().parent.parent / ".github" / "workflows"

FORBIDDEN = re.compile(
    r"quarto\s+(render|preview|publish)"  # invoking the Quarto CLI
    r"|quarto-actions"  # the quarto-actions/* marketplace actions
    r"|setup-quarto"
    r"|install-quarto"
    r"|(pip|uv|brew|apt|apt-get|conda)\s+(install|add)\b.*\bquarto\b",
    re.IGNORECASE,
)


def main() -> int:
    if not WORKFLOWS.is_dir():
        print(f"no workflows directory at {WORKFLOWS}; nothing to check.")
        return 0
    offenders: list[str] = []
    for path in sorted(WORKFLOWS.glob("*.y*ml")):
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if FORBIDDEN.search(line):
                offenders.append(f"{path.name}:{lineno}: {line.strip()}")
    if offenders:
        print(
            "Quarto rendering must stay local-only; "
            "CI may not render or install Quarto.",
            file=sys.stderr,
        )
        print("Offending workflow lines:", file=sys.stderr)
        for offender in offenders:
            print("  " + offender, file=sys.stderr)
        return 1
    print("OK: no workflow renders or installs Quarto.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
