#!/usr/bin/env python3
"""Render-stamp tool for the local-render release gate.

A stamp is a SHA-256 hash over every input that affects the rendered book:
all tracked ``.qmd`` files, ``_quarto.yml``, every ``.bib`` file, and the
resolved ``spotoptim`` version from ``uv.lock``. ``scripts/render.sh`` writes
the stamp only after a successful ``quarto render``; the pre-push hook checks
it. A push is allowed only when ``docs/.render-stamp`` matches the current
sources, which proves the committed ``docs/`` came from a successful render of
exactly those sources.

Usage:
    python3 scripts/render_stamp.py --write    # after a successful render
    python3 scripts/render_stamp.py --check     # pre-push gate
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAMP = ROOT / "docs" / ".render-stamp"


def _tracked_inputs() -> list[Path]:
    """Return the tracked render inputs, sorted by repository path."""
    result = subprocess.run(
        ["git", "ls-files", "-z", "*.qmd", "_quarto.yml", "*.bib"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    names = [n for n in result.stdout.split("\0") if n]
    return [ROOT / n for n in sorted(names)]


def _spotoptim_version() -> str:
    """Return the spotoptim version pinned in uv.lock, or a sentinel."""
    lock = ROOT / "uv.lock"
    if not lock.exists():
        return "no-uv-lock"
    data = tomllib.loads(lock.read_text())
    for pkg in data.get("package", []):
        if pkg.get("name") == "spotoptim":
            return str(pkg.get("version", "unknown"))
    return "spotoptim-not-pinned"


def compute_hash() -> str:
    """Compute a deterministic hash over all render inputs."""
    digest = hashlib.sha256()
    digest.update(b"spotoptim-version:" + _spotoptim_version().encode() + b"\n")
    for path in _tracked_inputs():
        rel = path.relative_to(ROOT).as_posix()
        digest.update(b"path:" + rel.encode() + b"\n")
        digest.update(path.read_bytes())
        digest.update(b"\n")
    return digest.hexdigest()


def cmd_write() -> int:
    digest = compute_hash()
    STAMP.parent.mkdir(parents=True, exist_ok=True)
    STAMP.write_text(digest + "\n")
    print(f"render stamp written: {digest[:12]} ({STAMP.relative_to(ROOT)})")
    return 0


def cmd_check() -> int:
    if not STAMP.exists():
        print(
            "render stamp missing (docs/.render-stamp). "
            "Run ./scripts/render.sh and commit docs/ before pushing.",
            file=sys.stderr,
        )
        return 1
    expected = STAMP.read_text().strip()
    actual = compute_hash()
    if expected != actual:
        print(
            "docs/ is stale relative to sources "
            "(.qmd, _quarto.yml, .bib, or the spotoptim version changed). "
            "Run ./scripts/render.sh and commit docs/ before pushing.",
            file=sys.stderr,
        )
        return 1
    print("render stamp OK: docs/ matches the current sources.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--write",
        action="store_true",
        help="write docs/.render-stamp from the current sources",
    )
    group.add_argument(
        "--check",
        action="store_true",
        help="verify docs/.render-stamp matches the current sources",
    )
    args = parser.parse_args()
    return cmd_write() if args.write else cmd_check()


if __name__ == "__main__":
    raise SystemExit(main())
