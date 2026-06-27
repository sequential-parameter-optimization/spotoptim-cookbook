#!/usr/bin/env bash
# Local render for the spotoptim-cookbook.
#
# Renders the whole book with Quarto and, only on success, writes
# docs/.render-stamp so the pre-push hook can prove the committed docs/ matches
# these sources. Use this instead of a bare `quarto render`.
set -euo pipefail

cd "$(dirname "$0")/.."

if ! grep -q "pre-commit" .git/hooks/pre-push 2>/dev/null; then
  echo "WARNING: the pre-push render gate is not installed in this clone." >&2
  echo "         Run once: uv run pre-commit install --hook-type pre-push" >&2
fi

echo "Rendering the book with Quarto (this takes a while) ..."
uv run quarto render

python3 scripts/render_stamp.py --write
echo "Done. Commit the updated docs/ (including docs/.render-stamp) before pushing."
