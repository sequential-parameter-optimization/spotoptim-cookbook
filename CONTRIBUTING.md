# Contributing to `spotoptim-cookbook`

Thank you for contributing.

This repository uses a local render plus committed output workflow:

- Work on `develop`
- Render Quarto locally with `./scripts/render.sh`
- Commit both the source and the rendered `docs/` (including `docs/.render-stamp`)
- Push only after a successful render: a pre-push hook blocks the push when `docs/` is stale
- Open a PR to `main`
- GitHub Actions runs `pytest`, deploys GitHub Pages from the committed `docs/`, and publishes a GitHub Release. It never renders the book.

## Branch and CI/CD model

- `develop`: day-to-day authoring branch
- `main`: publication branch
- CI workflow (`ci-pytest.yml`): runs `pytest` only
- Deploy workflow (`deploy-pages.yml`): runs after successful CI on `main` and publishes `docs/`
- Release workflow (`publish-release.yml`): runs after successful Pages deployment on `main` and creates/updates a GitHub Release

Quarto rendering is intentionally not done in GitHub Actions. A guard step (`scripts/check_no_quarto_in_ci.py`) runs in CI and fails the build if any workflow reintroduces a Quarto render or installation, so the local-only policy cannot regress silently.

Release tags follow the `pages-<sha12>` format so each GitHub Release maps to the exact commit deployed to Pages.

## Render gate (the fixed rule)

Rendering is local-only, and a push is allowed only when the committed `docs/` was produced by a successful render of the current sources. This is enforced mechanically, not by convention:

- `./scripts/render.sh` renders the book and, only on success, writes `docs/.render-stamp`, a hash of every render input (the tracked `.qmd` files, `_quarto.yml`, the `.bib` files, and the pinned `spotoptim` version from `uv.lock`).
- A pre-push hook (`render-stamp-check`, configured in `.pre-commit-config.yaml`) recomputes that hash and rejects the push if it does not match `docs/.render-stamp`. The check is fast and does not render.

Install the hook once per clone:

```bash
uv run pre-commit install --hook-type pre-push
```

If a push is rejected with a "docs/ is stale" message, run `./scripts/render.sh`, commit the regenerated `docs/`, and push again.

## Prerequisites

Use **uv only** for Python environment and dependency management.

Required tools:

- [uv](https://docs.astral.sh/uv/)
- [Quarto CLI](https://quarto.org/docs/get-started/)
- Git

Do **not** use `pip`, `conda`, `poetry`, or `virtualenv` commands in this project.

## Python environment setup (uv only)

From the repository root:

```bash
uv sync --group dev
```

This creates/updates `.venv` and installs all dependencies from `pyproject.toml` / `uv.lock`.

Then install the pre-push render gate once per clone:

```bash
uv run pre-commit install --hook-type pre-push
```

Optional shell activation:

```bash
source .venv/bin/activate
```

You can also run tools without activating by prefixing with `uv run`.

## Local development commands

Run tests:

```bash
uv run pytest
```

Render the Quarto project (renders the book and updates `docs/.render-stamp`):

```bash
./scripts/render.sh
```

Preview locally:

```bash
quarto preview
```

Rendered output is written to `docs/` (configured in `_quarto.yml`).

## Standard contribution workflow

1. Update your local `develop` branch.
2. Make your content/code changes.
3. Run tests: `uv run pytest`.
4. Render locally: `./scripts/render.sh`.
5. Verify updated pages.
6. Commit source changes and the rendered `docs/`, including `docs/.render-stamp`.
7. Push to `develop`. The pre-push hook blocks the push if `docs/` is stale.
8. Open a PR from `develop` to `main`.

## Recommended git command sequence

After finishing local render:

```bash
git status
git add -A
git status
git commit -m "Update content and rendered docs"
git push origin develop
```

If you want to be more selective (instead of git add -A), use:

```bash
git add docs
git add *.qmd *.ipynb _quarto.yml README.md
git commit -m "Update content and rendered site"
git push origin develop
```


If you use GitHub CLI for PR creation:

```bash
gh pr create --base main --head develop --fill
```

## Pull request expectations

Before requesting review, ensure:

- Tests pass locally (`uv run pytest`)
- Quarto render completes locally (`./scripts/render.sh`)
- `docs/` is up to date and committed when content changed, and the pre-push gate passes
- PR template checklist is completed

## Troubleshooting

Refresh dependencies:

```bash
uv sync --upgrade --group dev
```

Run a command inside the managed environment without activating:

```bash
uv run python -V
```

If your rendered site looks stale, run `quarto render` again and verify changed files under `docs/`.
