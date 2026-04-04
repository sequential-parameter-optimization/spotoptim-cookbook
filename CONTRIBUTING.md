# Contributing to `spotoptim-cookbook`

Thank you for contributing.

This repository uses a **local render + committed output** workflow:

- Work on `develop`
- Render Quarto **locally**
- Commit both source and rendered `docs/`
- Open PR to `main`
- GitHub Actions runs `pytest`, deploys GitHub Pages from committed `docs/`, and publishes a GitHub Release

## Branch and CI/CD model

- `develop`: day-to-day authoring branch
- `main`: publication branch
- CI workflow (`ci-pytest.yml`): runs `pytest` only
- Deploy workflow (`deploy-pages.yml`): runs after successful CI on `main` and publishes `docs/`
- Release workflow (`publish-release.yml`): runs after successful Pages deployment on `main` and creates/updates a GitHub Release

Quarto rendering is intentionally **not** done in GitHub Actions.

Release tags follow the `pages-<sha12>` format so each GitHub Release maps to the exact commit deployed to Pages.

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

Render the Quarto project:

```bash
quarto render
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
4. Render locally: `quarto render`.
5. Verify updated pages.
6. Commit source changes **and** rendered `docs/`.
7. Push to `develop`.
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
- Quarto render completes locally (`quarto render`)
- `docs/` is up to date and committed when content changed
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
