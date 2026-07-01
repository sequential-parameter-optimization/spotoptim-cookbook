# spotoptim-cookbook

[![CI Pytest](https://github.com/sequential-parameter-optimization/spotoptim-cookbook/actions/workflows/ci-pytest.yml/badge.svg)](https://github.com/sequential-parameter-optimization/spotoptim-cookbook/actions/workflows/ci-pytest.yml)
[![GitHub Pages](https://img.shields.io/website?url=https%3A%2F%2Fsequential-parameter-optimization.github.io%2Fspotoptim-cookbook%2F&label=GitHub%20Pages)](https://sequential-parameter-optimization.github.io/spotoptim-cookbook/)

Optimization cookbook.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the complete contributor workflow (uv setup, local Quarto render, testing, and PR process).


## Upgrading to the Recent spotoptim Version

```bash
uv lock --upgrade-package spotoptim   # relocks spotoptim to the recent ver, e.g. from 2.1.0 to 2.1.2
uv sync                               # installs it into .venv
```


## Workflow

* Work on `develop` and open pull requests to `main` for publication.
* CI (`ci-pytest.yml`) runs only `pytest` on pushes to `develop`, pushes to `main`, and pull requests. A guard step (`scripts/check_no_quarto_in_ci.py`) fails the build if any workflow ever tries to render or install Quarto, so rendering stays local-only.
* Quarto rendering is performed locally on `develop` with `./scripts/render.sh`, which renders the book and writes the render stamp `docs/.render-stamp`.
* A push is allowed only when the committed `docs/` matches a successful local render. A pre-push hook (`render-stamp-check`) verifies the stamp and blocks the push if `docs/` is stale. Install it once per clone with `uv run pre-commit install --hook-type pre-push`.
* GitHub Pages deployment (`deploy-pages.yml`) runs only after successful CI on `main` and publishes the committed `docs/` folder. GitHub never renders the book; it only serves the committed output.
* GitHub Release publishing (`publish-release.yml`) runs after successful Pages deployment on `main` (or manually via workflow dispatch).
* No personal access token is required for this setup; the workflow uses GitHub’s built-in `GITHUB_TOKEN`.

Release tags use the `pages-<sha12>` format to keep each release directly traceable to the exact commit that was deployed to GitHub Pages.

### Publish flow (contributors)

1. Update notebooks/qmd files on `develop`.
2. Render locally: `./scripts/render.sh` (renders the book and updates `docs/.render-stamp`).
3. Commit both source changes and the updated `docs/` output, including `docs/.render-stamp`.
4. Open and merge a pull request into `main`.
5. GitHub Actions runs `ci-pytest.yml` and then `deploy-pages.yml` publishes Pages from `docs/`.
6. After Pages deployment succeeds, `publish-release.yml` creates or updates a GitHub Release for that published commit.

## Local development (uv)

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- [Quarto CLI](https://quarto.org/docs/get-started/)

### Setup

```bash
uv sync --group dev
uv run pre-commit install --hook-type pre-push   # one-time: install the render gate
```

The second command installs the pre-push hook that enforces the render gate. It is required once per clone; without it a stale `docs/` could be pushed.

### Run tests

```bash
uv run pytest
```

### Render and preview the book

```bash
./scripts/render.sh   # render the whole book and update docs/.render-stamp
quarto preview        # live preview while editing (no stamp)
```

The rendered site is written to `docs/` as configured in `_quarto.yml`. Always render with `./scripts/render.sh` rather than a bare `quarto render`, so the render stamp stays in sync and the pre-push gate lets the push through.
