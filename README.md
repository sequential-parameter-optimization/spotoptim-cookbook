# spotoptim-cookbook

[![CI Pytest](https://github.com/sequential-parameter-optimization/spotoptim-cookbook/actions/workflows/ci-pytest.yml/badge.svg)](https://github.com/sequential-parameter-optimization/spotoptim-cookbook/actions/workflows/ci-pytest.yml)
[![GitHub Pages](https://img.shields.io/website?url=https%3A%2F%2Fsequential-parameter-optimization.github.io%2Fspotoptim-cookbook%2F&label=GitHub%20Pages)](https://sequential-parameter-optimization.github.io/spotoptim-cookbook/)

Optimization cookbook.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the complete contributor workflow (uv setup, local Quarto render, testing, and PR process).

## Workflow

- Work on `develop` and open pull requests to `main` for publication.
- CI (`ci-pytest.yml`) runs only `pytest` on pushes to `develop`, pushes to `main`, and pull requests.
- Quarto rendering is performed locally on `develop`.
- GitHub Pages deployment (`deploy-pages.yml`) runs only after successful CI on `main` and publishes the committed `docs/` folder.
- No personal access token is required for this setup; the workflow uses GitHub’s built-in `GITHUB_TOKEN`.

### Publish flow (contributors)

1. Update notebooks/qmd files on `develop`.
2. Render locally: `quarto render`.
3. Commit both source changes and updated `docs/` output.
4. Open and merge a pull request into `main`.
5. GitHub Actions runs `ci-pytest.yml` and then `deploy-pages.yml` publishes Pages from `docs/`.

## Local development (uv)

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- [Quarto CLI](https://quarto.org/docs/get-started/)

### Setup

```bash
uv sync --group dev
```

### Run tests

```bash
uv run pytest
```

### Render and preview the book

```bash
quarto render
quarto preview
```

The rendered site is written to `docs/` as configured in `_quarto.yml`.
