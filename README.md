# spotoptim-cookbook

[![Quarto CI](https://github.com/sequential-parameter-optimization/spotoptim-cookbook/actions/workflows/quarto.yml/badge.svg)](https://github.com/sequential-parameter-optimization/spotoptim-cookbook/actions/workflows/quarto.yml)
[![GitHub Pages](https://img.shields.io/website?url=https%3A%2F%2Fsequential-parameter-optimization.github.io%2Fspotoptim-cookbook%2F&label=GitHub%20Pages)](https://sequential-parameter-optimization.github.io/spotoptim-cookbook/)

Optimization cookbook.

## Workflow

- Work on `develop` and open pull requests to `main` for publication.
- CI (tests + Quarto render checks) runs on pushes to `develop`, pushes to `main`, and pull requests.
- GitHub Pages deployment runs only after changes land on `main`.
- No personal access token is required for this setup; the workflow uses GitHub’s built-in `GITHUB_TOKEN`.

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
