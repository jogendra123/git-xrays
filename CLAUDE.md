# CLAUDE.md

## Project Overview

git-xrays is a Behavioral & Architectural Code Intelligence Platform. It analyzes Git repositories to measure behavioral, architectural, and socio-technical signals — hotspots, knowledge risk, temporal coupling, anemic domain models, complexity, change clustering, effort modeling, and developer experience.

## Tech Stack

- Python 3.13+
- Package manager: [uv](https://docs.astral.sh/uv/)
- Build system: Hatchling
- Storage: DuckDB
- Web: FastAPI + Streamlit + Plotly (optional)
- Testing: pytest
- All analysis engines are pure Python (zero external dependencies for K-Means, ridge regression, AST parsing)

## Architecture

Hexagonal / ports-and-adapters layout under `src/git_xrays/`:

- **domain/** — Frozen dataclasses (`models.py`) and protocols (`ports.py`)
- **application/** — Use cases orchestrating 8 analyses + comparison (`use_cases.py`)
- **infrastructure/** — Git CLI adapter, AST analyzers, clustering/effort/DX engines, DuckDB persistence
- **interface/** — argparse CLI (`cli.py`)
- **web/** — FastAPI REST API, Streamlit dashboard, Pydantic response models

## Common Commands

```bash
# Install dependencies
uv sync                      # core (CLI + DuckDB)
uv sync --extra web          # web dashboard
uv sync --extra test         # test dependencies

# Run tests
uv run pytest -v             # 582 tests

# Run the CLI
uv run analyze-repo /path/to/repo          # basic hotspot analysis
uv run analyze-repo /path/to/repo --all    # all 8 analyses + DuckDB storage

# Web dashboard
uv run analyze-repo --serve                # FastAPI on :8000, Streamlit on :8001
```

## Testing Conventions

- Tests mirror the source structure under `tests/`
- Fakes: `FakeGitRepository` and `FakeSourceCodeReader` in `tests/application/fakes.py`
- Shared fixtures in `tests/conftest.py` (commit_file, commit_files, create_tag, etc.)
- RunStore tests use `tmp_path` for DB isolation
- API tests use FastAPI `TestClient`

## Key Design Decisions

- Domain models are frozen dataclasses (immutable)
- Ports are defined as Python protocols (structural typing)
- No external ML/math libraries — all algorithms implemented in pure Python
- DuckDB for persistence with 11 tables (runs + 10 child tables)
- CLI entry point: `analyze-repo` (defined in pyproject.toml scripts)
