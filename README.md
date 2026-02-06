# git-xrays

Behavioral & Architectural Code Intelligence Platform.

Analyzes Git repositories to measure behavioral, architectural, and socio-technical signals — hotspots, knowledge risk, temporal coupling, anemic domain models, complexity, change clustering, effort modeling, and developer experience.

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- Git

## Setup

```bash
uv sync                      # core (CLI + DuckDB)
uv sync --extra web          # web dashboard (FastAPI, Streamlit, Plotly)
uv sync --extra test         # test dependencies (pytest)
```

## Quick Start

```bash
# Basic hotspot analysis
uv run analyze-repo /path/to/repo

# All analyses + store results in DuckDB
uv run analyze-repo /path/to/repo --all

# Launch web dashboard
uv run analyze-repo --serve
```

## CLI Reference

```
analyze-repo [repo_path] [OPTIONS]
```

| Flag | Description |
|------|-------------|
| `--window DAYS` | Analysis window (e.g. `90d`, default: 90d) |
| `--knowledge` | Knowledge distribution analysis (KDI, islands, decay, DRI) |
| `--coupling` | Temporal coupling + PAIN metric |
| `--anemia` | Anemic domain model detection (DBSI, AMS) |
| `--complexity` | Function-level cyclomatic complexity + nesting depth |
| `--clustering` | K-Means change clustering + drift detection |
| `--effort` | Effort modeling with ridge regression (REI scores) |
| `--dx` | DX Core 4 overlay (throughput, feedback, focus, cognitive load) |
| `--at REF` | Anchor analysis at a commit, tag, branch, or ISO date |
| `--from REF` | Start ref for hotspot comparison (requires `--to`) |
| `--to REF` | End ref for hotspot comparison (requires `--from`) |
| `--all` | Run all 8 analyses, print output, store results in DuckDB |
| `--db PATH` | Custom DuckDB path (default: `~/.git-xrays/runs.db`) |
| `--list-runs` | Show past runs from DuckDB |
| `--serve` | Launch web dashboard (requires `git-xrays[web]`) |
| `--port PORT` | API port for `--serve` (Streamlit uses PORT+1, default: 8000) |

### Examples

```bash
# Hotspots + knowledge + coupling in one pass
uv run analyze-repo . --knowledge --coupling

# Snapshot at a tag
uv run analyze-repo . --all --at v1.0.0

# Compare two points in time
uv run analyze-repo . --from v1.0.0 --to v2.0.0

# List stored runs
uv run analyze-repo --list-runs

# Web dashboard on custom port
uv run analyze-repo --serve --port 9000
```

## Analyses

### Hotspot Analysis (default)
Identifies files with high change frequency and code churn. Hotspot score = normalized(churn) x normalized(frequency). Includes rework ratio and Pareto effort distribution.

### Knowledge Distribution (`--knowledge`)
Measures knowledge concentration using Shannon entropy (KDI = 1 - normalized entropy). Detects knowledge islands (primary author > 80% of churn), time-weighted decay (half-life = 90 days), and Developer Risk Index (minimum authors for > 50% of total churn).

### Temporal Coupling & PAIN (`--coupling`)
Finds files that change together using Jaccard similarity on co-commits (minimum 2 shared commits). PAIN metric per file: normalized(Size) x normalized(Distance) x normalized(Volatility), where Size = total churn, Volatility = commit count, Distance = mean coupling strength.

### Anemic Domain Model Detection (`--anemia`)
AST-based analysis of Python classes. Data-Behavior Separation Index (DBSI) = fields / (fields + behavior methods). Anemic Model Score (AMS) = DBSI x orchestration pressure. Classes with AMS > 0.5 are flagged. Includes cross-file touch count via import resolution.

### Complexity Analysis (`--complexity`)
AST-based cyclomatic complexity, max nesting depth, branch count, exception paths, and function length for all top-level functions and direct class methods. High complexity threshold: 10 (default).

### Change Clustering (`--clustering`)
Pure Python K-Means (Lloyd's algorithm) on per-commit feature vectors: [file_count, total_churn, add_ratio]. Auto-selects k (2-8) via silhouette score. Labels clusters as feature/bugfix/refactoring/config/mixed. Computes drift between first and second halves of the window.

### Effort Modeling (`--effort`)
Ridge regression (pure Python, Gauss-Jordan solver) on 5 features: code churn, change frequency, PAIN score, knowledge concentration, author count. Effort proxy = 0.5 x normalized(commit density) + 0.5 x normalized(rework ratio). Outputs Relative Effort Index (REI) in [0, 1] with per-file feature attribution.

### DX Core 4 Overlay (`--dx`)
Composite Developer Experience score from 4 metrics:
- **Throughput**: weighted commit rate (feature=1.0, refactoring=0.8, bugfix/mixed=0.5, config=0.3)
- **Feedback Delay**: mean(densities) x (1 - mean(rework ratios))
- **Focus Ratio**: feature commits / (feature + bugfix + config + refactoring)
- **Cognitive Load**: per-file mean of complexity, coordination, knowledge, change rate scores

DX Score = 0.3 x throughput + 0.25 x feedback + 0.25 x focus + 0.2 x (1 - cognitive load)

## Storage

`--all` stores each run in DuckDB (default `~/.git-xrays/runs.db`) with a UUID run ID. 11 tables: `runs` + 10 child tables (hotspot_files, knowledge_files, coupling_pairs, file_pain, anemia_classes, complexity_functions, cluster_summaries, cluster_drift, effort_files, dx_cognitive_files).

## Web Dashboard

Requires optional dependencies: `uv sync --extra web`

```bash
uv run analyze-repo . --all           # populate DuckDB first
uv run analyze-repo --serve           # http://localhost:8001
```

Architecture:
- **FastAPI** REST backend (uvicorn, default port 8000) reads from DuckDB via RunStore
- **Streamlit** frontend (port 8001) calls the API via httpx
- **Plotly** charts for interactive visualization

Dashboard features:
- Sidebar: repository selector, run picker (shows date + DX score), compare toggle
- 9 tabs: Overview (DX gauge + metric cards), Hotspots (table + bar chart), Knowledge (DRI + islands), Coupling (pairs + PAIN bar chart), Complexity (histogram + table), Clustering (pie chart + drift), Effort (REI bar chart), Anemia (class table), Time Travel (side-by-side comparison with deltas)

14 REST endpoints: `/api/repos`, `/api/runs`, `/api/runs/{id}`, `/api/runs/{id}/hotspots`, `/api/runs/{id}/knowledge`, `/api/runs/{id}/coupling`, `/api/runs/{id}/pain`, `/api/runs/{id}/anemia`, `/api/runs/{id}/complexity`, `/api/runs/{id}/clusters`, `/api/runs/{id}/drift`, `/api/runs/{id}/effort`, `/api/runs/{id}/cognitive`, `/api/compare`

## Project Structure

```
src/git_xrays/
├── domain/
│   ├── models.py              # 28 frozen dataclasses (all domain models)
│   └── ports.py               # GitRepository + SourceCodeReader protocols
├── application/
│   └── use_cases.py           # Analysis orchestration (8 analyses + comparison)
├── infrastructure/
│   ├── git_cli_reader.py      # GitRepository implementation (git CLI subprocess)
│   ├── git_source_reader.py   # SourceCodeReader implementation (git ls-tree + git show)
│   ├── ast_analyzer.py        # AST analysis for anemia detection
│   ├── complexity_analyzer.py # AST complexity metrics
│   ├── clustering_engine.py   # Pure Python K-Means
│   ├── effort_engine.py       # Ridge regression effort model
│   ├── dx_engine.py           # DX Core 4 computation
│   └── run_store.py           # DuckDB persistence (11 tables, read + write)
├── interface/
│   └── cli.py                 # argparse CLI (17 flags)
└── web/
    ├── models.py              # 14 Pydantic response models
    ├── api.py                 # FastAPI app (14 endpoints)
    ├── dashboard.py           # Streamlit frontend (9 tabs + Plotly)
    └── server.py              # uvicorn thread + streamlit subprocess launcher
```

## Testing

```bash
uv run pytest -v             # 582 tests
```

Tests mirror the source structure under `tests/`. Key patterns:
- `FakeGitRepository` and `FakeSourceCodeReader` in `tests/application/fakes.py`
- Per-phase fixtures in `tests/conftest.py` (commit_file, commit_files, create_tag, etc.)
- RunStore tests use `tmp_path` for DB isolation
- API tests use FastAPI `TestClient`

## Dependencies

- **Core**: `duckdb>=1.0`
- **Test**: `pytest>=8.0`
- **Web** (optional): `fastapi>=0.110`, `uvicorn[standard]>=0.29`, `streamlit>=1.35`, `httpx>=0.27`, `plotly>=5.20`

Zero external dependencies for all analysis engines (K-Means, ridge regression, AST parsing — all pure Python).
