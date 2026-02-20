# git-xrays

Behavioral & Architectural Code Intelligence Platform.

Analyzes Git repositories to measure behavioral, architectural, and socio-technical signals — hotspots, knowledge risk, temporal coupling, anemic domain models, god classes, complexity, change clustering, effort modeling, and developer experience. Supports both Python and Java codebases.

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
| `--anemic` | Anemic domain model detection (DBSI, AMS) — Python + Java |
| `--god-class` | God class detection (WMC, TCC, GCS) — Python + Java |
| `--complexity` | Function-level cyclomatic + cognitive complexity — Python + Java |
| `--clustering` | K-Means change clustering + drift detection |
| `--effort` | Effort modeling with ridge regression (REI scores) |
| `--dx` | DX Core 4 overlay (throughput, feedback, focus, cognitive load) |
| `--at REF` | Anchor analysis at a commit, tag, branch, or ISO date |
| `--from REF` | Start ref for hotspot comparison (requires `--to`) |
| `--to REF` | End ref for hotspot comparison (requires `--from`) |
| `--all` | Run all 9 analyses, print output, store results in DuckDB |
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
Identifies files with high change frequency and code churn. Uses temporal decay weighting (half-life = 30 days) and relative churn (churn/file_size). Hotspot score = normalized(weighted_churn) x normalized(weighted_frequency). Includes rework ratio (14-day temporal window) and Pareto effort distribution.

### Knowledge Distribution (`--knowledge`)
Measures knowledge concentration using recency-weighted Shannon entropy (KDI = 1 - normalized entropy). Detects knowledge islands (primary author > 80% of churn), time-weighted decay (half-life = 90 days), and Developer Risk Index (Gini coefficient of author churn, [0,1]).

### Temporal Coupling & PAIN (`--coupling`)
Finds files that change together using temporal-proximity Jaccard similarity on co-commits (minimum 2 shared commits, lift > 1.0 filter). PAIN metric per file: normalized(Size) x normalized(Distance) x normalized(Volatility), where Size = total churn, Volatility = commit count, Distance = mean coupling strength.

### Anemic Domain Model Detection (`--anemic`)
AST-based analysis of Python and Java classes. Data-Behavior Separation Index (DBSI) = fields / (fields + behavior methods). Anemic Model Score (AMS) = DBSI x orchestration pressure. Classes with AMS > 0.5 are flagged. Includes cross-file touch count via import resolution. Java support via tree-sitter with getter/setter detection.

### God Class Detection (`--god-class`)
Detects classes that do too much — too many methods, too many fields, high complexity, low cohesion. Supports Python (AST) and Java (tree-sitter).
- **WMC** (Weighted Methods per Class): sum of cyclomatic complexity of all methods
- **TCC** (Tight Class Cohesion): fraction of method pairs sharing field access
- **GCS** = 0.3 x norm(methods) + 0.3 x norm(WMC) + 0.2 x norm(fields) + 0.2 x (1 - TCC)
- Classes with GCS > 0.6 (default threshold) are flagged as god classes

### Complexity Analysis (`--complexity`)
Cyclomatic and cognitive complexity (SonarSource algorithm), max nesting depth, branch count, exception paths, and function length. Supports Python (AST) and Java (tree-sitter). High complexity threshold: 10 (default).

### Change Clustering (`--clustering`)
Pure Python K-Means++ (Lloyd's algorithm with K-Means++ initialization) on per-commit feature vectors: [file_count, total_churn, add_ratio]. Auto-selects k (2-8) via silhouette score. Labels clusters as feature/bugfix/refactoring/config/mixed. Computes drift between first and second halves of the window.

### Effort Modeling (`--effort`)
Ridge regression (pure Python, Gauss-Jordan solver) on 6 features: code churn, change frequency, PAIN score, knowledge concentration, author count, and knowledge x pain interaction term. Auto-tunes alpha via grid search. Effort proxy = 0.5 x normalized(commit density) + 0.5 x normalized(rework ratio). Outputs Relative Effort Index (REI) in [0, 1] with per-file feature attribution.

### DX Core 4 Overlay (`--dx`)
Composite Developer Experience score from 4 metrics:
- **Throughput**: weighted commit rate (feature=1.0, refactoring=0.8, bugfix/mixed=0.5, config=0.3)
- **Feedback Delay**: mean(densities) x (1 - mean(rework ratios))
- **Focus Ratio**: feature commits / (feature + bugfix + config + refactoring)
- **Cognitive Load**: per-file weighted average (0.35 x complexity + 0.25 x coordination + 0.25 x knowledge + 0.15 x change_rate)

DX Score = 0.3 x throughput + 0.25 x feedback + 0.25 x focus + 0.2 x (1 - cognitive load)

## Storage

`--all` stores each run in DuckDB (default `~/.git-xrays/runs.db`) with a UUID run ID. 12 tables: `runs` + 11 child tables (hotspot_files, knowledge_files, coupling_pairs, file_pain, anemic_classes, complexity_functions, cluster_summaries, cluster_drift, effort_files, dx_cognitive_files, god_class_classes).

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
- 10 tabs: Overview (DX gauge + metric cards), Hotspots (table + bar chart), Knowledge (DRI + islands), Coupling (pairs + PAIN bar chart), Complexity (histogram + table), Clustering (pie chart + drift), Effort (REI bar chart), Anemia (class table), God Classes (GCS table + bar chart), Time Travel (side-by-side comparison with deltas)

15 REST endpoints: `/api/repos`, `/api/runs`, `/api/runs/{id}`, `/api/runs/{id}/hotspots`, `/api/runs/{id}/knowledge`, `/api/runs/{id}/coupling`, `/api/runs/{id}/pain`, `/api/runs/{id}/anemic`, `/api/runs/{id}/complexity`, `/api/runs/{id}/clusters`, `/api/runs/{id}/drift`, `/api/runs/{id}/effort`, `/api/runs/{id}/cognitive`, `/api/runs/{id}/god-classes`, `/api/compare`

## Project Structure

```
src/git_xrays/
├── domain/
│   ├── models.py                  # 31 frozen dataclasses (all domain models)
│   └── ports.py                   # GitRepository + SourceCodeReader protocols
├── application/
│   └── use_cases.py               # Analysis orchestration (9 analyses + comparison)
├── infrastructure/
│   ├── git_cli_reader.py          # GitRepository implementation (git CLI subprocess)
│   ├── git_source_reader.py       # SourceCodeReader implementation (git ls-tree + git show)
│   ├── ast_analyzer.py            # Python AST analysis for anemic model detection
│   ├── java_anemic_analyzer.py    # Java tree-sitter anemic model detection
│   ├── complexity_analyzer.py     # Python AST complexity metrics
│   ├── java_complexity_analyzer.py # Java tree-sitter complexity metrics
│   ├── god_class_analyzer.py      # Python AST god class detection
│   ├── java_god_class_analyzer.py # Java tree-sitter god class detection
│   ├── clustering_engine.py       # Pure Python K-Means++
│   ├── effort_engine.py           # Ridge regression effort model
│   ├── dx_engine.py               # DX Core 4 computation
│   └── run_store.py               # DuckDB persistence (12 tables, read + write)
├── interface/
│   └── cli.py                     # argparse CLI (17 flags)
└── web/
    ├── models.py                  # 14 Pydantic response models
    ├── api.py                     # FastAPI app (15 endpoints)
    ├── dashboard.py               # Streamlit frontend (10 tabs + Plotly)
    └── server.py                  # uvicorn thread + streamlit subprocess launcher
```

## Testing

```bash
uv run pytest -v             # 752 tests
```

Tests mirror the source structure under `tests/`. Key patterns:
- `FakeGitRepository` and `FakeSourceCodeReader` in `tests/application/fakes.py`
- Per-phase fixtures in `tests/conftest.py` (commit_file, commit_files, create_tag, etc.)
- RunStore tests use `tmp_path` for DB isolation
- API tests use FastAPI `TestClient`

## Dependencies

- **Core**: `duckdb>=1.0`, `tree-sitter>=0.23`, `tree-sitter-java>=0.23`
- **Test**: `pytest>=8.0`
- **Web** (optional): `fastapi>=0.110`, `uvicorn[standard]>=0.29`, `streamlit>=1.35`, `httpx>=0.27`, `plotly>=5.20`

Zero external dependencies for Python analysis engines (K-Means, ridge regression, AST parsing — all pure Python). Java support uses tree-sitter for parsing.
