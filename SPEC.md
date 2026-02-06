# SPEC.md
## Behavioral & Architectural Code Intelligence Platform

### Version
v2.0 (Post-Implementation — reflects actual system as built)

---

## 1. Vision

A **time-aware, architecture-first code intelligence system** that analyzes Git repositories to:
- Measure **behavioral, architectural, and socio-technical signals**
- Explain **where effort, risk, and friction originate**
- Track **trends across the full Git history** via DuckDB storage
- Remain **explainable, vendor-neutral, and ML-light**

Inspired by:
- Adam Tornhill (Software X-Rays, Code as a Crime Scene)
- Vlad Khononov (PAIN, coupling balance)
- Gregor Hohpe (architecture as decision records & forces)
- DX Core 4 (developer productivity signals)

---

## 2. Non-Goals

- No individual developer performance scoring
- No black-box AI or LLM-based scoring
- No static "quality score" abstractions
- No mandatory cloud or SaaS dependency

---

## 3. Target Users

- Staff / Principal Engineers
- Architects
- Engineering Managers
- Platform & DevEx teams
- Due diligence & modernization teams

---

## 4. Architecture

### 4.1 Clean Architecture (as implemented)

```
src/git_xrays/
├── domain/          # Models (28 frozen dataclasses), Ports (2 Protocols)
├── application/     # Use cases (8 analyses + comparison orchestration)
├── infrastructure/  # Git CLI, AST parsers, engines, DuckDB storage
├── interface/       # CLI (argparse, 17 flags)
└── web/             # FastAPI REST API + Streamlit dashboard (optional)
```

**Domain layer** has zero external dependencies. All models are frozen dataclasses. Two Protocol interfaces (`GitRepository`, `SourceCodeReader`) define the ports.

**Infrastructure layer** contains all I/O and computation engines. Analysis engines (K-Means, ridge regression, AST parsing) are pure Python with zero external dependencies beyond the standard library.

**Interface layer** is a thin CLI using argparse. Web layer is optional.

### 4.2 Protocols

```python
class GitRepository(Protocol):
    def commit_count(self) -> int: ...
    def first_commit_date(self) -> datetime | None: ...
    def last_commit_date(self) -> datetime | None: ...
    def file_changes(self, since=None, until=None) -> list[FileChange]: ...
    def resolve_ref(self, ref: str) -> datetime: ...

class SourceCodeReader(Protocol):
    def list_python_files(self, ref=None) -> list[str]: ...
    def read_file(self, file_path: str, ref=None) -> str: ...
```

Implementations: `GitCliReader` (git subprocess), `GitSourceReader` (git ls-tree + git show).

---

## 5. Technology Stack (as implemented)

| Component | Technology | Notes |
|-----------|-----------|-------|
| Language | Python 3.13+ | Type hints throughout |
| Package manager | uv | Lock file, extras support |
| Storage | DuckDB >= 1.0 | 11 tables, single-file DB |
| Git access | Git CLI (subprocess) | No libgit2 |
| AST parsing | Python `ast` module | stdlib only |
| ML | Pure Python ridge regression | Gauss-Jordan solver, no numpy/scipy |
| Clustering | Pure Python K-Means | Lloyd's algorithm, no sklearn |
| Web API | FastAPI >= 0.110 | Optional, 14 endpoints |
| Web UI | Streamlit >= 1.35 | Optional, 9 tabs |
| Charts | Plotly >= 5.20 | Optional, interactive |
| HTTP client | httpx >= 0.27 | Dashboard to API |
| Testing | pytest >= 8.0 | 582 tests, TDD workflow |

---

## 6. Repository Input

- Local checked-out repository (path argument)
- Synchronous CLI execution
- Results stored per-run in DuckDB for trend analysis

---

## 7. Time Travel Support (implemented)

### `--at REF`
Anchors analysis at a specific point in time. Accepts commits, tags, branches, or ISO dates. Resolves to a datetime via `git log -1 --format=%aI <ref>`, then passes as `current_time` to analysis functions.

### `--from REF --to REF`
Compares hotspot snapshots between two points. Computes per-file deltas (score, churn, frequency) and assigns status: `unchanged`, `improved`, `degraded`, `new`, `removed`.

### DuckDB Time Travel
Each `--all` run is stored with a UUID. `--list-runs` shows past runs. The web dashboard allows selecting and comparing any two runs.

---

## 8. Core Behavioral Metrics (Phase 1)

### 8.1 Change Frequency
Commits per file within the time window.

### 8.2 Code Churn
Lines added + lines deleted per file.

### 8.3 Hotspot Score
`normalized(churn) * normalized(frequency)` — identifies files that change often and change a lot.

### 8.4 Rework Ratio
`(frequency - 1) / frequency` — approaches 1.0 for frequently modified files.

### 8.5 Effort Distribution
Pareto analysis: reports how many files account for 50%/80%/90% of total churn.

**Domain model**: `FileMetrics`, `HotspotReport`

---

## 9. Knowledge & Developer Risk (Phase 2)

### 9.1 Knowledge Distribution Index (KDI)
`KDI = 1 - normalized_shannon_entropy` over author churn proportions. 1.0 = single author, 0.0 = perfectly distributed.

### 9.2 Knowledge Islands
Files where `primary_author_pct > 0.8`. Flagged as bus-factor risks.

### 9.3 Knowledge Decay
Time-weighted contributions: `weight = 2^(-age_days / half_life)`, half_life = 90 days. Recent contributions count more.

### 9.4 Developer Risk Index (DRI)
Minimum number of authors needed to cover > 50% of total weighted churn. Team-level metric only.

**Domain model**: `AuthorContribution`, `FileKnowledge`, `KnowledgeReport`

---

## 10. Temporal Coupling & PAIN (Phase 3)

### 10.1 Coupling Detection
Jaccard similarity on co-commits: `shared_commits / union_commits`. Minimum 2 shared commits to qualify. Reports coupling strength and support (shared / total commits).

### 10.2 PAIN Metric (Vlad Khononov)
Per-file: `PAIN = normalized(Size) * normalized(Distance) * normalized(Volatility)`
- **Size**: total code churn
- **Volatility**: commit count
- **Distance**: mean coupling strength to other files

**Domain model**: `CouplingPair`, `FilePain`, `CouplingReport`

---

## 11. Anemic Domain Model Detection (Phase 6)

AST-based Python class analysis:

- **Fields**: class-level attributes + `self.x` assignments in `__init__` only
- **Behavior methods**: non-dunder, non-property methods containing `if`/`for`/`while`/`try`/`with`
- **DBSI** = `field_count / (field_count + behavior_method_count)`
- **Logic density** = methods with logic / non-dunder non-property methods
- **Orchestration pressure** = `1 - logic_density`
- **AMS** = `DBSI * orchestration_pressure` (threshold: 0.5)
- **Touch count**: heuristic import resolution across `.py` files

Uses `SourceCodeReader` protocol. Supports `--at` for point-in-time snapshots.

**Domain model**: `ClassMetrics`, `FileAnemic`, `AnemicReport`

---

## 12. AST Complexity Metrics (Phase 7)

Per-function metrics via Python `ast` module:

- **Cyclomatic complexity**: `1 + if/elif/for/while/except/assert/IfExp + BoolOp(len(values)-1)`
- **Max nesting depth**: recursive visitor for `if`/`for`/`while`/`with`/`try` (elif counts as nested)
- **Branch count**: `ast.If` node count
- **Exception paths**: `ast.ExceptHandler` count
- **Length**: `end_lineno - lineno + 1`

Analyzes top-level functions and direct class methods (skips nested defs). High complexity threshold: 10.

**Domain model**: `FunctionComplexity`, `FileComplexity`, `ComplexityReport`

---

## 13. Change Clustering (Phase 8)

Pure Python K-Means (Lloyd's algorithm), zero dependencies:

- **Feature vector per commit**: `[file_count, total_churn, add_ratio]`
- **Normalization**: min-max per feature
- **Auto-k selection**: tries k=2..8, picks highest silhouette score
- **Cluster labeling**: based on normalized centroids — `feature`, `bugfix`, `refactoring`, `config`, or `mixed`
- **Drift detection**: first-half vs second-half percentage comparison, `abs(drift) < 5` = stable

**Domain model**: `CommitFeatures`, `ClusterSummary`, `ClusterDrift`, `ClusteringReport`

---

## 14. Effort Modeling (Phase 9)

Ridge regression via Gauss-Jordan elimination, zero dependencies:

### Features (5)
`[code_churn, change_frequency, pain_score, knowledge_concentration, author_count]`

### Effort Proxy Label
`0.5 * normalized(commit_density) + 0.5 * normalized(rework_ratio)`

Commit density = `1 / (1 + median_interval_days)`.

### Output
- **Relative Effort Index (REI)** in [0, 1] per file (dot product + min-max normalization)
- **Feature attribution**: per-file breakdown of each feature's contribution
- **Model R-squared**: goodness of fit
- **Fallback**: < 3 files = equal weights (0.2 each), R² = 0.0

**Domain model**: `FeatureAttribution`, `FileEffort`, `EffortReport`

---

## 15. DX Core 4 Overlay (Phase 10)

Composite developer experience score from 4 git-derived metrics:

| Metric | Formula |
|--------|---------|
| Throughput | Weighted commit rate: feature=1.0, refactoring=0.8, bugfix/mixed=0.5, config=0.3 |
| Feedback Delay | `mean(densities) * (1 - mean(rework_ratios))` |
| Focus Ratio | `feature_commits / (feature + bugfix + config + refactoring)`, mixed excluded, empty = 0.5 |
| Cognitive Load | Per-file mean of (complexity, coordination, knowledge, change_rate) scores |

**DX Score** = `0.3*throughput + 0.25*feedback + 0.25*focus + 0.2*(1-cognitive_load)`

Internally runs hotspot + knowledge + coupling + clustering + complexity analyses.

**Domain model**: `FileCognitiveLoad`, `DXMetrics`, `DXReport`

---

## 16. DuckDB Storage (Phase 11)

`--all` persists every run to DuckDB (default `~/.git-xrays/runs.db`).

### Schema: 11 tables
| Table | Primary Key | Records |
|-------|-------------|---------|
| `runs` | `run_id` | 1 per run (32 scalar columns) |
| `hotspot_files` | `(run_id, file_path)` | Per-file hotspot metrics |
| `knowledge_files` | `(run_id, file_path)` | Per-file knowledge metrics |
| `coupling_pairs` | `(run_id, file_a, file_b)` | Coupling pair data |
| `file_pain` | `(run_id, file_path)` | PAIN scores |
| `anemia_classes` | `(run_id, file_path, class_name)` | Class-level anemia |
| `complexity_functions` | `(run_id, file_path, function_name, line_number)` | Function complexity |
| `cluster_summaries` | `(run_id, cluster_id)` | Cluster centroids + labels |
| `cluster_drift` | none (labels can duplicate) | Drift per cluster label |
| `effort_files` | `(run_id, file_path)` | REI scores |
| `dx_cognitive_files` | `(run_id, file_path)` | Cognitive load breakdown |

JSON-encoded fields in `runs`: `effort_coefficients`, `dx_weights`.

### Read Methods (13)
`get_run`, `list_repos`, `list_runs_for_repo`, `get_hotspot_files`, `get_knowledge_files`, `get_coupling_pairs`, `get_file_pain`, `get_anemia_classes`, `get_complexity_functions`, `get_cluster_summaries`, `get_cluster_drift`, `get_effort_files`, `get_dx_cognitive_files` — all use a generic `_query_child` helper.

---

## 17. Web Dashboard (Phase 12)

Optional dependency group `[web]`: FastAPI, uvicorn, Streamlit, httpx, Plotly.

### Architecture

```
analyze-repo --serve [--db PATH] [--port 8000]
        │
        ├── FastAPI (uvicorn, port 8000)  ← reads DuckDB via RunStore
        │     └── 14 REST endpoints under /api/
        │
        └── Streamlit (port 8001)  ← calls FastAPI via httpx
              └── Sidebar + 9 tabs + Plotly charts
```

Server orchestration: uvicorn runs in a daemon thread, Streamlit launches as a subprocess. Lazy import with clear error if web deps are missing.

### REST API (14 endpoints)

| Method | Route | Response |
|--------|-------|----------|
| GET | `/api/repos` | `list[str]` |
| GET | `/api/runs?repo=<path>` | `list[RunSummary]` |
| GET | `/api/runs/{run_id}` | `RunDetail` |
| GET | `/api/runs/{run_id}/hotspots` | `list[HotspotFile]` |
| GET | `/api/runs/{run_id}/knowledge` | `list[KnowledgeFile]` |
| GET | `/api/runs/{run_id}/coupling` | `list[CouplingPairRow]` |
| GET | `/api/runs/{run_id}/pain` | `list[FilePainRow]` |
| GET | `/api/runs/{run_id}/anemia` | `list[AnemiaClassRow]` |
| GET | `/api/runs/{run_id}/complexity` | `list[ComplexityFnRow]` |
| GET | `/api/runs/{run_id}/clusters` | `list[ClusterRow]` |
| GET | `/api/runs/{run_id}/drift` | `list[DriftRow]` |
| GET | `/api/runs/{run_id}/effort` | `list[EffortFileRow]` |
| GET | `/api/runs/{run_id}/cognitive` | `list[CognitiveRow]` |
| GET | `/api/compare?a={id}&b={id}` | `RunComparison` |

14 Pydantic response models in `web/models.py`. FastAPI lifespan context manager initializes/closes RunStore. 404 for missing run IDs.

### Dashboard (9 tabs)

1. **Overview** — DX Score gauge (Plotly), 4 core metric cards, summary stats row
2. **Hotspots** — Data table + horizontal bar chart (top 20 by hotspot score)
3. **Knowledge** — DRI + island count metrics, file table with island indicators
4. **Coupling** — Coupling pairs table, PAIN scores table, top 20 PAIN bar chart
5. **Complexity** — Summary stats, function table, cyclomatic complexity histogram
6. **Clustering** — Pie chart of cluster sizes, summary table, drift table with trend arrows
7. **Effort** — R² display, REI file table, top 20 REI bar chart
8. **Anemia** — Summary stats, class-level table with AMS values
9. **Time Travel** — Two-column layout, DX score comparison with delta, per-metric delta cards, side-by-side hotspot tables

Data layer: `httpx.get()` calls cached with `@st.cache_data(ttl=60)`.

---

## 18. CLI Design (as implemented)

```
analyze-repo [repo_path] [OPTIONS]

  repo_path              Path to a local git repository (optional for --list-runs, --serve)

  --window DAYS          Analysis window (e.g. 90d, default: 90d)
  --knowledge            Knowledge distribution analysis
  --coupling             Temporal coupling and PAIN analysis
  --anemia               Anemic domain model analysis
  --complexity           Function-level complexity analysis
  --clustering           Change clustering analysis
  --effort               Effort modeling (REI scores)
  --dx                   DX Core 4 analysis
  --at REF               Anchor at commit/tag/branch/date
  --from REF             Start ref for comparison (requires --to)
  --to REF               End ref for comparison (requires --from)
  --all                  Run all 8 analyses + store in DuckDB
  --db PATH              Custom DuckDB path (default: ~/.git-xrays/runs.db)
  --list-runs            Show past runs from DuckDB
  --serve                Launch web dashboard
  --port PORT            API port for --serve (default: 8000)
```

Mutual exclusivity: `--at` cannot combine with `--from/--to`. `--all` cannot combine with `--from/--to`.

---

## 19. Implementation Phases (completed)

| Phase | Feature | Tests Added | Total |
|-------|---------|-------------|-------|
| 0 | Scaffold, Clean Architecture, CLI, GitCliReader | — | — |
| 1 | Hotspot analysis (frequency, churn, score, rework) | — | — |
| 2 | Knowledge risk (KDI, islands, decay, DRI) | — | 86 |
| 3 | Temporal coupling + PAIN metric | 44 | 130 |
| 5 | Time travel (`--at`, `--from/--to`, comparison) | 50 | 180 |
| 6 | Anemic domain model detection | 58 | 238 |
| 7 | AST complexity metrics | 64 | 302 |
| 8 | Change clustering (K-Means) | 62 | 364 |
| 9 | Effort modeling (ridge regression) | 60 | 424 |
| 10 | DX Core 4 overlay | 60 | 484 |
| 11 | CLI `--all` + DuckDB storage | 52 | 536 |
| 12 | Web dashboard (FastAPI + Streamlit + Plotly) | 46 | 582 |

---

## 20. Testing

582 tests using pytest with TDD workflow (RED-GREEN-REFACTOR).

- **Unit tests**: domain models, use cases (with `FakeGitRepository`/`FakeSourceCodeReader`), all engines
- **Integration tests**: `GitCliReader` and `GitSourceReader` against real temp git repos (via `conftest.py` fixtures)
- **Storage tests**: `RunStore` with isolated DuckDB files (`tmp_path`)
- **API tests**: FastAPI `TestClient` with pre-populated DuckDB

```bash
uv run pytest -v
```

---

## 21. Ethics & Guardrails

- No individual scoring
- No hidden models
- All metrics explainable with formulas documented
- Architecture over judgment
- Ridge regression coefficients and R² are always exposed
- Feature attribution provided for every REI score

---

## 22. Final Principle

"Software architecture is visible in how code changes over time."

This system exists to **make that visibility unavoidable**.
