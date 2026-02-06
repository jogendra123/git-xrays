# SPEC.md
## Behavioral & Architectural Code Intelligence Platform

### Version
v1.0 (Final Consolidated Specification)

---

## 1. Vision

Build a **time-aware, architecture-first code intelligence system** that analyzes Git repositories to:
- Measure **behavioral, architectural, and socio-technical signals**
- Explain **where effort, risk, and friction originate**
- Track **trends across the full Git history**
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
- No static “quality score” abstractions
- No mandatory cloud or SaaS dependency

---

## 3. Target Users

- Staff / Principal Engineers
- Architects
- Engineering Managers
- Platform & DevEx teams
- Due diligence & modernization teams

---

## 4. Architecture Principles

- Clean Architecture (domain-first, dependency inversion)
- Hohpe-style narratives over dashboards
- Metrics must be:
  - Trendable
  - Explainable
  - Actionable

---

## 5. Technology Stack

- Python 3.13+
- uv (dependency & environment management)
- DuckDB (on-disk analytics)
- Git (CLI + libgit2 bindings)
- Follow TDD practices
- Optional: Streamlit / FastAPI (later phase)

---

## 6. System Architecture (Clean)

### Layers
- Domain: metrics, signals, models
- Application: use cases, analysis flows
- Infrastructure: git, AST parsers, storage
- Interface: CLI (first), UI (later)

---

## 7. Repository Input Modes

- Local checked-out repository
- Public Git URL
- Background job or synchronous CLI execution

---

## 8. Time Travel Support

- Analyze repo at:
  - Commit
  - Tag
  - Branch
  - Date
- Compare snapshots across time
- Immutable metric snapshots

---

## 9. Core Behavioral Metrics (Git-Based)

### 9.1 Change Frequency
Commits per file / module over time.

### 9.2 Code Churn
Added + deleted lines per time window.

### 9.3 Hotspot Score
High churn × high frequency.

### 9.4 Rework Ratio
Repeated changes to same lines/files.

### 9.5 Effort Distribution
Pareto-style distribution of effort across files/modules.

---

## 10. Temporal Coupling

Files or modules that change together frequently.

- File-level
- Module-level
- Deployment-unit level

---

## 11. Knowledge & Developer Risk

- Knowledge Distribution Index (KDI)
- Knowledge islands
- Knowledge decay
- Developer Risk Index (team-level only)

---

## 12. Architectural Dependency Analysis

### 12.1 Static Dependencies
Imports, references, package relationships.

### 12.2 Behavioral Dependencies
Derived from temporal coupling.

### 12.3 Module Monolith Detection
Identify:
- Multiple logical modules
- Single repo, multiple Docker images
- Shared deployment blast radius



## 14. PAIN Metric (Vlad Khononov)

PAIN = Size × Distance × Volatility

Tracked at:
- File
- Module
- Deployment unit

PAIN drift over time is a top refactoring signal.

---

## 15. Anemic Domain Model Detection

- Data–Behavior Separation Index
- Orchestration pressure
- Entity touch radius
- Anemic Model Score (AMS)

Tracked as trends, not absolutes.

---

## 16. AST-Based Complexity Metrics

- Cyclomatic complexity
- Nesting depth
- Branch count
- Exception paths
- Function length (AST-aware)

### Derived
- Complexity delta
- Complexity density (complexity / churn)

---

## 17. Effort Modeling (Explainable ML)

### 17.1 Effort Proxy Label
Derived from:
- Time between commits
- Rework ratio
- Complexity delta

### 17.2 Feature Set
- Churn
- Temporal coupling
- PAIN
- AST delta
- Touch radius

### 17.3 Models
- Linear / ElasticNet
- Small Gradient Boosted Trees

### 17.4 Output
Relative Effort Index (REI) ∈ [0,1]

Each score must provide feature attribution.

---

## 18. Change Grouping & Clustering

### 18.1 Feature Vectors
Per commit / PR.

### 18.2 K-Means Clustering
- Windowed by time
- Elbow / silhouette for k

### 18.3 Cluster Drift
Detect shifts from feature work → maintenance or refactoring.

---

## 19. DX Core 4 Overlay

Derived Git-based signals:
- Throughput
- Feedback delay
- Focus vs toil ratio
- Cognitive load proxy

Balanced, not gamed.

---

## 20. Trend Analysis (Mandatory)

All metrics must support:
- Rolling windows (30/90/180 days)
- Before/after comparison
- Longitudinal drift

---

## 21. Output Modes

- CLI text narratives
- JSON / Parquet exports
- DuckDB queries
- Optional UI later

---

## 22. CLI Design

Examples:
analyze-repo /repo --window 90d
analyze-repo /repo --architecture
analyze-repo /repo --effort
compare-repo /repo --from v1.0.0 --to v1.2.0

---

## 23. MVP Phases

### Phase 0
Scaffold, Git access, Clean Architecture.

### Phase 1
Behavioral Git metrics + hotspots.

### Phase 2
Knowledge risk & ownership.

### Phase 3
Architectural coupling + PAIN.

### Phase 5
Time travel & comparisons.

### Phase 6
Anemic domain model detection.

### Phase 7
AST complexity.

### Phase 8
Change clustering.

### Phase 9
Effort modeling (ML).

### Phase 10
DX Core overlay.

---

## 24. Ethics & Guardrails

- No individual scoring
- No hidden models
- All metrics explainable
- Architecture over judgment

---

## 25. Final Principle

"Software architecture is visible in how code changes over time."

This system exists to **make that visibility unavoidable**.
