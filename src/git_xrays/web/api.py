from __future__ import annotations

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query

from git_xrays.infrastructure.run_store import RunStore
from git_xrays.web.models import (
    AnemiaClassRow,
    ClusterRow,
    CognitiveRow,
    ComplexityFnRow,
    CouplingPairRow,
    DriftRow,
    EffortFileRow,
    FilePainRow,
    HotspotFile,
    KnowledgeFile,
    RunComparison,
    RunDetail,
    RunSummary,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db_path = getattr(app.state, "db_path", None)
    app.state.store = RunStore(db_path=db_path)
    yield
    app.state.store.close()


app = FastAPI(title="git-xrays", lifespan=lifespan)


def _store() -> RunStore:
    return app.state.store


def _run_dict_to_detail(row: dict) -> RunDetail:
    """Convert a raw runs dict to RunDetail, parsing JSON fields."""
    row = dict(row)
    if isinstance(row.get("effort_coefficients"), str):
        row["effort_coefficients"] = json.loads(row["effort_coefficients"])
    if isinstance(row.get("dx_weights"), str):
        row["dx_weights"] = json.loads(row["dx_weights"])
    return RunDetail(**row)


@app.get("/api/repos", response_model=list[str])
def list_repos():
    return _store().list_repos()


@app.get("/api/runs", response_model=list[RunSummary])
def list_runs(repo: str = Query(..., description="Repository path")):
    return [RunSummary(**r) for r in _store().list_runs_for_repo(repo)]


@app.get("/api/runs/{run_id}", response_model=RunDetail)
def get_run(run_id: str):
    row = _store().get_run(run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return _run_dict_to_detail(row)


@app.get("/api/runs/{run_id}/hotspots", response_model=list[HotspotFile])
def get_hotspots(run_id: str):
    _assert_run_exists(run_id)
    return [HotspotFile(**r) for r in _store().get_hotspot_files(run_id)]


@app.get("/api/runs/{run_id}/knowledge", response_model=list[KnowledgeFile])
def get_knowledge(run_id: str):
    _assert_run_exists(run_id)
    return [KnowledgeFile(**r) for r in _store().get_knowledge_files(run_id)]


@app.get("/api/runs/{run_id}/coupling", response_model=list[CouplingPairRow])
def get_coupling(run_id: str):
    _assert_run_exists(run_id)
    return [CouplingPairRow(**r) for r in _store().get_coupling_pairs(run_id)]


@app.get("/api/runs/{run_id}/pain", response_model=list[FilePainRow])
def get_pain(run_id: str):
    _assert_run_exists(run_id)
    return [FilePainRow(**r) for r in _store().get_file_pain(run_id)]


@app.get("/api/runs/{run_id}/anemia", response_model=list[AnemiaClassRow])
def get_anemia(run_id: str):
    _assert_run_exists(run_id)
    return [AnemiaClassRow(**r) for r in _store().get_anemia_classes(run_id)]


@app.get("/api/runs/{run_id}/complexity", response_model=list[ComplexityFnRow])
def get_complexity(run_id: str):
    _assert_run_exists(run_id)
    return [ComplexityFnRow(**r) for r in _store().get_complexity_functions(run_id)]


@app.get("/api/runs/{run_id}/clusters", response_model=list[ClusterRow])
def get_clusters(run_id: str):
    _assert_run_exists(run_id)
    return [ClusterRow(**r) for r in _store().get_cluster_summaries(run_id)]


@app.get("/api/runs/{run_id}/drift", response_model=list[DriftRow])
def get_drift(run_id: str):
    _assert_run_exists(run_id)
    return [DriftRow(**r) for r in _store().get_cluster_drift(run_id)]


@app.get("/api/runs/{run_id}/effort", response_model=list[EffortFileRow])
def get_effort(run_id: str):
    _assert_run_exists(run_id)
    return [EffortFileRow(**r) for r in _store().get_effort_files(run_id)]


@app.get("/api/runs/{run_id}/cognitive", response_model=list[CognitiveRow])
def get_cognitive(run_id: str):
    _assert_run_exists(run_id)
    return [CognitiveRow(**r) for r in _store().get_dx_cognitive_files(run_id)]


@app.get("/api/compare", response_model=RunComparison)
def compare_runs(
    a: str = Query(..., description="First run ID"),
    b: str = Query(..., description="Second run ID"),
):
    row_a = _store().get_run(a)
    row_b = _store().get_run(b)
    if row_a is None:
        raise HTTPException(status_code=404, detail=f"Run {a} not found")
    if row_b is None:
        raise HTTPException(status_code=404, detail=f"Run {b} not found")

    detail_a = _run_dict_to_detail(row_a)
    detail_b = _run_dict_to_detail(row_b)

    delta_fields = [
        "dx_score", "dx_throughput", "dx_feedback_delay",
        "dx_focus_ratio", "dx_cognitive_load",
        "complexity_avg", "anemia_anemic_pct",
        "effort_model_r_squared", "clustering_silhouette",
    ]
    deltas = {}
    for field in delta_fields:
        va = getattr(detail_a, field)
        vb = getattr(detail_b, field)
        deltas[field] = round(vb - va, 6)

    return RunComparison(run_a=detail_a, run_b=detail_b, deltas=deltas)


def _assert_run_exists(run_id: str) -> None:
    if _store().get_run(run_id) is None:
        raise HTTPException(status_code=404, detail="Run not found")
