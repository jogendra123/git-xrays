from __future__ import annotations

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query

from git_xrays.infrastructure.run_store import RunStore
from git_xrays.web.models import (
    AnemicClassRow,
    ClusterRow,
    CognitiveRow,
    ComplexityFnRow,
    CouplingPairRow,
    DriftRow,
    EffortFileRow,
    FilePainRow,
    GodClassRow,
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


# Child endpoint definitions: (url_suffix, response_model, store_method_name)
_CHILD_ENDPOINTS: list[tuple[str, type, str]] = [
    ("hotspots", HotspotFile, "get_hotspot_files"),
    ("knowledge", KnowledgeFile, "get_knowledge_files"),
    ("coupling", CouplingPairRow, "get_coupling_pairs"),
    ("pain", FilePainRow, "get_file_pain"),
    ("anemic", AnemicClassRow, "get_anemic_classes"),
    ("complexity", ComplexityFnRow, "get_complexity_functions"),
    ("clusters", ClusterRow, "get_cluster_summaries"),
    ("drift", DriftRow, "get_cluster_drift"),
    ("effort", EffortFileRow, "get_effort_files"),
    ("cognitive", CognitiveRow, "get_dx_cognitive_files"),
    ("god-classes", GodClassRow, "get_god_classes"),
]


def _make_child_endpoint(model, store_method_name):
    """Factory to create a child endpoint handler."""
    def handler(run_id: str):
        _assert_run_exists(run_id)
        return [model(**r) for r in getattr(_store(), store_method_name)(run_id)]
    return handler


for _suffix, _model, _method in _CHILD_ENDPOINTS:
    app.get(
        f"/api/runs/{{run_id}}/{_suffix}",
        response_model=list[_model],
    )(_make_child_endpoint(_model, _method))


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
        "complexity_avg", "anemic_anemic_pct",
        "effort_model_r_squared", "clustering_silhouette",
        "god_class_god_pct",
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
