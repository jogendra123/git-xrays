import json
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from git_xrays.domain.models import (
    AnemicReport,
    ClassMetrics,
    ClusterDrift,
    ClusteringReport,
    ClusterSummary,
    CommitFeatures,
    ComplexityReport,
    CouplingPair,
    CouplingReport,
    DXMetrics,
    DXReport,
    EffortReport,
    FeatureAttribution,
    FileAnemic,
    FileCognitiveLoad,
    FileComplexity,
    FileEffort,
    FileKnowledge,
    FileMetrics,
    FilePain,
    FunctionComplexity,
    HotspotReport,
    KnowledgeReport,
    RepoSummary,
)
from git_xrays.infrastructure.run_store import RunStore
from git_xrays.web.api import app


_NOW = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
_SINCE = datetime(2024, 10, 17, 12, 0, 0, tzinfo=timezone.utc)


def _make_all_reports():
    summary = RepoSummary("/repo", 42, datetime(2024, 1, 1, tzinfo=timezone.utc), _NOW)
    hotspot = HotspotReport("/repo", 90, _SINCE, _NOW, 10, [
        FileMetrics("src/a.py", 5, 100, 1.0, 0.8),
        FileMetrics("src/b.py", 3, 60, 0.36, 0.6667),
    ])
    knowledge = KnowledgeReport("/repo", 90, _SINCE, _NOW, 10, 2, 1, [
        FileKnowledge("src/a.py", 0.85, "Alice", 0.9, True, 2, []),
    ])
    coupling = CouplingReport("/repo", 90, _SINCE, _NOW, 10, [
        CouplingPair("src/a.py", "src/b.py", 3, 10, 0.75, 0.3),
    ], [
        FilePain("src/a.py", 100, 1.0, 5, 1.0, 0.75, 1.0, 1.0),
    ])
    anemia = AnemicReport("/repo", None, 1, 1, 1, 100.0, 0.75, 0.5, [
        FileAnemic("src/a.py", 1, 1, 0.75, [
            ClassMetrics("UserDTO", "src/a.py", 3, 2, 0, 0, 0, 1.0, 0.0, 1.0, 0.75),
        ], 2),
    ])
    complexity = ComplexityReport("/repo", None, 1, 2, 3.5, 5, 0, 10, 8.0, 12, [
        FileComplexity("src/a.py", 2, 7, 3.5, 5, "process", 8.0, 12, 2.0, 3, [
            FunctionComplexity("process", "src/a.py", None, 1, 12, 5, 3, 2, 1),
            FunctionComplexity("helper", "src/a.py", None, 14, 4, 2, 1, 1, 0),
        ]),
    ])
    clustering = ClusteringReport("/repo", 90, _SINCE, _NOW, 10, 2, 0.65, [
        ClusterSummary(0, "feature", 6, 3.0, 50.0, 0.8, [
            CommitFeatures("abc", _NOW, 3, 50, 0.8),
        ]),
        ClusterSummary(1, "bugfix", 4, 1.0, 10.0, 0.5, [
            CommitFeatures("def", _NOW, 1, 10, 0.5),
        ]),
    ], [
        ClusterDrift("feature", 55.0, 50.0, -5.0, "shrinking"),
        ClusterDrift("bugfix", 45.0, 50.0, 5.0, "growing"),
    ])
    effort = EffortReport("/repo", 90, _SINCE, _NOW, 2, 0.85, 1.0,
        ["code_churn", "change_frequency", "pain_score",
         "knowledge_concentration", "author_count"],
        [0.3, 0.25, 0.2, 0.15, 0.1], [
            FileEffort("src/a.py", 0.9, 0.8, 0.7, 0.6, [
                FeatureAttribution("code_churn", 100.0, 0.3, 0.27),
            ]),
        ])
    dx = DXReport("/repo", 90, _SINCE, _NOW, 10, 2, 0.72, [0.3, 0.25, 0.25, 0.2],
        DXMetrics(0.8, 0.7, 0.65, 0.35), [
            FileCognitiveLoad("src/a.py", 0.6, 0.5, 0.4, 0.3, 0.45),
        ])
    return summary, hotspot, knowledge, coupling, anemia, complexity, clustering, effort, dx


@pytest.fixture
def client(tmp_path):
    db_file = str(tmp_path / "test.db")
    app.state.db_path = db_file
    store = RunStore(db_path=db_file)
    summary, hotspot, knowledge, coupling, anemia, complexity, clustering, effort, dx = _make_all_reports()
    store.save_run("run-1", "/repo-a", 90, summary, hotspot, knowledge,
                    coupling, anemia, complexity, clustering, effort, dx)
    store.save_run("run-2", "/repo-a", 90, summary, hotspot, knowledge,
                    coupling, anemia, complexity, clustering, effort, dx)
    store.save_run("run-3", "/repo-b", 90, summary, hotspot, knowledge,
                    coupling, anemia, complexity, clustering, effort, dx)
    store.close()
    with TestClient(app) as c:
        yield c


class TestListRepos:
    def test_returns_repos(self, client):
        resp = client.get("/api/repos")
        assert resp.status_code == 200
        data = resp.json()
        assert "/repo-a" in data
        assert "/repo-b" in data

    def test_sorted_alphabetically(self, client):
        resp = client.get("/api/repos")
        data = resp.json()
        assert data == sorted(data)


class TestListRuns:
    def test_filters_by_repo(self, client):
        resp = client.get("/api/runs", params={"repo": "/repo-a"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert all(r["repo_path"] == "/repo-a" for r in data)

    def test_empty_for_unknown_repo(self, client):
        resp = client.get("/api/runs", params={"repo": "/nonexistent"})
        assert resp.status_code == 200
        assert resp.json() == []

    def test_requires_repo_param(self, client):
        resp = client.get("/api/runs")
        assert resp.status_code == 422


class TestGetRun:
    def test_returns_run_detail(self, client):
        resp = client.get("/api/runs/run-1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["run_id"] == "run-1"
        assert data["dx_score"] == 0.72

    def test_404_for_missing_run(self, client):
        resp = client.get("/api/runs/nonexistent")
        assert resp.status_code == 404

    def test_json_fields_parsed(self, client):
        resp = client.get("/api/runs/run-1")
        data = resp.json()
        assert isinstance(data["effort_coefficients"], list)
        assert isinstance(data["dx_weights"], list)


class TestHotspots:
    def test_returns_hotspot_files(self, client):
        resp = client.get("/api/runs/run-1/hotspots")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert data[0]["file_path"] == "src/a.py"

    def test_404_for_missing_run(self, client):
        resp = client.get("/api/runs/nonexistent/hotspots")
        assert resp.status_code == 404


class TestKnowledge:
    def test_returns_knowledge_files(self, client):
        resp = client.get("/api/runs/run-1/knowledge")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["primary_author"] == "Alice"


class TestCoupling:
    def test_returns_coupling_pairs(self, client):
        resp = client.get("/api/runs/run-1/coupling")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1


class TestPain:
    def test_returns_pain_files(self, client):
        resp = client.get("/api/runs/run-1/pain")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1


class TestAnemia:
    def test_returns_anemia_classes(self, client):
        resp = client.get("/api/runs/run-1/anemia")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["class_name"] == "UserDTO"


class TestComplexity:
    def test_returns_complexity_functions(self, client):
        resp = client.get("/api/runs/run-1/complexity")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2


class TestClusters:
    def test_returns_cluster_summaries(self, client):
        resp = client.get("/api/runs/run-1/clusters")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2


class TestDrift:
    def test_returns_drift_rows(self, client):
        resp = client.get("/api/runs/run-1/drift")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2


class TestEffort:
    def test_returns_effort_files(self, client):
        resp = client.get("/api/runs/run-1/effort")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1


class TestCognitive:
    def test_returns_cognitive_files(self, client):
        resp = client.get("/api/runs/run-1/cognitive")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["file_path"] == "src/a.py"


class TestCompare:
    def test_compare_two_runs(self, client):
        resp = client.get("/api/compare", params={"a": "run-1", "b": "run-2"})
        assert resp.status_code == 200
        data = resp.json()
        assert "run_a" in data
        assert "run_b" in data
        assert "deltas" in data
        assert "dx_score" in data["deltas"]

    def test_404_for_missing_first_run(self, client):
        resp = client.get("/api/compare", params={"a": "nonexistent", "b": "run-1"})
        assert resp.status_code == 404

    def test_404_for_missing_second_run(self, client):
        resp = client.get("/api/compare", params={"a": "run-1", "b": "nonexistent"})
        assert resp.status_code == 404

    def test_deltas_are_zero_for_same_data(self, client):
        resp = client.get("/api/compare", params={"a": "run-1", "b": "run-2"})
        data = resp.json()
        assert data["deltas"]["dx_score"] == 0.0
