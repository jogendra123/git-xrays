import json
import threading
from datetime import datetime, timezone

import duckdb
import pytest

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


# ── Helpers ──────────────────────────────────────────────────────────

_NOW = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
_SINCE = datetime(2024, 10, 17, 12, 0, 0, tzinfo=timezone.utc)


def _make_summary() -> RepoSummary:
    return RepoSummary(
        repo_path="/repo",
        commit_count=42,
        first_commit_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
        last_commit_date=_NOW,
    )


def _make_hotspot() -> HotspotReport:
    return HotspotReport(
        repo_path="/repo", window_days=90,
        from_date=_SINCE, to_date=_NOW, total_commits=10,
        files=[
            FileMetrics("src/a.py", 5, 100, 1.0, 0.8),
            FileMetrics("src/b.py", 3, 60, 0.36, 0.6667),
        ],
    )


def _make_knowledge() -> KnowledgeReport:
    return KnowledgeReport(
        repo_path="/repo", window_days=90,
        from_date=_SINCE, to_date=_NOW, total_commits=10,
        developer_risk_index=2,
        knowledge_island_count=1,
        files=[
            FileKnowledge("src/a.py", 0.85, "Alice", 0.9, True, 2, []),
        ],
    )


def _make_coupling() -> CouplingReport:
    return CouplingReport(
        repo_path="/repo", window_days=90,
        from_date=_SINCE, to_date=_NOW, total_commits=10,
        coupling_pairs=[
            CouplingPair("src/a.py", "src/b.py", 3, 10, 0.75, 0.3),
        ],
        file_pain=[
            FilePain("src/a.py", 100, 1.0, 5, 1.0, 0.75, 1.0, 1.0),
            FilePain("src/b.py", 60, 0.6, 3, 0.6, 0.75, 1.0, 0.36),
        ],
    )


def _make_anemia() -> AnemicReport:
    return AnemicReport(
        repo_path="/repo", ref=None,
        total_files=1, total_classes=1,
        anemic_count=1, anemic_percentage=100.0,
        average_ams=0.75, ams_threshold=0.5,
        files=[
            FileAnemic("src/a.py", 1, 1, 0.75, [
                ClassMetrics("UserDTO", "src/a.py", 3, 2, 0, 0, 0,
                             1.0, 0.0, 1.0, 0.75),
            ], 2),
        ],
    )


def _make_complexity() -> ComplexityReport:
    return ComplexityReport(
        repo_path="/repo", ref=None,
        total_files=1, total_functions=2,
        avg_complexity=3.5, max_complexity=5,
        high_complexity_count=0, complexity_threshold=10,
        avg_length=8.0, max_length=12,
        files=[
            FileComplexity("src/a.py", 2, 7, 3.5, 5, "process", 8.0, 12, 2.0, 3, [
                FunctionComplexity("process", "src/a.py", None, 1, 12, 5, 3, 2, 1),
                FunctionComplexity("helper", "src/a.py", None, 14, 4, 2, 1, 1, 0),
            ]),
        ],
    )


def _make_clustering() -> ClusteringReport:
    return ClusteringReport(
        repo_path="/repo", window_days=90,
        from_date=_SINCE, to_date=_NOW, total_commits=10,
        k=2, silhouette_score=0.65,
        clusters=[
            ClusterSummary(0, "feature", 6, 3.0, 50.0, 0.8, [
                CommitFeatures("abc", _NOW, 3, 50, 0.8),
            ]),
            ClusterSummary(1, "bugfix", 4, 1.0, 10.0, 0.5, [
                CommitFeatures("def", _NOW, 1, 10, 0.5),
            ]),
        ],
        drift=[
            ClusterDrift("feature", 55.0, 50.0, -5.0, "shrinking"),
            ClusterDrift("bugfix", 45.0, 50.0, 5.0, "growing"),
        ],
    )


def _make_effort() -> EffortReport:
    return EffortReport(
        repo_path="/repo", window_days=90,
        from_date=_SINCE, to_date=_NOW,
        total_files=2, model_r_squared=0.85, alpha=1.0,
        feature_names=["code_churn", "change_frequency", "pain_score",
                        "knowledge_concentration", "author_count"],
        coefficients=[0.3, 0.25, 0.2, 0.15, 0.1],
        files=[
            FileEffort("src/a.py", 0.9, 0.8, 0.7, 0.6, [
                FeatureAttribution("code_churn", 100.0, 0.3, 0.27),
            ]),
            FileEffort("src/b.py", 0.5, 0.4, 0.3, 0.2, [
                FeatureAttribution("code_churn", 60.0, 0.3, 0.15),
            ]),
        ],
    )


def _make_dx() -> DXReport:
    return DXReport(
        repo_path="/repo", window_days=90,
        from_date=_SINCE, to_date=_NOW,
        total_commits=10, total_files=2,
        dx_score=0.72,
        weights=[0.3, 0.25, 0.25, 0.2],
        metrics=DXMetrics(0.8, 0.7, 0.65, 0.35),
        cognitive_load_files=[
            FileCognitiveLoad("src/a.py", 0.6, 0.5, 0.4, 0.3, 0.45),
        ],
    )


def _make_all_reports():
    return (
        _make_summary(),
        _make_hotspot(),
        _make_knowledge(),
        _make_coupling(),
        _make_anemia(),
        _make_complexity(),
        _make_clustering(),
        _make_effort(),
        _make_dx(),
    )


# ── TestRunStoreInit ──────────────────────────────────────────────────


class TestRunStoreInit:
    def test_custom_path_creates_db(self, tmp_path):
        db_file = str(tmp_path / "custom.db")
        store = RunStore(db_path=db_file)
        store.close()
        assert (tmp_path / "custom.db").exists()

    def test_creates_parent_directory(self, tmp_path):
        nested = tmp_path / "sub" / "dir" / "runs.db"
        store = RunStore(db_path=str(nested))
        store.close()
        assert nested.exists()

    def test_tables_exist_after_init(self, tmp_path):
        db_file = str(tmp_path / "test.db")
        store = RunStore(db_path=db_file)
        tables = [r[0] for r in store._conn.execute("SHOW TABLES").fetchall()]
        store.close()
        assert "runs" in tables
        assert "hotspot_files" in tables
        assert "knowledge_files" in tables
        assert "coupling_pairs" in tables
        assert "file_pain" in tables
        assert "anemia_classes" in tables
        assert "complexity_functions" in tables
        assert "cluster_summaries" in tables
        assert "cluster_drift" in tables
        assert "effort_files" in tables
        assert "dx_cognitive_files" in tables

    def test_idempotent_init(self, tmp_path):
        db_file = str(tmp_path / "test.db")
        store1 = RunStore(db_path=db_file)
        store1.close()
        store2 = RunStore(db_path=db_file)
        store2.close()
        # No error means idempotent

    def test_eleven_tables_created(self, tmp_path):
        db_file = str(tmp_path / "test.db")
        store = RunStore(db_path=db_file)
        tables = store._conn.execute("SHOW TABLES").fetchall()
        store.close()
        assert len(tables) == 11


# ── TestRunStoreSaveRun ───────────────────────────────────────────────


class TestRunStoreSaveRun:
    def test_inserts_runs_row(self, tmp_path):
        db_file = str(tmp_path / "test.db")
        store = RunStore(db_path=db_file)
        summary, hotspot, knowledge, coupling, anemia, complexity, clustering, effort, dx = _make_all_reports()
        store.save_run("run-1", "/repo", 90, summary, hotspot, knowledge,
                        coupling, anemia, complexity, clustering, effort, dx)
        rows = store._conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        store.close()
        assert rows == 1

    def test_correct_run_id_stored(self, tmp_path):
        db_file = str(tmp_path / "test.db")
        store = RunStore(db_path=db_file)
        summary, hotspot, knowledge, coupling, anemia, complexity, clustering, effort, dx = _make_all_reports()
        store.save_run("my-uuid", "/repo", 90, summary, hotspot, knowledge,
                        coupling, anemia, complexity, clustering, effort, dx)
        rid = store._conn.execute("SELECT run_id FROM runs").fetchone()[0]
        store.close()
        assert rid == "my-uuid"

    def test_hotspot_files_populated(self, tmp_path):
        db_file = str(tmp_path / "test.db")
        store = RunStore(db_path=db_file)
        summary, hotspot, knowledge, coupling, anemia, complexity, clustering, effort, dx = _make_all_reports()
        store.save_run("run-1", "/repo", 90, summary, hotspot, knowledge,
                        coupling, anemia, complexity, clustering, effort, dx)
        count = store._conn.execute("SELECT COUNT(*) FROM hotspot_files").fetchone()[0]
        store.close()
        assert count == 2

    def test_knowledge_files_populated(self, tmp_path):
        db_file = str(tmp_path / "test.db")
        store = RunStore(db_path=db_file)
        summary, hotspot, knowledge, coupling, anemia, complexity, clustering, effort, dx = _make_all_reports()
        store.save_run("run-1", "/repo", 90, summary, hotspot, knowledge,
                        coupling, anemia, complexity, clustering, effort, dx)
        count = store._conn.execute("SELECT COUNT(*) FROM knowledge_files").fetchone()[0]
        store.close()
        assert count == 1

    def test_coupling_pairs_populated(self, tmp_path):
        db_file = str(tmp_path / "test.db")
        store = RunStore(db_path=db_file)
        summary, hotspot, knowledge, coupling, anemia, complexity, clustering, effort, dx = _make_all_reports()
        store.save_run("run-1", "/repo", 90, summary, hotspot, knowledge,
                        coupling, anemia, complexity, clustering, effort, dx)
        count = store._conn.execute("SELECT COUNT(*) FROM coupling_pairs").fetchone()[0]
        store.close()
        assert count == 1

    def test_file_pain_populated(self, tmp_path):
        db_file = str(tmp_path / "test.db")
        store = RunStore(db_path=db_file)
        summary, hotspot, knowledge, coupling, anemia, complexity, clustering, effort, dx = _make_all_reports()
        store.save_run("run-1", "/repo", 90, summary, hotspot, knowledge,
                        coupling, anemia, complexity, clustering, effort, dx)
        count = store._conn.execute("SELECT COUNT(*) FROM file_pain").fetchone()[0]
        store.close()
        assert count == 2

    def test_anemia_classes_populated(self, tmp_path):
        db_file = str(tmp_path / "test.db")
        store = RunStore(db_path=db_file)
        summary, hotspot, knowledge, coupling, anemia, complexity, clustering, effort, dx = _make_all_reports()
        store.save_run("run-1", "/repo", 90, summary, hotspot, knowledge,
                        coupling, anemia, complexity, clustering, effort, dx)
        count = store._conn.execute("SELECT COUNT(*) FROM anemia_classes").fetchone()[0]
        store.close()
        assert count == 1

    def test_complexity_functions_populated(self, tmp_path):
        db_file = str(tmp_path / "test.db")
        store = RunStore(db_path=db_file)
        summary, hotspot, knowledge, coupling, anemia, complexity, clustering, effort, dx = _make_all_reports()
        store.save_run("run-1", "/repo", 90, summary, hotspot, knowledge,
                        coupling, anemia, complexity, clustering, effort, dx)
        count = store._conn.execute("SELECT COUNT(*) FROM complexity_functions").fetchone()[0]
        store.close()
        assert count == 2

    def test_cluster_summaries_populated(self, tmp_path):
        db_file = str(tmp_path / "test.db")
        store = RunStore(db_path=db_file)
        summary, hotspot, knowledge, coupling, anemia, complexity, clustering, effort, dx = _make_all_reports()
        store.save_run("run-1", "/repo", 90, summary, hotspot, knowledge,
                        coupling, anemia, complexity, clustering, effort, dx)
        count = store._conn.execute("SELECT COUNT(*) FROM cluster_summaries").fetchone()[0]
        store.close()
        assert count == 2

    def test_cluster_drift_populated(self, tmp_path):
        db_file = str(tmp_path / "test.db")
        store = RunStore(db_path=db_file)
        summary, hotspot, knowledge, coupling, anemia, complexity, clustering, effort, dx = _make_all_reports()
        store.save_run("run-1", "/repo", 90, summary, hotspot, knowledge,
                        coupling, anemia, complexity, clustering, effort, dx)
        count = store._conn.execute("SELECT COUNT(*) FROM cluster_drift").fetchone()[0]
        store.close()
        assert count == 2

    def test_effort_files_populated(self, tmp_path):
        db_file = str(tmp_path / "test.db")
        store = RunStore(db_path=db_file)
        summary, hotspot, knowledge, coupling, anemia, complexity, clustering, effort, dx = _make_all_reports()
        store.save_run("run-1", "/repo", 90, summary, hotspot, knowledge,
                        coupling, anemia, complexity, clustering, effort, dx)
        count = store._conn.execute("SELECT COUNT(*) FROM effort_files").fetchone()[0]
        store.close()
        assert count == 2

    def test_dx_cognitive_files_populated(self, tmp_path):
        db_file = str(tmp_path / "test.db")
        store = RunStore(db_path=db_file)
        summary, hotspot, knowledge, coupling, anemia, complexity, clustering, effort, dx = _make_all_reports()
        store.save_run("run-1", "/repo", 90, summary, hotspot, knowledge,
                        coupling, anemia, complexity, clustering, effort, dx)
        count = store._conn.execute("SELECT COUNT(*) FROM dx_cognitive_files").fetchone()[0]
        store.close()
        assert count == 1


# ── TestRunStoreSaveRunScalars ────────────────────────────────────────


class TestRunStoreSaveRunScalars:
    @pytest.fixture
    def stored_run(self, tmp_path):
        db_file = str(tmp_path / "test.db")
        store = RunStore(db_path=db_file)
        summary, hotspot, knowledge, coupling, anemia, complexity, clustering, effort, dx = _make_all_reports()
        store.save_run("run-1", "/repo", 90, summary, hotspot, knowledge,
                        coupling, anemia, complexity, clustering, effort, dx)
        row = store._conn.execute("SELECT * FROM runs WHERE run_id = 'run-1'").fetchone()
        cols = [desc[0] for desc in store._conn.description]
        store.close()
        return dict(zip(cols, row))

    def test_dx_score(self, stored_run):
        assert stored_run["dx_score"] == 0.72

    def test_dx_throughput(self, stored_run):
        assert stored_run["dx_throughput"] == 0.8

    def test_knowledge_scalars(self, stored_run):
        assert stored_run["developer_risk_index"] == 2
        assert stored_run["knowledge_island_count"] == 1

    def test_anemia_scalars(self, stored_run):
        assert stored_run["anemia_total_classes"] == 1
        assert stored_run["anemia_anemic_count"] == 1
        assert stored_run["anemia_anemic_pct"] == 100.0
        assert stored_run["anemia_average_ams"] == 0.75

    def test_complexity_scalars(self, stored_run):
        assert stored_run["complexity_total_functions"] == 2
        assert stored_run["complexity_avg"] == 3.5
        assert stored_run["complexity_max"] == 5
        assert stored_run["complexity_high_count"] == 0

    def test_clustering_scalars(self, stored_run):
        assert stored_run["clustering_k"] == 2
        assert stored_run["clustering_silhouette"] == 0.65

    def test_effort_scalars(self, stored_run):
        assert stored_run["effort_total_files"] == 2
        assert stored_run["effort_model_r_squared"] == 0.85
        coefficients = json.loads(stored_run["effort_coefficients"])
        assert coefficients == [0.3, 0.25, 0.2, 0.15, 0.1]

    def test_dx_weights_json(self, stored_run):
        weights = json.loads(stored_run["dx_weights"])
        assert weights == [0.3, 0.25, 0.25, 0.2]


# ── TestRunStoreListRuns ──────────────────────────────────────────────


class TestRunStoreListRuns:
    def test_empty_db_returns_empty(self, tmp_path):
        store = RunStore(db_path=str(tmp_path / "test.db"))
        result = store.list_runs()
        store.close()
        assert result == []

    def test_returns_saved_run(self, tmp_path):
        store = RunStore(db_path=str(tmp_path / "test.db"))
        summary, hotspot, knowledge, coupling, anemia, complexity, clustering, effort, dx = _make_all_reports()
        store.save_run("run-1", "/repo", 90, summary, hotspot, knowledge,
                        coupling, anemia, complexity, clustering, effort, dx)
        result = store.list_runs()
        store.close()
        assert len(result) == 1
        assert result[0]["run_id"] == "run-1"

    def test_ordered_desc_by_created_at(self, tmp_path):
        store = RunStore(db_path=str(tmp_path / "test.db"))
        summary, hotspot, knowledge, coupling, anemia, complexity, clustering, effort, dx = _make_all_reports()
        store.save_run("run-1", "/repo", 90, summary, hotspot, knowledge,
                        coupling, anemia, complexity, clustering, effort, dx)
        store.save_run("run-2", "/repo", 90, summary, hotspot, knowledge,
                        coupling, anemia, complexity, clustering, effort, dx)
        result = store.list_runs()
        store.close()
        # run-2 saved later, should appear first
        assert result[0]["run_id"] == "run-2"
        assert result[1]["run_id"] == "run-1"

    def test_correct_fields_returned(self, tmp_path):
        store = RunStore(db_path=str(tmp_path / "test.db"))
        summary, hotspot, knowledge, coupling, anemia, complexity, clustering, effort, dx = _make_all_reports()
        store.save_run("run-1", "/repo", 90, summary, hotspot, knowledge,
                        coupling, anemia, complexity, clustering, effort, dx)
        result = store.list_runs()
        store.close()
        row = result[0]
        expected_keys = {"run_id", "repo_path", "created_at", "window_days",
                          "total_commits", "hotspot_file_count", "dx_score"}
        assert set(row.keys()) == expected_keys

    def test_multiple_repos(self, tmp_path):
        store = RunStore(db_path=str(tmp_path / "test.db"))
        summary, hotspot, knowledge, coupling, anemia, complexity, clustering, effort, dx = _make_all_reports()
        store.save_run("run-1", "/repo-a", 90, summary, hotspot, knowledge,
                        coupling, anemia, complexity, clustering, effort, dx)
        store.save_run("run-2", "/repo-b", 90, summary, hotspot, knowledge,
                        coupling, anemia, complexity, clustering, effort, dx)
        result = store.list_runs()
        store.close()
        repo_paths = {r["repo_path"] for r in result}
        assert repo_paths == {"/repo-a", "/repo-b"}


# ── TestRunStoreConcurrency ───────────────────────────────────────────


class TestRunStoreConcurrency:
    def test_duplicate_run_id_rejected(self, tmp_path):
        store = RunStore(db_path=str(tmp_path / "test.db"))
        summary, hotspot, knowledge, coupling, anemia, complexity, clustering, effort, dx = _make_all_reports()
        store.save_run("dup-id", "/repo", 90, summary, hotspot, knowledge,
                        coupling, anemia, complexity, clustering, effort, dx)
        with pytest.raises(Exception):
            store.save_run("dup-id", "/repo", 90, summary, hotspot, knowledge,
                            coupling, anemia, complexity, clustering, effort, dx)
        store.close()

    def test_rollback_on_failure_keeps_db_clean(self, tmp_path):
        store = RunStore(db_path=str(tmp_path / "test.db"))
        summary, hotspot, knowledge, coupling, anemia, complexity, clustering, effort, dx = _make_all_reports()
        store.save_run("good-run", "/repo", 90, summary, hotspot, knowledge,
                        coupling, anemia, complexity, clustering, effort, dx)
        # Try a duplicate that should fail
        with pytest.raises(Exception):
            store.save_run("good-run", "/repo", 90, summary, hotspot, knowledge,
                            coupling, anemia, complexity, clustering, effort, dx)
        # good-run should still be intact
        count = store._conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        store.close()
        assert count == 1

    def test_concurrent_writes_sequential_stores(self, tmp_path):
        """Multiple sequential RunStore instances can each write to the same DB."""
        db_file = str(tmp_path / "test.db")
        summary, hotspot, knowledge, coupling, anemia, complexity, clustering, effort, dx = _make_all_reports()
        for i in range(4):
            store = RunStore(db_path=db_file)
            store.save_run(f"run-{i}", "/repo", 90, summary, hotspot,
                            knowledge, coupling, anemia, complexity,
                            clustering, effort, dx)
            store.close()

        store = RunStore(db_path=db_file)
        runs = store.list_runs()
        store.close()
        assert len(runs) == 4

    def test_datetime_precision_preserved(self, tmp_path):
        store = RunStore(db_path=str(tmp_path / "test.db"))
        summary, hotspot, knowledge, coupling, anemia, complexity, clustering, effort, dx = _make_all_reports()
        store.save_run("run-1", "/repo", 90, summary, hotspot, knowledge,
                        coupling, anemia, complexity, clustering, effort, dx)
        row = store._conn.execute("SELECT from_date, to_date FROM runs").fetchone()
        store.close()
        # DuckDB timestamps are returned as datetime objects
        assert row[0].year == _SINCE.year
        assert row[0].month == _SINCE.month
        assert row[1].year == _NOW.year

    def test_json_coefficients_roundtrip(self, tmp_path):
        store = RunStore(db_path=str(tmp_path / "test.db"))
        summary, hotspot, knowledge, coupling, anemia, complexity, clustering, effort, dx = _make_all_reports()
        store.save_run("run-1", "/repo", 90, summary, hotspot, knowledge,
                        coupling, anemia, complexity, clustering, effort, dx)
        row = store._conn.execute("SELECT effort_coefficients, dx_weights FROM runs").fetchone()
        store.close()
        assert json.loads(row[0]) == [0.3, 0.25, 0.2, 0.15, 0.1]
        assert json.loads(row[1]) == [0.3, 0.25, 0.25, 0.2]


# ── TestRunStoreGetRun ──────────────────────────────────────────────


class TestRunStoreGetRun:
    def test_returns_none_for_missing_id(self, tmp_path):
        store = RunStore(db_path=str(tmp_path / "test.db"))
        result = store.get_run("nonexistent")
        store.close()
        assert result is None

    def test_returns_dict_for_existing_run(self, tmp_path):
        store = RunStore(db_path=str(tmp_path / "test.db"))
        summary, hotspot, knowledge, coupling, anemia, complexity, clustering, effort, dx = _make_all_reports()
        store.save_run("run-1", "/repo", 90, summary, hotspot, knowledge,
                        coupling, anemia, complexity, clustering, effort, dx)
        result = store.get_run("run-1")
        store.close()
        assert result is not None
        assert result["run_id"] == "run-1"

    def test_contains_all_columns(self, tmp_path):
        store = RunStore(db_path=str(tmp_path / "test.db"))
        summary, hotspot, knowledge, coupling, anemia, complexity, clustering, effort, dx = _make_all_reports()
        store.save_run("run-1", "/repo", 90, summary, hotspot, knowledge,
                        coupling, anemia, complexity, clustering, effort, dx)
        result = store.get_run("run-1")
        store.close()
        assert "dx_score" in result
        assert "effort_coefficients" in result
        assert "repo_path" in result

    def test_dx_score_value(self, tmp_path):
        store = RunStore(db_path=str(tmp_path / "test.db"))
        summary, hotspot, knowledge, coupling, anemia, complexity, clustering, effort, dx = _make_all_reports()
        store.save_run("run-1", "/repo", 90, summary, hotspot, knowledge,
                        coupling, anemia, complexity, clustering, effort, dx)
        result = store.get_run("run-1")
        store.close()
        assert result["dx_score"] == 0.72


# ── TestRunStoreListRepos ───────────────────────────────────────────


class TestRunStoreListRepos:
    def test_empty_db_returns_empty(self, tmp_path):
        store = RunStore(db_path=str(tmp_path / "test.db"))
        result = store.list_repos()
        store.close()
        assert result == []

    def test_returns_distinct_repos(self, tmp_path):
        store = RunStore(db_path=str(tmp_path / "test.db"))
        summary, hotspot, knowledge, coupling, anemia, complexity, clustering, effort, dx = _make_all_reports()
        store.save_run("run-1", "/repo-a", 90, summary, hotspot, knowledge,
                        coupling, anemia, complexity, clustering, effort, dx)
        store.save_run("run-2", "/repo-b", 90, summary, hotspot, knowledge,
                        coupling, anemia, complexity, clustering, effort, dx)
        store.save_run("run-3", "/repo-a", 90, summary, hotspot, knowledge,
                        coupling, anemia, complexity, clustering, effort, dx)
        result = store.list_repos()
        store.close()
        assert sorted(result) == ["/repo-a", "/repo-b"]

    def test_ordered_alphabetically(self, tmp_path):
        store = RunStore(db_path=str(tmp_path / "test.db"))
        summary, hotspot, knowledge, coupling, anemia, complexity, clustering, effort, dx = _make_all_reports()
        store.save_run("run-1", "/zebra", 90, summary, hotspot, knowledge,
                        coupling, anemia, complexity, clustering, effort, dx)
        store.save_run("run-2", "/alpha", 90, summary, hotspot, knowledge,
                        coupling, anemia, complexity, clustering, effort, dx)
        result = store.list_repos()
        store.close()
        assert result == ["/alpha", "/zebra"]


# ── TestRunStoreListRunsForRepo ─────────────────────────────────────


class TestRunStoreListRunsForRepo:
    def test_empty_db_returns_empty(self, tmp_path):
        store = RunStore(db_path=str(tmp_path / "test.db"))
        result = store.list_runs_for_repo("/nonexistent")
        store.close()
        assert result == []

    def test_filters_by_repo_path(self, tmp_path):
        store = RunStore(db_path=str(tmp_path / "test.db"))
        summary, hotspot, knowledge, coupling, anemia, complexity, clustering, effort, dx = _make_all_reports()
        store.save_run("run-1", "/repo-a", 90, summary, hotspot, knowledge,
                        coupling, anemia, complexity, clustering, effort, dx)
        store.save_run("run-2", "/repo-b", 90, summary, hotspot, knowledge,
                        coupling, anemia, complexity, clustering, effort, dx)
        result = store.list_runs_for_repo("/repo-a")
        store.close()
        assert len(result) == 1
        assert result[0]["run_id"] == "run-1"

    def test_ordered_desc_by_created_at(self, tmp_path):
        store = RunStore(db_path=str(tmp_path / "test.db"))
        summary, hotspot, knowledge, coupling, anemia, complexity, clustering, effort, dx = _make_all_reports()
        store.save_run("run-1", "/repo", 90, summary, hotspot, knowledge,
                        coupling, anemia, complexity, clustering, effort, dx)
        store.save_run("run-2", "/repo", 90, summary, hotspot, knowledge,
                        coupling, anemia, complexity, clustering, effort, dx)
        result = store.list_runs_for_repo("/repo")
        store.close()
        assert result[0]["run_id"] == "run-2"

    def test_returns_key_columns(self, tmp_path):
        store = RunStore(db_path=str(tmp_path / "test.db"))
        summary, hotspot, knowledge, coupling, anemia, complexity, clustering, effort, dx = _make_all_reports()
        store.save_run("run-1", "/repo", 90, summary, hotspot, knowledge,
                        coupling, anemia, complexity, clustering, effort, dx)
        result = store.list_runs_for_repo("/repo")
        store.close()
        row = result[0]
        assert "run_id" in row
        assert "created_at" in row
        assert "dx_score" in row
        assert "total_commits" in row


# ── TestRunStoreChildGetters ────────────────────────────────────────


class TestRunStoreChildGetters:
    @pytest.fixture
    def store_with_run(self, tmp_path):
        db_file = str(tmp_path / "test.db")
        store = RunStore(db_path=db_file)
        summary, hotspot, knowledge, coupling, anemia, complexity, clustering, effort, dx = _make_all_reports()
        store.save_run("run-1", "/repo", 90, summary, hotspot, knowledge,
                        coupling, anemia, complexity, clustering, effort, dx)
        yield store
        store.close()

    def test_get_hotspot_files_returns_correct_count(self, store_with_run):
        result = store_with_run.get_hotspot_files("run-1")
        assert len(result) == 2

    def test_get_hotspot_files_has_expected_keys(self, store_with_run):
        result = store_with_run.get_hotspot_files("run-1")
        assert "file_path" in result[0]
        assert "hotspot_score" in result[0]

    def test_get_hotspot_files_empty_for_missing_run(self, store_with_run):
        result = store_with_run.get_hotspot_files("nonexistent")
        assert result == []

    def test_get_knowledge_files(self, store_with_run):
        result = store_with_run.get_knowledge_files("run-1")
        assert len(result) == 1
        assert result[0]["file_path"] == "src/a.py"

    def test_get_coupling_pairs(self, store_with_run):
        result = store_with_run.get_coupling_pairs("run-1")
        assert len(result) == 1
        assert result[0]["file_a"] == "src/a.py"

    def test_get_file_pain(self, store_with_run):
        result = store_with_run.get_file_pain("run-1")
        assert len(result) == 2

    def test_get_anemia_classes(self, store_with_run):
        result = store_with_run.get_anemia_classes("run-1")
        assert len(result) == 1
        assert result[0]["class_name"] == "UserDTO"

    def test_get_complexity_functions(self, store_with_run):
        result = store_with_run.get_complexity_functions("run-1")
        assert len(result) == 2

    def test_get_cluster_summaries(self, store_with_run):
        result = store_with_run.get_cluster_summaries("run-1")
        assert len(result) == 2

    def test_get_cluster_drift(self, store_with_run):
        result = store_with_run.get_cluster_drift("run-1")
        assert len(result) == 2

    def test_get_effort_files(self, store_with_run):
        result = store_with_run.get_effort_files("run-1")
        assert len(result) == 2

    def test_get_dx_cognitive_files(self, store_with_run):
        result = store_with_run.get_dx_cognitive_files("run-1")
        assert len(result) == 1
        assert result[0]["file_path"] == "src/a.py"
