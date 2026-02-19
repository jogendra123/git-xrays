import argparse
import sys

import pytest

from git_xrays.interface.cli import _parse_window, main


class TestParseWindow:
    def test_90d(self):
        assert _parse_window("90d") == 90

    def test_30d(self):
        assert _parse_window("30d") == 30

    def test_1d(self):
        assert _parse_window("1d") == 1

    def test_invalid_no_suffix(self):
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_window("90")

    def test_invalid_string(self):
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_window("abc")

    def test_invalid_empty(self):
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_window("")


class TestMainCli:
    def test_valid_repo_prints_summary(self, git_repo_with_history, capsys, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["analyze-repo", str(git_repo_with_history)])
        main()
        captured = capsys.readouterr()
        assert "Repository:" in captured.out
        assert "Commits:" in captured.out
        assert "Hotspot Analysis" in captured.out

    def test_invalid_path_exits_with_error(self, tmp_path, capsys, monkeypatch):
        bad_path = str(tmp_path / "nonexistent")
        monkeypatch.setattr(sys, "argv", ["analyze-repo", bad_path])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Error" in captured.err

    def test_window_flag_parsed(self, git_repo_with_history, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(git_repo_with_history), "--window", "30d"],
        )
        main()
        captured = capsys.readouterr()
        assert "last 30 days" in captured.out

    def test_empty_repo_prints_no_commits(self, tmp_git_repo, capsys, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["analyze-repo", str(tmp_git_repo)])
        main()
        captured = capsys.readouterr()
        assert "No commits found." in captured.out


class TestKnowledgeCli:
    def test_knowledge_flag_prints_section(self, multi_author_repo, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(multi_author_repo), "--knowledge"],
        )
        main()
        captured = capsys.readouterr()
        assert "Knowledge Analysis" in captured.out

    def test_knowledge_shows_island_count(self, multi_author_repo, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(multi_author_repo), "--knowledge"],
        )
        main()
        captured = capsys.readouterr()
        assert "Knowledge Islands:" in captured.out

    def test_knowledge_shows_dri(self, multi_author_repo, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(multi_author_repo), "--knowledge"],
        )
        main()
        captured = capsys.readouterr()
        assert "Developer Risk Index:" in captured.out

    def test_knowledge_shows_concentration_column(self, multi_author_repo, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(multi_author_repo), "--knowledge"],
        )
        main()
        captured = capsys.readouterr()
        assert "Concentration" in captured.out

    def test_no_knowledge_flag_no_section(self, multi_author_repo, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(multi_author_repo)],
        )
        main()
        captured = capsys.readouterr()
        assert "Knowledge Analysis" not in captured.out

    def test_knowledge_with_window(self, multi_author_repo, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(multi_author_repo), "--knowledge", "--window", "30d"],
        )
        main()
        captured = capsys.readouterr()
        assert "Knowledge Analysis" in captured.out
        assert "last 30 days" in captured.out


class TestCouplingCli:
    def test_coupling_flag_prints_section(self, coupled_repo, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(coupled_repo), "--coupling"],
        )
        main()
        captured = capsys.readouterr()
        assert "Coupling Analysis" in captured.out

    def test_coupling_shows_strength_column(self, coupled_repo, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(coupled_repo), "--coupling"],
        )
        main()
        captured = capsys.readouterr()
        assert "Strength" in captured.out

    def test_coupling_shows_pain_section(self, coupled_repo, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(coupled_repo), "--coupling"],
        )
        main()
        captured = capsys.readouterr()
        assert "PAIN" in captured.out

    def test_coupling_shows_size_volatility_distance(self, coupled_repo, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(coupled_repo), "--coupling"],
        )
        main()
        captured = capsys.readouterr()
        assert "Size" in captured.out
        assert "Volatility" in captured.out
        assert "Distance" in captured.out

    def test_no_coupling_flag_no_section(self, coupled_repo, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(coupled_repo)],
        )
        main()
        captured = capsys.readouterr()
        assert "Coupling Analysis" not in captured.out

    def test_coupling_with_window(self, coupled_repo, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(coupled_repo), "--coupling", "--window", "30d"],
        )
        main()
        captured = capsys.readouterr()
        assert "Coupling Analysis" in captured.out
        assert "last 30 days" in captured.out

    def test_coupling_empty_window_shows_header(self, coupled_repo, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(coupled_repo), "--coupling", "--window", "1d"],
        )
        main()
        captured = capsys.readouterr()
        assert "Coupling Analysis" in captured.out


class TestAtFlag:
    def test_at_tag_prints_snapshot_line(self, tagged_repo, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(tagged_repo), "--at", "v1.0"],
        )
        main()
        captured = capsys.readouterr()
        assert "Snapshot at:" in captured.out

    def test_at_iso_date_works(self, git_repo_with_history, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(git_repo_with_history), "--at", "2024-06-01"],
        )
        main()
        captured = capsys.readouterr()
        assert "Snapshot at:" in captured.out

    def test_at_with_knowledge_flag(self, tagged_repo, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(tagged_repo), "--at", "v1.0", "--knowledge"],
        )
        main()
        captured = capsys.readouterr()
        assert "Knowledge Analysis" in captured.out

    def test_at_with_coupling_flag(self, tagged_repo, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(tagged_repo), "--at", "v1.0", "--coupling"],
        )
        main()
        captured = capsys.readouterr()
        assert "Coupling Analysis" in captured.out


class TestCompareFlags:
    def test_from_to_prints_comparison_header(self, tagged_repo, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(tagged_repo), "--from", "v1.0", "--to", "v2.0"],
        )
        main()
        captured = capsys.readouterr()
        assert "Hotspot Comparison" in captured.out

    def test_from_to_prints_summary_counts(self, tagged_repo, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(tagged_repo), "--from", "v1.0", "--to", "v2.0"],
        )
        main()
        captured = capsys.readouterr()
        assert "new" in captured.out
        assert "degraded" in captured.out

    def test_from_to_prints_delta_table(self, tagged_repo, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(tagged_repo), "--from", "v1.0", "--to", "v2.0"],
        )
        main()
        captured = capsys.readouterr()
        assert "Delta" in captured.out
        assert "Status" in captured.out

    def test_from_without_to_errors(self, tagged_repo, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(tagged_repo), "--from", "v1.0"],
        )
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "--to" in captured.err

    def test_to_without_from_errors(self, tagged_repo, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(tagged_repo), "--to", "v2.0"],
        )
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "--from" in captured.err

    def test_at_with_from_to_errors(self, tagged_repo, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(tagged_repo), "--at", "v1.0", "--from", "v1.0", "--to", "v2.0"],
        )
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "--at" in captured.err

    def test_from_to_with_window(self, tagged_repo, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(tagged_repo), "--from", "v1.0", "--to", "v2.0", "--window", "30d"],
        )
        main()
        captured = capsys.readouterr()
        assert "30-day window" in captured.out


class TestAnemicCli:
    def test_anemic_flag_prints_section(self, anemic_repo, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(anemic_repo), "--anemic"],
        )
        main()
        captured = capsys.readouterr()
        assert "Anemia Analysis" in captured.out

    def test_anemic_shows_total_classes(self, anemic_repo, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(anemic_repo), "--anemic"],
        )
        main()
        captured = capsys.readouterr()
        assert "Total classes:" in captured.out

    def test_anemic_shows_anemic_count(self, anemic_repo, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(anemic_repo), "--anemic"],
        )
        main()
        captured = capsys.readouterr()
        assert "Anemic classes:" in captured.out

    def test_no_anemic_flag_no_section(self, anemic_repo, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(anemic_repo)],
        )
        main()
        captured = capsys.readouterr()
        assert "Anemia Analysis" not in captured.out

    def test_anemic_with_at_flag(self, anemic_repo, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(anemic_repo), "--anemic", "--at", "HEAD"],
        )
        main()
        captured = capsys.readouterr()
        assert "Anemia Analysis" in captured.out


class TestComplexityCli:
    def test_complexity_flag_prints_section(self, complexity_repo, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(complexity_repo), "--complexity"],
        )
        main()
        captured = capsys.readouterr()
        assert "Complexity Analysis" in captured.out

    def test_complexity_shows_total_functions(self, complexity_repo, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(complexity_repo), "--complexity"],
        )
        main()
        captured = capsys.readouterr()
        assert "Total functions:" in captured.out

    def test_complexity_shows_high_complexity_count(self, complexity_repo, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(complexity_repo), "--complexity"],
        )
        main()
        captured = capsys.readouterr()
        assert "High complexity:" in captured.out

    def test_complexity_shows_avg_complexity(self, complexity_repo, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(complexity_repo), "--complexity"],
        )
        main()
        captured = capsys.readouterr()
        assert "Avg complexity:" in captured.out

    def test_no_complexity_flag_no_section(self, complexity_repo, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(complexity_repo)],
        )
        main()
        captured = capsys.readouterr()
        assert "Complexity Analysis" not in captured.out

    def test_complexity_with_at_flag(self, complexity_repo, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(complexity_repo), "--complexity", "--at", "HEAD"],
        )
        main()
        captured = capsys.readouterr()
        assert "Complexity Analysis" in captured.out

    def test_complexity_shows_file_table(self, complexity_repo, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(complexity_repo), "--complexity"],
        )
        main()
        captured = capsys.readouterr()
        assert "Functions" in captured.out
        assert "Max CC" in captured.out


class TestClusteringCli:
    def test_clustering_flag_prints_section(self, clustering_repo, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(clustering_repo), "--clustering"],
        )
        main()
        captured = capsys.readouterr()
        assert "Clustering Analysis" in captured.out

    def test_clustering_shows_k_and_silhouette(self, clustering_repo, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(clustering_repo), "--clustering"],
        )
        main()
        captured = capsys.readouterr()
        assert "Clusters:" in captured.out
        assert "Silhouette:" in captured.out

    def test_clustering_shows_cluster_table(self, clustering_repo, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(clustering_repo), "--clustering"],
        )
        main()
        captured = capsys.readouterr()
        assert "Label" in captured.out
        assert "Size" in captured.out
        assert "Churn" in captured.out

    def test_clustering_shows_drift_section(self, clustering_repo, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(clustering_repo), "--clustering"],
        )
        main()
        captured = capsys.readouterr()
        assert "Cluster Drift" in captured.out

    def test_no_clustering_flag_no_section(self, clustering_repo, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(clustering_repo)],
        )
        main()
        captured = capsys.readouterr()
        assert "Clustering Analysis" not in captured.out

    def test_clustering_with_at_flag(self, clustering_repo, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(clustering_repo), "--clustering", "--at", "HEAD"],
        )
        main()
        captured = capsys.readouterr()
        assert "Clustering Analysis" in captured.out


class TestEffortCli:
    def test_effort_flag_prints_section(self, effort_repo, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(effort_repo), "--effort"],
        )
        main()
        captured = capsys.readouterr()
        assert "Effort Analysis" in captured.out

    def test_effort_shows_r_squared(self, effort_repo, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(effort_repo), "--effort"],
        )
        main()
        captured = capsys.readouterr()
        assert "R²" in captured.out or "R-squared" in captured.out

    def test_effort_shows_rei_column(self, effort_repo, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(effort_repo), "--effort"],
        )
        main()
        captured = capsys.readouterr()
        assert "REI" in captured.out

    def test_effort_shows_top_factor(self, effort_repo, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(effort_repo), "--effort"],
        )
        main()
        captured = capsys.readouterr()
        assert "Top Factor" in captured.out

    def test_no_effort_flag_no_section(self, effort_repo, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(effort_repo)],
        )
        main()
        captured = capsys.readouterr()
        assert "Effort Analysis" not in captured.out

    def test_effort_with_at_flag(self, effort_repo, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(effort_repo), "--effort", "--at", "HEAD"],
        )
        main()
        captured = capsys.readouterr()
        assert "Effort Analysis" in captured.out

    def test_effort_with_window(self, effort_repo, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(effort_repo), "--effort", "--window", "30d"],
        )
        main()
        captured = capsys.readouterr()
        assert "Effort Analysis" in captured.out
        assert "last 30 days" in captured.out


class TestDXCli:
    def test_dx_flag_prints_section(self, dx_repo, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(dx_repo), "--dx"],
        )
        main()
        captured = capsys.readouterr()
        assert "Developer Experience Analysis" in captured.out

    def test_dx_shows_score(self, dx_repo, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(dx_repo), "--dx"],
        )
        main()
        captured = capsys.readouterr()
        assert "DX Score" in captured.out

    def test_dx_shows_metrics(self, dx_repo, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(dx_repo), "--dx"],
        )
        main()
        captured = capsys.readouterr()
        assert "Throughput" in captured.out
        assert "Feedback" in captured.out
        assert "Focus" in captured.out
        assert "Cognitive" in captured.out

    def test_no_dx_flag_no_section(self, dx_repo, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(dx_repo)],
        )
        main()
        captured = capsys.readouterr()
        assert "Developer Experience" not in captured.out

    def test_dx_with_at_flag(self, dx_repo, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(dx_repo), "--dx", "--at", "HEAD"],
        )
        main()
        captured = capsys.readouterr()
        assert "Developer Experience Analysis" in captured.out


class TestListRunsFlag:
    def test_list_runs_no_repo_required(self, tmp_path, capsys, monkeypatch):
        db_file = str(tmp_path / "test.db")
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", "--list-runs", "--db", db_file],
        )
        main()
        captured = capsys.readouterr()
        assert "No runs found." in captured.out

    def test_list_runs_empty_db_message(self, tmp_path, capsys, monkeypatch):
        db_file = str(tmp_path / "test.db")
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", "--list-runs", "--db", db_file],
        )
        main()
        captured = capsys.readouterr()
        assert "No runs found." in captured.out

    def test_list_runs_shows_table_after_all(self, dx_repo, tmp_path_factory, capsys, monkeypatch):
        db_dir = tmp_path_factory.mktemp("db")
        db_file = str(db_dir / "test.db")
        # First do --all to store a run
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(dx_repo), "--all", "--db", db_file],
        )
        main()
        capsys.readouterr()  # discard output

        # Now list runs
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", "--list-runs", "--db", db_file],
        )
        main()
        captured = capsys.readouterr()
        assert "Run ID" in captured.out
        assert "Repository" in captured.out
        assert "DX" in captured.out

    def test_list_runs_with_custom_db(self, tmp_path, capsys, monkeypatch):
        db_file = str(tmp_path / "custom.db")
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", "--list-runs", "--db", db_file],
        )
        main()
        captured = capsys.readouterr()
        assert "No runs found." in captured.out

    def test_list_runs_shows_repo_path(self, dx_repo, tmp_path_factory, capsys, monkeypatch):
        db_dir = tmp_path_factory.mktemp("db")
        db_file = str(db_dir / "test.db")
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(dx_repo), "--all", "--db", db_file],
        )
        main()
        capsys.readouterr()

        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", "--list-runs", "--db", db_file],
        )
        main()
        captured = capsys.readouterr()
        assert str(dx_repo) in captured.out or "..." in captured.out

    def test_no_repo_path_without_list_runs_errors(self, capsys, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["analyze-repo"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "repo_path" in captured.err


class TestAllFlag:
    def test_all_prints_all_sections(self, dx_repo, tmp_path_factory, capsys, monkeypatch):
        db_dir = tmp_path_factory.mktemp("db")
        db_file = str(db_dir / "test.db")
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(dx_repo), "--all", "--db", db_file],
        )
        main()
        captured = capsys.readouterr()
        assert "Hotspot Analysis" in captured.out
        assert "Knowledge Analysis" in captured.out
        assert "Coupling Analysis" in captured.out
        assert "Anemia Analysis" in captured.out
        assert "Complexity Analysis" in captured.out
        assert "Clustering Analysis" in captured.out
        assert "Effort Analysis" in captured.out
        assert "Developer Experience Analysis" in captured.out

    def test_all_prints_run_id(self, dx_repo, tmp_path_factory, capsys, monkeypatch):
        db_dir = tmp_path_factory.mktemp("db")
        db_file = str(db_dir / "test.db")
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(dx_repo), "--all", "--db", db_file],
        )
        main()
        captured = capsys.readouterr()
        assert "Run stored:" in captured.out

    def test_all_creates_db(self, dx_repo, tmp_path_factory, capsys, monkeypatch):
        db_dir = tmp_path_factory.mktemp("db")
        db_file = str(db_dir / "test.db")
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(dx_repo), "--all", "--db", db_file],
        )
        main()
        from pathlib import Path
        assert Path(db_file).exists()

    def test_all_stores_run_in_db(self, dx_repo, tmp_path_factory, capsys, monkeypatch):
        db_dir = tmp_path_factory.mktemp("db")
        db_file = str(db_dir / "test.db")
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(dx_repo), "--all", "--db", db_file],
        )
        main()
        from git_xrays.infrastructure.run_store import RunStore
        store = RunStore(db_path=db_file)
        runs = store.list_runs()
        store.close()
        assert len(runs) == 1

    def test_all_custom_db(self, dx_repo, tmp_path_factory, capsys, monkeypatch):
        db_dir = tmp_path_factory.mktemp("db")
        db_file = str(db_dir / "custom" / "my.db")
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(dx_repo), "--all", "--db", db_file],
        )
        main()
        from pathlib import Path
        assert Path(db_file).exists()

    def test_all_with_at_flag(self, dx_repo, tmp_path_factory, capsys, monkeypatch):
        db_dir = tmp_path_factory.mktemp("db")
        db_file = str(db_dir / "test.db")
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(dx_repo), "--all", "--db", db_file, "--at", "HEAD"],
        )
        main()
        captured = capsys.readouterr()
        assert "Snapshot at:" in captured.out
        assert "Run stored:" in captured.out

    def test_all_incompatible_with_from_to(self, dx_repo, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(dx_repo), "--all", "--from", "HEAD", "--to", "HEAD"],
        )
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "--all" in captured.err

    def test_all_with_window(self, dx_repo, tmp_path_factory, capsys, monkeypatch):
        db_dir = tmp_path_factory.mktemp("db")
        db_file = str(db_dir / "test.db")
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(dx_repo), "--all", "--db", db_file, "--window", "30d"],
        )
        main()
        captured = capsys.readouterr()
        assert "last 30 days" in captured.out
        assert "Run stored:" in captured.out


class TestDbFlag:
    def test_db_without_all_is_noop(self, git_repo_with_history, tmp_path_factory, capsys, monkeypatch):
        db_dir = tmp_path_factory.mktemp("db")
        db_file = str(db_dir / "test.db")
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(git_repo_with_history), "--db", db_file],
        )
        main()
        from pathlib import Path
        # DB should NOT be created when --db is used without --all/--list-runs
        assert not Path(db_file).exists()

    def test_db_with_all(self, dx_repo, tmp_path_factory, capsys, monkeypatch):
        db_dir = tmp_path_factory.mktemp("db")
        db_file = str(db_dir / "test.db")
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(dx_repo), "--all", "--db", db_file],
        )
        main()
        from pathlib import Path
        assert Path(db_file).exists()

class TestGodClassCli:
    def test_god_class_flag_prints_section(self, god_class_repo, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(god_class_repo), "--god-class"],
        )
        main()
        captured = capsys.readouterr()
        assert "God Class Analysis" in captured.out

    def test_god_class_shows_total_classes(self, god_class_repo, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(god_class_repo), "--god-class"],
        )
        main()
        captured = capsys.readouterr()
        assert "Total classes:" in captured.out

    def test_god_class_shows_count(self, god_class_repo, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(god_class_repo), "--god-class"],
        )
        main()
        captured = capsys.readouterr()
        assert "God classes:" in captured.out

    def test_no_god_class_flag_no_section(self, god_class_repo, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(god_class_repo)],
        )
        main()
        captured = capsys.readouterr()
        assert "God Class Analysis" not in captured.out

    def test_god_class_lists_flagged_classes(self, god_class_repo, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(god_class_repo), "--god-class"],
        )
        main()
        captured = capsys.readouterr()
        # GodClass should be listed somewhere (either in file table or individual listing)
        assert "GodClass" in captured.out or "god.py" in captured.out


class TestDbFlagListRuns:
    def test_db_with_list_runs(self, tmp_path, capsys, monkeypatch):
        db_file = str(tmp_path / "test.db")
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", "--list-runs", "--db", db_file],
        )
        main()
        captured = capsys.readouterr()
        assert "No runs found." in captured.out
