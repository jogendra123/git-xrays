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


class TestAnemiaCli:
    def test_anemia_flag_prints_section(self, anemia_repo, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(anemia_repo), "--anemia"],
        )
        main()
        captured = capsys.readouterr()
        assert "Anemia Analysis" in captured.out

    def test_anemia_shows_total_classes(self, anemia_repo, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(anemia_repo), "--anemia"],
        )
        main()
        captured = capsys.readouterr()
        assert "Total classes:" in captured.out

    def test_anemia_shows_anemic_count(self, anemia_repo, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(anemia_repo), "--anemia"],
        )
        main()
        captured = capsys.readouterr()
        assert "Anemic classes:" in captured.out

    def test_no_anemia_flag_no_section(self, anemia_repo, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(anemia_repo)],
        )
        main()
        captured = capsys.readouterr()
        assert "Anemia Analysis" not in captured.out

    def test_anemia_with_at_flag(self, anemia_repo, capsys, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["analyze-repo", str(anemia_repo), "--anemia", "--at", "HEAD"],
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
