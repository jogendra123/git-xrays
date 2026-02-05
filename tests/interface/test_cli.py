import argparse
import sys

import pytest

from git_xrays.interface.cli import _parse_window, main
from tests.conftest import commit_file


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
