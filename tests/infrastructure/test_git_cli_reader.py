import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from git_xrays.application.use_cases import analyze_coupling
from git_xrays.infrastructure.git_cli_reader import GitCliReader
from tests.conftest import commit_file


class TestGitCliReaderInit:
    def test_valid_repo(self, tmp_git_repo: Path):
        reader = GitCliReader(str(tmp_git_repo))
        assert reader is not None

    def test_non_git_dir_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="Not a git repository"):
            GitCliReader(str(tmp_path))


class TestCommitCount:
    def test_empty_repo(self, tmp_git_repo: Path):
        reader = GitCliReader(str(tmp_git_repo))
        assert reader.commit_count() == 0

    def test_correct_count(self, git_repo_with_history: Path):
        reader = GitCliReader(str(git_repo_with_history))
        assert reader.commit_count() == 5


class TestCommitDates:
    def test_empty_repo_returns_none(self, tmp_git_repo: Path):
        reader = GitCliReader(str(tmp_git_repo))
        assert reader.first_commit_date() is None
        assert reader.last_commit_date() is None

    def test_dates_populated(self, git_repo_with_history: Path):
        reader = GitCliReader(str(git_repo_with_history))
        first = reader.first_commit_date()
        last = reader.last_commit_date()
        assert first is not None
        assert last is not None
        assert first < last


class TestFileChanges:
    def test_returns_all_changes(self, git_repo_with_history: Path):
        reader = GitCliReader(str(git_repo_with_history))
        changes = reader.file_changes()
        assert len(changes) > 0
        file_paths = {c.file_path for c in changes}
        assert "README.md" in file_paths
        assert "src/main.py" in file_paths
        assert "src/utils.py" in file_paths

    def test_since_filter(self, git_repo_with_history: Path):
        reader = GitCliReader(str(git_repo_with_history))
        since = datetime.now(timezone.utc) - timedelta(days=20)
        changes = reader.file_changes(since=since)
        # Only commits from last 20 days (days_ago=15 and days_ago=5)
        for c in changes:
            assert c.date >= since

    def test_until_filter(self, git_repo_with_history: Path):
        reader = GitCliReader(str(git_repo_with_history))
        until = datetime.now(timezone.utc) - timedelta(days=40)
        changes = reader.file_changes(until=until)
        # Only commits older than 40 days (days_ago=45, days_ago=60)
        for c in changes:
            assert c.date <= until

    def test_since_and_until_filter(self, git_repo_with_history: Path):
        reader = GitCliReader(str(git_repo_with_history))
        since = datetime.now(timezone.utc) - timedelta(days=50)
        until = datetime.now(timezone.utc) - timedelta(days=20)
        changes = reader.file_changes(since=since, until=until)
        for c in changes:
            assert c.date >= since
            assert c.date <= until

    def test_empty_repo_returns_empty(self, tmp_git_repo: Path):
        reader = GitCliReader(str(tmp_git_repo))
        assert reader.file_changes() == []

    def test_binary_files_excluded(self, tmp_git_repo: Path):
        # Create a binary file commit
        binary_path = tmp_git_repo / "image.png"
        binary_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        subprocess.run(
            ["git", "-C", str(tmp_git_repo), "add", "image.png"],
            capture_output=True, check=True,
        )
        subprocess.run(
            ["git", "-C", str(tmp_git_repo), "commit", "-m", "Add binary"],
            capture_output=True, check=True,
        )
        reader = GitCliReader(str(tmp_git_repo))
        changes = reader.file_changes()
        file_paths = [c.file_path for c in changes]
        assert "image.png" not in file_paths

    def test_author_names_extracted(self, git_repo_with_history: Path):
        reader = GitCliReader(str(git_repo_with_history))
        changes = reader.file_changes()
        for c in changes:
            assert c.author_name == "Test User"

    def test_author_emails_extracted(self, git_repo_with_history: Path):
        reader = GitCliReader(str(git_repo_with_history))
        changes = reader.file_changes()
        for c in changes:
            assert c.author_email == "test@example.com"


class TestMultiAuthorFileChanges:
    def test_multi_author_names_extracted(self, multi_author_repo: Path):
        reader = GitCliReader(str(multi_author_repo))
        changes = reader.file_changes()
        author_names = {c.author_name for c in changes}
        assert "Alice" in author_names
        assert "Bob" in author_names
        assert "Carol" in author_names

    def test_multi_author_emails_extracted(self, multi_author_repo: Path):
        reader = GitCliReader(str(multi_author_repo))
        changes = reader.file_changes()
        author_emails = {c.author_email for c in changes}
        assert "alice@example.com" in author_emails
        assert "bob@example.com" in author_emails
        assert "carol@example.com" in author_emails

    def test_author_file_mapping(self, multi_author_repo: Path):
        reader = GitCliReader(str(multi_author_repo))
        changes = reader.file_changes()
        # Alice should have 3 main.py commits
        alice_main = [c for c in changes if c.file_path == "main.py" and c.author_name == "Alice"]
        assert len(alice_main) == 3


class TestCouplingIntegration:
    """Integration tests: analyze_coupling with real git repos via GitCliReader."""

    def test_coupled_files_detected(self, coupled_repo: Path):
        reader = GitCliReader(str(coupled_repo))
        report = analyze_coupling(reader, str(coupled_repo), 90)
        assert len(report.coupling_pairs) > 0

    def test_strongest_pair_is_a_b(self, coupled_repo: Path):
        reader = GitCliReader(str(coupled_repo))
        report = analyze_coupling(reader, str(coupled_repo), 90)
        top = report.coupling_pairs[0]
        assert top.file_a == "a.py"
        assert top.file_b == "b.py"
        # a+b share 3 commits, union=3 → strength=1.0
        assert top.coupling_strength == 1.0

    def test_uncoupled_file_has_zero_distance(self, coupled_repo: Path):
        reader = GitCliReader(str(coupled_repo))
        report = analyze_coupling(reader, str(coupled_repo), 90)
        pain_map = {fp.file_path: fp for fp in report.file_pain}
        assert pain_map["d.py"].distance_raw == 0.0

    def test_all_files_have_pain_scores(self, coupled_repo: Path):
        reader = GitCliReader(str(coupled_repo))
        report = analyze_coupling(reader, str(coupled_repo), 90)
        paths = {fp.file_path for fp in report.file_pain}
        assert paths == {"a.py", "b.py", "c.py", "d.py"}

    def test_pain_scores_in_range(self, coupled_repo: Path):
        reader = GitCliReader(str(coupled_repo))
        report = analyze_coupling(reader, str(coupled_repo), 90)
        for fp in report.file_pain:
            assert 0.0 <= fp.pain_score <= 1.0
