import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

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
