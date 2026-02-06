import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from git_xrays.application.use_cases import analyze_coupling, compare_hotspots
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


class TestResolveRef:
    def test_resolve_commit_hash(self, git_repo_with_history: Path):
        reader = GitCliReader(str(git_repo_with_history))
        # Get HEAD hash
        result = subprocess.run(
            ["git", "-C", str(git_repo_with_history), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        head_hash = result.stdout.strip()
        dt = reader.resolve_ref(head_hash)
        assert isinstance(dt, datetime)
        assert dt.tzinfo is not None

    def test_resolve_tag(self, tagged_repo: Path):
        reader = GitCliReader(str(tagged_repo))
        dt = reader.resolve_ref("v1.0")
        assert isinstance(dt, datetime)
        assert dt.tzinfo is not None

    def test_resolve_branch(self, git_repo_with_history: Path):
        reader = GitCliReader(str(git_repo_with_history))
        # Get current branch name dynamically
        result = subprocess.run(
            ["git", "-C", str(git_repo_with_history), "branch", "--show-current"],
            capture_output=True, text=True, check=True,
        )
        branch = result.stdout.strip()
        dt = reader.resolve_ref(branch)
        assert isinstance(dt, datetime)
        last = reader.last_commit_date()
        assert dt == last

    def test_resolve_short_hash(self, git_repo_with_history: Path):
        reader = GitCliReader(str(git_repo_with_history))
        result = subprocess.run(
            ["git", "-C", str(git_repo_with_history), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        short_hash = result.stdout.strip()
        dt = reader.resolve_ref(short_hash)
        assert isinstance(dt, datetime)

    def test_resolve_invalid_ref_raises(self, git_repo_with_history: Path):
        reader = GitCliReader(str(git_repo_with_history))
        with pytest.raises(ValueError, match="Cannot resolve ref"):
            reader.resolve_ref("nonexistent_ref_xyz")


class TestCompareIntegration:
    """Integration tests: compare_hotspots with real git repos via GitCliReader."""

    def test_compare_two_tags_produces_report(self, tagged_repo: Path):
        reader = GitCliReader(str(tagged_repo))
        report = compare_hotspots(reader, str(tagged_repo), 90, "v1.0", "v2.0")
        assert report.from_ref == "v1.0"
        assert report.to_ref == "v2.0"
        assert len(report.files) > 0

    def test_compare_detects_new_file_after_tag(self, tagged_repo: Path):
        """utils.py was added after v1.0 → shows as 'new'."""
        reader = GitCliReader(str(tagged_repo))
        report = compare_hotspots(reader, str(tagged_repo), 90, "v1.0", "v2.0")
        utils = [f for f in report.files if f.file_path == "src/utils.py"]
        assert len(utils) == 1
        assert utils[0].status == "new"

    def test_compare_detects_increased_churn(self, tagged_repo: Path):
        """main.py has more activity at v2.0 than v1.0."""
        reader = GitCliReader(str(tagged_repo))
        report = compare_hotspots(reader, str(tagged_repo), 90, "v1.0", "v2.0")
        main = [f for f in report.files if f.file_path == "src/main.py"]
        assert len(main) == 1
        assert main[0].to_churn >= main[0].from_churn

    def test_at_tag_matches_manual_time(self, tagged_repo: Path):
        """--at with tag produces same result as manual current_time."""
        from git_xrays.application.use_cases import _resolve_ref_to_datetime, analyze_hotspots
        reader = GitCliReader(str(tagged_repo))
        tag_time = _resolve_ref_to_datetime("v1.0", reader)
        report_via_tag = analyze_hotspots(reader, str(tagged_repo), 90, current_time=tag_time)
        # Resolve the tag date manually and compare
        report_manual = analyze_hotspots(reader, str(tagged_repo), 90, current_time=reader.resolve_ref("v1.0"))
        assert report_via_tag.total_commits == report_manual.total_commits
        assert len(report_via_tag.files) == len(report_manual.files)

    def test_compare_same_tag_all_unchanged(self, tagged_repo: Path):
        """Comparing a tag to itself → all unchanged."""
        reader = GitCliReader(str(tagged_repo))
        report = compare_hotspots(reader, str(tagged_repo), 90, "v1.0", "v1.0")
        for f in report.files:
            assert f.status == "unchanged"
            assert f.score_delta == 0.0
