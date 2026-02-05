import dataclasses
from datetime import datetime, timezone

from git_xrays.domain.models import (
    FileChange,
    FileMetrics,
    HotspotReport,
    RepoSummary,
)


class TestRepoSummary:
    def test_creation(self):
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        s = RepoSummary(
            repo_path="/repo", commit_count=10,
            first_commit_date=dt, last_commit_date=dt,
        )
        assert s.repo_path == "/repo"
        assert s.commit_count == 10
        assert s.first_commit_date == dt
        assert s.last_commit_date == dt

    def test_frozen(self):
        s = RepoSummary(
            repo_path="/repo", commit_count=1,
            first_commit_date=None, last_commit_date=None,
        )
        with __import__("pytest").raises(dataclasses.FrozenInstanceError):
            s.commit_count = 5  # type: ignore[misc]

    def test_none_dates(self):
        s = RepoSummary(
            repo_path="/repo", commit_count=0,
            first_commit_date=None, last_commit_date=None,
        )
        assert s.first_commit_date is None
        assert s.last_commit_date is None


class TestFileChange:
    def test_creation(self):
        dt = datetime(2024, 6, 15, tzinfo=timezone.utc)
        fc = FileChange(
            commit_hash="abc123", date=dt,
            file_path="src/main.py", lines_added=10, lines_deleted=3,
        )
        assert fc.commit_hash == "abc123"
        assert fc.date == dt
        assert fc.file_path == "src/main.py"
        assert fc.lines_added == 10
        assert fc.lines_deleted == 3

    def test_frozen(self):
        fc = FileChange(
            commit_hash="abc", date=datetime.now(timezone.utc),
            file_path="f.py", lines_added=1, lines_deleted=0,
        )
        with __import__("pytest").raises(dataclasses.FrozenInstanceError):
            fc.lines_added = 99  # type: ignore[misc]


class TestFileMetrics:
    def test_creation(self):
        fm = FileMetrics(
            file_path="a.py", change_frequency=5, code_churn=100,
            hotspot_score=0.75, rework_ratio=0.8,
        )
        assert fm.file_path == "a.py"
        assert fm.change_frequency == 5
        assert fm.code_churn == 100
        assert fm.hotspot_score == 0.75
        assert fm.rework_ratio == 0.8

    def test_frozen(self):
        fm = FileMetrics(
            file_path="a.py", change_frequency=1, code_churn=10,
            hotspot_score=1.0, rework_ratio=0.0,
        )
        with __import__("pytest").raises(dataclasses.FrozenInstanceError):
            fm.hotspot_score = 0.5  # type: ignore[misc]


class TestHotspotReport:
    def test_creation(self):
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        fm = FileMetrics(
            file_path="a.py", change_frequency=2, code_churn=50,
            hotspot_score=1.0, rework_ratio=0.5,
        )
        report = HotspotReport(
            repo_path="/repo", window_days=90,
            from_date=dt, to_date=dt,
            total_commits=5, files=[fm],
        )
        assert report.repo_path == "/repo"
        assert report.window_days == 90
        assert report.total_commits == 5
        assert len(report.files) == 1

    def test_empty_files_list(self):
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        report = HotspotReport(
            repo_path="/repo", window_days=30,
            from_date=dt, to_date=dt,
            total_commits=0, files=[],
        )
        assert report.files == []
        assert report.total_commits == 0
