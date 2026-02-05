from datetime import datetime, timedelta, timezone

from git_xrays.application.use_cases import analyze_hotspots, get_repo_summary
from git_xrays.domain.models import FileChange

from .fakes import FakeGitRepository

NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def _make_change(
    file_path: str, commit_hash: str,
    days_ago: int = 0, added: int = 10, deleted: int = 5,
) -> FileChange:
    return FileChange(
        commit_hash=commit_hash,
        date=NOW - timedelta(days=days_ago),
        file_path=file_path,
        lines_added=added,
        lines_deleted=deleted,
    )


class TestGetRepoSummary:
    def test_standard_case(self):
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        repo = FakeGitRepository(
            commit_count_val=42,
            first_commit_date_val=dt,
            last_commit_date_val=NOW,
        )
        summary = get_repo_summary(repo, "/my/repo")
        assert summary.repo_path == "/my/repo"
        assert summary.commit_count == 42
        assert summary.first_commit_date == dt
        assert summary.last_commit_date == NOW

    def test_empty_repo(self):
        repo = FakeGitRepository()
        summary = get_repo_summary(repo, "/empty")
        assert summary.commit_count == 0
        assert summary.first_commit_date is None
        assert summary.last_commit_date is None


class TestAnalyzeHotspots:
    def test_single_file_single_commit(self):
        changes = [_make_change("a.py", "aaa", days_ago=1, added=10, deleted=5)]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_hotspots(repo, "/repo", 90, current_time=NOW)
        assert len(report.files) == 1
        f = report.files[0]
        assert f.change_frequency == 1
        assert f.rework_ratio == 0.0
        assert f.hotspot_score == 1.0

    def test_multiple_files_different_churn(self):
        changes = [
            _make_change("a.py", "c1", days_ago=1, added=100, deleted=50),
            _make_change("a.py", "c2", days_ago=2, added=50, deleted=25),
            _make_change("b.py", "c1", days_ago=1, added=10, deleted=5),
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_hotspots(repo, "/repo", 90, current_time=NOW)
        assert len(report.files) == 2
        # a.py has highest hotspot (freq=2, churn=225 vs b.py freq=1, churn=15)
        assert report.files[0].file_path == "a.py"
        assert report.files[1].file_path == "b.py"
        # a.py: norm_freq=1.0, norm_churn=1.0 → hotspot=1.0
        assert report.files[0].hotspot_score == 1.0
        # b.py: norm_freq=0.5, norm_churn=15/225 ≈ 0.0667 → hotspot ≈ 0.0333
        assert report.files[1].hotspot_score == round(0.5 * (15 / 225), 4)

    def test_rework_ratio(self):
        changes = [
            _make_change("a.py", "c1", days_ago=1),
            _make_change("a.py", "c2", days_ago=2),
            _make_change("a.py", "c3", days_ago=3),
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_hotspots(repo, "/repo", 90, current_time=NOW)
        # freq=3 → rework = (3-1)/3 = 0.6667
        assert report.files[0].rework_ratio == round(2 / 3, 4)

    def test_empty_changes(self):
        repo = FakeGitRepository(file_changes_val=[])
        report = analyze_hotspots(repo, "/repo", 90, current_time=NOW)
        assert report.files == []
        assert report.total_commits == 0

    def test_window_filtering(self):
        changes = [
            _make_change("a.py", "c1", days_ago=10),   # inside 30-day window
            _make_change("b.py", "c2", days_ago=60),   # outside 30-day window
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_hotspots(repo, "/repo", 30, current_time=NOW)
        # FakeGitRepository filters by since/until, b.py should be excluded
        assert len(report.files) == 1
        assert report.files[0].file_path == "a.py"

    def test_sorted_by_hotspot_descending(self):
        changes = [
            _make_change("low.py", "c1", days_ago=1, added=1, deleted=0),
            _make_change("high.py", "c1", days_ago=1, added=100, deleted=50),
            _make_change("high.py", "c2", days_ago=2, added=80, deleted=40),
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_hotspots(repo, "/repo", 90, current_time=NOW)
        assert report.files[0].file_path == "high.py"
        assert report.files[1].file_path == "low.py"

    def test_total_commits_counts_unique_hashes(self):
        changes = [
            _make_change("a.py", "c1", days_ago=1),
            _make_change("b.py", "c1", days_ago=1),
            _make_change("a.py", "c2", days_ago=2),
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_hotspots(repo, "/repo", 90, current_time=NOW)
        assert report.total_commits == 2  # c1 and c2

    def test_report_dates(self):
        repo = FakeGitRepository(file_changes_val=[])
        report = analyze_hotspots(repo, "/repo", 30, current_time=NOW)
        assert report.to_date == NOW
        assert report.from_date == NOW - timedelta(days=30)
        assert report.window_days == 30

    def test_code_churn_is_added_plus_deleted(self):
        changes = [_make_change("a.py", "c1", days_ago=1, added=20, deleted=8)]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_hotspots(repo, "/repo", 90, current_time=NOW)
        assert report.files[0].code_churn == 28

    def test_multiple_commits_same_file_accumulate_churn(self):
        changes = [
            _make_change("a.py", "c1", days_ago=1, added=10, deleted=5),
            _make_change("a.py", "c2", days_ago=2, added=20, deleted=10),
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_hotspots(repo, "/repo", 90, current_time=NOW)
        assert report.files[0].code_churn == 45  # (10+5) + (20+10)

    def test_current_time_default_is_used_when_omitted(self):
        """When current_time is not passed, the function still works (uses now())."""
        changes = [_make_change("a.py", "c1", days_ago=1)]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_hotspots(repo, "/repo", 90)
        assert report.to_date is not None
