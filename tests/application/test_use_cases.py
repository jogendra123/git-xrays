from datetime import datetime, timedelta, timezone

import pytest

from git_xrays.application.use_cases import (
    _compute_gini,
    _compute_rework_ratio,
    _resolve_ref_to_datetime,
    analyze_anemia,
    analyze_change_clusters,
    analyze_complexity,
    analyze_coupling,
    analyze_dx,
    analyze_effort,
    analyze_hotspots,
    analyze_knowledge,
    compare_hotspots,
    get_repo_summary,
)
from git_xrays.domain.models import FileChange

from .fakes import FakeGitRepository, FakeSourceCodeReader

NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def _make_change(
    file_path: str, commit_hash: str,
    days_ago: int = 0, added: int = 10, deleted: int = 5,
    author_name: str = "Test User", author_email: str = "test@example.com",
) -> FileChange:
    return FileChange(
        commit_hash=commit_hash,
        date=NOW - timedelta(days=days_ago),
        file_path=file_path,
        lines_added=added,
        lines_deleted=deleted,
        author_name=author_name,
        author_email=author_email,
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
        # a.py: max weighted freq and churn → hotspot=1.0
        assert report.files[0].hotspot_score == 1.0
        # b.py: lower score (temporal decay slightly adjusts raw ratios)
        assert report.files[1].hotspot_score < 0.1

    def test_rework_ratio_commits_within_14_days(self):
        changes = [
            _make_change("a.py", "c1", days_ago=1),
            _make_change("a.py", "c2", days_ago=2),
            _make_change("a.py", "c3", days_ago=3),
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_hotspots(repo, "/repo", 90, current_time=NOW)
        # 3 commits all within 14 days of each other → 2 rework / 3 total
        assert report.files[0].rework_ratio == round(2 / 3, 4)

    def test_rework_ratio_commits_far_apart(self):
        changes = [
            _make_change("a.py", "c1", days_ago=1),
            _make_change("a.py", "c2", days_ago=30),
            _make_change("a.py", "c3", days_ago=60),
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_hotspots(repo, "/repo", 90, current_time=NOW)
        # All commits >14 days apart → 0 rework / 3 total
        assert report.files[0].rework_ratio == 0.0

    def test_rework_ratio_single_commit(self):
        changes = [_make_change("a.py", "c1", days_ago=1)]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_hotspots(repo, "/repo", 90, current_time=NOW)
        assert report.files[0].rework_ratio == 0.0

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

    def test_relative_churn_small_file_ranks_higher(self):
        """Small file with same churn ranks higher than large file (relative churn)."""
        changes = [
            _make_change("small.py", "c1", days_ago=1, added=50, deleted=50),
            _make_change("big.py", "c1", days_ago=1, added=50, deleted=50),
        ]
        repo = FakeGitRepository(
            file_changes_val=changes,
            file_sizes_val={"small.py": 100, "big.py": 10000},
        )
        report = analyze_hotspots(repo, "/repo", 90, current_time=NOW)
        # Both have same freq=1, same absolute churn=100
        # Relative churn: small=100/100=1.0, big=100/10000=0.01
        # small.py should rank higher
        assert report.files[0].file_path == "small.py"
        assert report.files[0].hotspot_score > report.files[1].hotspot_score

    def test_file_size_stored_in_metrics(self):
        changes = [_make_change("a.py", "c1", days_ago=1)]
        repo = FakeGitRepository(
            file_changes_val=changes,
            file_sizes_val={"a.py": 500},
        )
        report = analyze_hotspots(repo, "/repo", 90, current_time=NOW)
        assert report.files[0].file_size == 500

    def test_unknown_file_size_fallback(self):
        """File not in file_sizes → file_size=0, falls back to absolute churn."""
        changes = [_make_change("a.py", "c1", days_ago=1)]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_hotspots(repo, "/repo", 90, current_time=NOW)
        assert report.files[0].file_size == 0
        assert report.files[0].hotspot_score == 1.0

    def test_recent_commit_ranks_higher_than_old(self):
        """Identical raw freq/churn, but different recency → different scores due to decay."""
        changes = [
            # recent.py: 1 commit, 1 day ago
            _make_change("recent.py", "c1", days_ago=1, added=10, deleted=5),
            # old.py: 1 commit, 85 days ago (same raw churn)
            _make_change("old.py", "c2", days_ago=85, added=10, deleted=5),
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_hotspots(repo, "/repo", 90, current_time=NOW)
        file_map = {f.file_path: f for f in report.files}
        # Same raw freq/churn, but recent gets higher score
        assert file_map["recent.py"].change_frequency == file_map["old.py"].change_frequency
        assert file_map["recent.py"].code_churn == file_map["old.py"].code_churn
        assert file_map["recent.py"].hotspot_score > file_map["old.py"].hotspot_score

    def test_decay_half_life_30_days(self):
        """Commit at 30 days ago gets weight ≈ 0.5 relative to day 0."""
        changes = [
            _make_change("recent.py", "c1", days_ago=0, added=10, deleted=5),
            _make_change("old.py", "c2", days_ago=30, added=10, deleted=5),
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_hotspots(repo, "/repo", 90, current_time=NOW)
        file_map = {f.file_path: f for f in report.files}
        # recent.py has weight ~1.0, old.py has weight ~0.5
        # hotspot = norm_freq * norm_churn; for recent norm_freq=1, norm_churn=1 → 1.0
        assert file_map["recent.py"].hotspot_score == 1.0
        # old.py: weight=0.5, norm_freq=0.5/1.0=0.5, norm_churn=0.5/1.0=0.5 → 0.25
        assert abs(file_map["old.py"].hotspot_score - 0.25) < 0.01

    def test_raw_churn_and_frequency_unchanged(self):
        """Raw FileMetrics fields are not affected by temporal decay."""
        changes = [
            _make_change("a.py", "c1", days_ago=1, added=20, deleted=10),
            _make_change("a.py", "c2", days_ago=60, added=30, deleted=15),
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_hotspots(repo, "/repo", 90, current_time=NOW)
        f = report.files[0]
        assert f.change_frequency == 2
        assert f.code_churn == 75  # (20+10) + (30+15)


class TestAnalyzeKnowledge:
    def test_single_author_concentration_is_one(self):
        changes = [
            _make_change("a.py", "c1", days_ago=1, author_name="Alice", author_email="alice@example.com"),
            _make_change("a.py", "c2", days_ago=2, author_name="Alice", author_email="alice@example.com"),
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_knowledge(repo, "/repo", 90, current_time=NOW)
        assert len(report.files) == 1
        assert report.files[0].knowledge_concentration == 1.0
        assert report.files[0].is_knowledge_island is True

    def test_two_equal_authors_concentration_is_zero(self):
        changes = [
            _make_change("a.py", "c1", days_ago=1, added=10, deleted=5,
                         author_name="Alice", author_email="alice@example.com"),
            _make_change("a.py", "c2", days_ago=1, added=10, deleted=5,
                         author_name="Bob", author_email="bob@example.com"),
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_knowledge(repo, "/repo", 90, current_time=NOW)
        assert report.files[0].knowledge_concentration == 0.0

    def test_dominant_author_high_concentration(self):
        # Alice: 90 churn, Bob: 10 churn → dominant
        changes = [
            _make_change("a.py", "c1", days_ago=1, added=50, deleted=40,
                         author_name="Alice", author_email="alice@example.com"),
            _make_change("a.py", "c2", days_ago=2, added=5, deleted=5,
                         author_name="Bob", author_email="bob@example.com"),
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_knowledge(repo, "/repo", 90, current_time=NOW)
        f = report.files[0]
        assert f.knowledge_concentration > 0.5
        assert f.primary_author == "Alice"
        assert f.primary_author_pct == 0.9
        assert f.is_knowledge_island is True

    def test_three_authors_proportional(self):
        changes = [
            _make_change("a.py", "c1", days_ago=1, added=30, deleted=20,
                         author_name="Alice", author_email="alice@example.com"),
            _make_change("a.py", "c2", days_ago=2, added=15, deleted=10,
                         author_name="Bob", author_email="bob@example.com"),
            _make_change("a.py", "c3", days_ago=3, added=5, deleted=5,
                         author_name="Carol", author_email="carol@example.com"),
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_knowledge(repo, "/repo", 90, current_time=NOW)
        f = report.files[0]
        assert f.author_count == 3
        assert f.primary_author == "Alice"
        # Alice: 50/85, Bob: 25/85, Carol: 10/85
        authors = {a.author_name: a for a in f.authors}
        assert authors["Alice"].proportion > authors["Bob"].proportion
        assert authors["Bob"].proportion > authors["Carol"].proportion

    def test_multiple_files_sorted_by_concentration_descending(self):
        changes = [
            # File with single author (concentration=1.0)
            _make_change("single.py", "c1", days_ago=1,
                         author_name="Alice", author_email="alice@example.com"),
            # File with two equal authors (concentration=0.0)
            _make_change("shared.py", "c2", days_ago=1, added=10, deleted=5,
                         author_name="Alice", author_email="alice@example.com"),
            _make_change("shared.py", "c3", days_ago=2, added=10, deleted=5,
                         author_name="Bob", author_email="bob@example.com"),
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_knowledge(repo, "/repo", 90, current_time=NOW)
        assert report.files[0].file_path == "single.py"
        assert report.files[1].file_path == "shared.py"

    def test_empty_changes_returns_empty_report(self):
        repo = FakeGitRepository(file_changes_val=[])
        report = analyze_knowledge(repo, "/repo", 90, current_time=NOW)
        assert report.files == []
        assert report.total_commits == 0
        assert report.developer_risk_index == 0.0
        assert report.knowledge_island_count == 0

    def test_window_filtering_excludes_old_changes(self):
        changes = [
            _make_change("a.py", "c1", days_ago=10,
                         author_name="Alice", author_email="alice@example.com"),
            _make_change("b.py", "c2", days_ago=60,
                         author_name="Bob", author_email="bob@example.com"),
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_knowledge(repo, "/repo", 30, current_time=NOW)
        assert len(report.files) == 1
        assert report.files[0].file_path == "a.py"

    def test_report_dates_and_window(self):
        repo = FakeGitRepository(file_changes_val=[])
        report = analyze_knowledge(repo, "/repo", 30, current_time=NOW)
        assert report.to_date == NOW
        assert report.from_date == NOW - timedelta(days=30)
        assert report.window_days == 30

    def test_knowledge_island_count(self):
        changes = [
            # Island: single author
            _make_change("island.py", "c1", days_ago=1,
                         author_name="Alice", author_email="alice@example.com"),
            # Not island: two equal authors
            _make_change("shared.py", "c2", days_ago=1, added=10, deleted=5,
                         author_name="Alice", author_email="alice@example.com"),
            _make_change("shared.py", "c3", days_ago=2, added=10, deleted=5,
                         author_name="Bob", author_email="bob@example.com"),
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_knowledge(repo, "/repo", 90, current_time=NOW)
        assert report.knowledge_island_count == 1

    def test_total_commits_counts_unique_hashes(self):
        changes = [
            _make_change("a.py", "c1", days_ago=1,
                         author_name="Alice", author_email="alice@example.com"),
            _make_change("b.py", "c1", days_ago=1,
                         author_name="Alice", author_email="alice@example.com"),
            _make_change("a.py", "c2", days_ago=2,
                         author_name="Bob", author_email="bob@example.com"),
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_knowledge(repo, "/repo", 90, current_time=NOW)
        assert report.total_commits == 2

    def test_author_contributions_have_correct_churn(self):
        changes = [
            _make_change("a.py", "c1", days_ago=1, added=20, deleted=8,
                         author_name="Alice", author_email="alice@example.com"),
            _make_change("a.py", "c2", days_ago=2, added=10, deleted=2,
                         author_name="Bob", author_email="bob@example.com"),
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_knowledge(repo, "/repo", 90, current_time=NOW)
        authors = {a.author_name: a for a in report.files[0].authors}
        assert authors["Alice"].total_churn == 28
        assert authors["Alice"].change_count == 1
        assert authors["Bob"].total_churn == 12
        assert authors["Bob"].change_count == 1

    def test_recent_contributions_get_higher_weighted_proportion(self):
        """Recent contributions should have higher weighted_proportion than old ones with equal raw churn."""
        changes = [
            # Alice: recent commit (1 day ago)
            _make_change("a.py", "c1", days_ago=1, added=10, deleted=5,
                         author_name="Alice", author_email="alice@example.com"),
            # Bob: old commit (80 days ago), same raw churn
            _make_change("a.py", "c2", days_ago=80, added=10, deleted=5,
                         author_name="Bob", author_email="bob@example.com"),
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_knowledge(repo, "/repo", 90, current_time=NOW)
        authors = {a.author_name: a for a in report.files[0].authors}
        # Equal raw proportion
        assert authors["Alice"].proportion == authors["Bob"].proportion
        # But Alice has higher weighted proportion (more recent)
        assert authors["Alice"].weighted_proportion > authors["Bob"].weighted_proportion

    def test_weighted_proportions_sum_to_one(self):
        changes = [
            _make_change("a.py", "c1", days_ago=1, added=20, deleted=10,
                         author_name="Alice", author_email="alice@example.com"),
            _make_change("a.py", "c2", days_ago=30, added=10, deleted=5,
                         author_name="Bob", author_email="bob@example.com"),
            _make_change("a.py", "c3", days_ago=60, added=5, deleted=5,
                         author_name="Carol", author_email="carol@example.com"),
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_knowledge(repo, "/repo", 90, current_time=NOW)
        total = sum(a.weighted_proportion for a in report.files[0].authors)
        assert abs(total - 1.0) < 0.01

    def test_kdi_uses_weighted_proportions(self):
        """Equal raw churn but different recency → KDI > 0 (weighted props differ)."""
        changes = [
            _make_change("a.py", "c1", days_ago=1, added=10, deleted=5,
                         author_name="Alice", author_email="alice@example.com"),
            _make_change("a.py", "c2", days_ago=80, added=10, deleted=5,
                         author_name="Bob", author_email="bob@example.com"),
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_knowledge(repo, "/repo", 90, current_time=NOW)
        # Raw proportions are equal (0.5 each) → raw KDI=0
        # But weighted proportions differ (Alice recent, Bob old) → KDI > 0
        assert report.files[0].knowledge_concentration > 0.0

    def test_kdi_same_day_equals_raw(self):
        """Same days_ago → weighted == raw → same KDI as raw computation."""
        changes = [
            _make_change("a.py", "c1", days_ago=5, added=10, deleted=5,
                         author_name="Alice", author_email="alice@example.com"),
            _make_change("a.py", "c2", days_ago=5, added=10, deleted=5,
                         author_name="Bob", author_email="bob@example.com"),
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_knowledge(repo, "/repo", 90, current_time=NOW)
        # Same day → equal weighted proportions → KDI = 0.0
        assert report.files[0].knowledge_concentration == 0.0


class TestComputeGini:
    def test_single_author_gini_is_zero(self):
        changes = [
            _make_change("a.py", "c1", days_ago=1,
                         author_name="Alice", author_email="alice@example.com"),
        ]
        # Single author → perfectly equal → Gini = 0.0
        assert _compute_gini(changes) == 0.0

    def test_two_equal_authors_gini_is_zero(self):
        changes = [
            _make_change("a.py", "c1", days_ago=1, added=10, deleted=5,
                         author_name="Alice", author_email="alice@example.com"),
            _make_change("a.py", "c2", days_ago=2, added=10, deleted=5,
                         author_name="Bob", author_email="bob@example.com"),
        ]
        # Equal churn → Gini = 0.0
        assert _compute_gini(changes) == 0.0

    def test_one_dominant_author_high_gini(self):
        # Alice: 60 churn, Bob: 25 churn, Carol: 15 churn → sorted: [15,25,60]
        changes = [
            _make_change("a.py", "c1", days_ago=1, added=50, deleted=10,
                         author_name="Alice", author_email="alice@example.com"),
            _make_change("a.py", "c2", days_ago=2, added=20, deleted=5,
                         author_name="Bob", author_email="bob@example.com"),
            _make_change("a.py", "c3", days_ago=3, added=10, deleted=5,
                         author_name="Carol", author_email="carol@example.com"),
        ]
        # sorted=[15,25,60], n=3, total=100
        # numerator = 2*(1*15 + 2*25 + 3*60) - 4*100 = 2*(15+50+180) - 400 = 490 - 400 = 90
        # denominator = 3*100 = 300
        # gini = 90/300 = 0.3
        assert _compute_gini(changes) == 0.3

    def test_empty_changes_gini_is_zero(self):
        assert _compute_gini([]) == 0.0


class TestComputeReworkRatio:
    def test_empty_dates(self):
        assert _compute_rework_ratio([]) == 0.0

    def test_single_date(self):
        assert _compute_rework_ratio([NOW]) == 0.0

    def test_all_within_window(self):
        dates = [NOW - timedelta(days=i) for i in range(5)]
        # 4 consecutive pairs all within 14 days → 4/5
        assert _compute_rework_ratio(dates) == round(4 / 5, 4)

    def test_all_beyond_window(self):
        dates = [NOW - timedelta(days=i * 20) for i in range(4)]
        # Gaps of 20 days, all > 14 → 0/4
        assert _compute_rework_ratio(dates) == 0.0

    def test_mixed_window(self):
        # 3 commits: day 0, day 5 (rework), day 30 (not rework)
        dates = [NOW, NOW - timedelta(days=5), NOW - timedelta(days=30)]
        # sorted: day-30, day-5 (25d gap → not rework), day-0 (5d gap → rework)
        # 1 rework / 3 total
        assert abs(_compute_rework_ratio(dates) - 1 / 3) < 1e-10

    def test_custom_window(self):
        dates = [NOW, NOW - timedelta(days=10)]
        # 10 days apart, window_days=5 → not rework
        assert _compute_rework_ratio(dates, window_days=5) == 0.0
        # window_days=10 → rework
        assert _compute_rework_ratio(dates, window_days=10) == round(1 / 2, 4)


class TestAnalyzeCoupling:
    """Temporal coupling: Jaccard, min_shared filter, lift filter, sorting."""

    def test_all_commits_shared_lift_equals_one_filtered(self):
        """If one file appears in every commit, lift=1.0 → filtered."""
        # a in c1,c2,c3; b in c1,c2 → total=3
        # expected_ab = (3/3)*(2/3)*3=2.0, lift=2/2.0=1.0 → filtered
        changes = [
            _make_change("a.py", "c1", days_ago=1),
            _make_change("b.py", "c1", days_ago=1),
            _make_change("a.py", "c2", days_ago=2),
            _make_change("b.py", "c2", days_ago=2),
            _make_change("a.py", "c3", days_ago=3),
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_coupling(repo, "/repo", 90, current_time=NOW)
        assert len(report.coupling_pairs) == 0

    def test_lift_filters_random_cochange(self):
        """Files appearing together only as often as random → lift ≤ 1.0 → filtered."""
        # a in c1,c2; b in c1,c2; total=2 → expected=2, lift=1.0 → filtered
        changes = [
            _make_change("a.py", "c1", days_ago=1),
            _make_change("b.py", "c1", days_ago=1),
            _make_change("a.py", "c2", days_ago=2),
            _make_change("b.py", "c2", days_ago=2),
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_coupling(repo, "/repo", 90, current_time=NOW)
        assert len(report.coupling_pairs) == 0

    def test_files_in_different_commits_no_pairs(self):
        """Files never in same commit → no coupling pairs."""
        changes = [
            _make_change("a.py", "c1", days_ago=1),
            _make_change("b.py", "c2", days_ago=2),
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_coupling(repo, "/repo", 90, current_time=NOW)
        assert len(report.coupling_pairs) == 0

    def test_empty_changes_empty_report(self):
        repo = FakeGitRepository(file_changes_val=[])
        report = analyze_coupling(repo, "/repo", 90, current_time=NOW)
        assert len(report.coupling_pairs) == 0
        assert len(report.file_pain) == 0
        assert report.total_commits == 0

    def test_jaccard_partial_overlap_with_lift(self):
        """Jaccard with partial overlap and lift > 1."""
        # a in c1,c2,c3,c4; b in c1,c2; c in c3,c4 → total=4
        # a+b: shared=2, union=4, jaccard=0.5
        # expected_ab = (4/4)*(2/4)*4 = 2.0, lift=2/2=1.0 → filtered
        # We need a better scenario:
        # a in c1,c2; b in c1,c2; c in c3 → total=3
        # a+b: shared=2, expected = (2/3)*(2/3)*3 = 4/3 ≈ 1.333, lift=2/1.333≈1.5
        changes = [
            _make_change("a.py", "c1", days_ago=1),
            _make_change("b.py", "c1", days_ago=1),
            _make_change("a.py", "c2", days_ago=2),
            _make_change("b.py", "c2", days_ago=2),
            _make_change("c.py", "c3", days_ago=3),  # unrelated commit
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_coupling(repo, "/repo", 90, current_time=NOW)
        assert len(report.coupling_pairs) == 1
        pair = report.coupling_pairs[0]
        assert pair.file_a == "a.py"
        assert pair.file_b == "b.py"
        # Jaccard: shared=2, union=2, strength=1.0
        assert pair.coupling_strength == 1.0
        # expected = (2/3)*(2/3)*3 = 4/3
        assert pair.expected_cochange == round(4 / 3, 4)
        # lift = 2 / (4/3) = 1.5
        assert pair.lift == 1.5

    def test_min_shared_commits_filters_weak_pairs(self):
        """Default min_shared_commits=2 filters pairs with only 1 co-occurrence."""
        changes = [
            _make_change("a.py", "c1", days_ago=1),
            _make_change("b.py", "c1", days_ago=1),
            _make_change("c.py", "c2", days_ago=2),
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_coupling(repo, "/repo", 90, current_time=NOW)
        assert len(report.coupling_pairs) == 0

    def test_support_and_lift(self):
        """Support = shared/total, lift = shared/expected."""
        # a in c1,c2,c3; b in c1,c2; d in c3 → total=3
        # a+b: shared=2, expected=(3/3)*(2/3)*3=2.0, lift=1.0 → filtered
        # Let's use: a in c1,c2; b in c1,c2; d in c3,c4 → total=4
        # a+b: shared=2, expected=(2/4)*(2/4)*4=1.0, lift=2.0
        changes = [
            _make_change("a.py", "c1", days_ago=1),
            _make_change("b.py", "c1", days_ago=1),
            _make_change("a.py", "c2", days_ago=2),
            _make_change("b.py", "c2", days_ago=2),
            _make_change("d.py", "c3", days_ago=3),
            _make_change("d.py", "c4", days_ago=4),
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_coupling(repo, "/repo", 90, current_time=NOW)
        assert report.total_commits == 4
        assert len(report.coupling_pairs) == 1
        pair = report.coupling_pairs[0]
        assert pair.support == round(2 / 4, 4)
        assert pair.lift == 2.0

    def test_pairs_sorted_by_strength_descending(self):
        """Pairs sorted by coupling_strength descending."""
        # a+b in c1,c2; a alone in c3; c in c1,c4 → total=4
        # a+b: shared=2, a_count=3, b_count=2, expected=(3/4)*(2/4)*4=1.5, lift=2/1.5≈1.33
        # a+c: shared=1 → filtered by min_shared
        # b+c: shared=1 → filtered by min_shared
        changes = [
            _make_change("a.py", "c1", days_ago=1),
            _make_change("b.py", "c1", days_ago=1),
            _make_change("a.py", "c2", days_ago=2),
            _make_change("b.py", "c2", days_ago=2),
            _make_change("a.py", "c3", days_ago=3),
            _make_change("c.py", "c1", days_ago=1),
            _make_change("c.py", "c4", days_ago=4),
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_coupling(repo, "/repo", 90, current_time=NOW)
        strengths = [p.coupling_strength for p in report.coupling_pairs]
        assert strengths == sorted(strengths, reverse=True)

    def test_file_a_less_than_file_b_alphabetically(self):
        """file_a < file_b alphabetically for deterministic pairs."""
        changes = [
            _make_change("z.py", "c1", days_ago=1),
            _make_change("a.py", "c1", days_ago=1),
            _make_change("z.py", "c2", days_ago=2),
            _make_change("a.py", "c2", days_ago=2),
            _make_change("c.py", "c3", days_ago=3),  # extra commit for lift>1
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_coupling(repo, "/repo", 90, current_time=NOW)
        assert len(report.coupling_pairs) == 1
        assert report.coupling_pairs[0].file_a == "a.py"
        assert report.coupling_pairs[0].file_b == "z.py"

    def test_recent_coupling_stronger_than_old(self):
        """Same raw shared counts but different recency → different temporal strength."""
        # Pair a+b: shared in recent commits (days_ago=1,2)
        # Pair c+d: shared in old commits (days_ago=80,85)
        # Both need lift>1 so add unrelated commits
        changes = [
            _make_change("a.py", "c1", days_ago=1),
            _make_change("b.py", "c1", days_ago=1),
            _make_change("a.py", "c2", days_ago=2),
            _make_change("b.py", "c2", days_ago=2),
            _make_change("c.py", "c3", days_ago=80),
            _make_change("d.py", "c3", days_ago=80),
            _make_change("c.py", "c4", days_ago=85),
            _make_change("d.py", "c4", days_ago=85),
            _make_change("x.py", "c5", days_ago=40),  # unrelated for lift>1
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_coupling(repo, "/repo", 90, current_time=NOW)
        pair_map = {(p.file_a, p.file_b): p for p in report.coupling_pairs}
        # Both pairs have raw shared=2 but a+b is recent, c+d is old
        assert ("a.py", "b.py") in pair_map
        assert ("c.py", "d.py") in pair_map
        assert pair_map[("a.py", "b.py")].coupling_strength >= pair_map[("c.py", "d.py")].coupling_strength

    def test_temporal_jaccard_same_day_equals_raw(self):
        """When all commits are on the same day, temporal Jaccard equals raw Jaccard."""
        changes = [
            _make_change("a.py", "c1", days_ago=5),
            _make_change("b.py", "c1", days_ago=5),
            _make_change("a.py", "c2", days_ago=5),
            _make_change("b.py", "c2", days_ago=5),
            _make_change("c.py", "c3", days_ago=5),  # extra for lift>1
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_coupling(repo, "/repo", 90, current_time=NOW)
        assert len(report.coupling_pairs) == 1
        pair = report.coupling_pairs[0]
        # Raw Jaccard: shared=2, union=2, strength=1.0
        # Temporal: same day → all weights equal → same result
        assert pair.coupling_strength == 1.0

    def test_min_shared_uses_raw_counts(self):
        """min_shared_commits filter uses raw int counts, not weighted float."""
        # 2 co-occurrences at very old dates → low weight but still raw count=2
        changes = [
            _make_change("a.py", "c1", days_ago=85),
            _make_change("b.py", "c1", days_ago=85),
            _make_change("a.py", "c2", days_ago=88),
            _make_change("b.py", "c2", days_ago=88),
            _make_change("c.py", "c3", days_ago=1),  # extra for lift>1
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_coupling(repo, "/repo", 90, current_time=NOW)
        # Should pass min_shared=2 filter (raw count=2)
        assert len(report.coupling_pairs) == 1
        assert report.coupling_pairs[0].shared_commits == 2


class TestPainMetric:
    """Steps 5-7: PAIN = normalized(size) x normalized(distance) x normalized(volatility)."""

    # Step 5: Size and volatility normalization
    def test_coupled_files_with_lift(self):
        """Coupled files with lift > 1.0 have non-zero pain."""
        changes = [
            _make_change("a.py", "c1", days_ago=1, added=10, deleted=5),
            _make_change("b.py", "c1", days_ago=1, added=10, deleted=5),
            _make_change("a.py", "c2", days_ago=2, added=10, deleted=5),
            _make_change("b.py", "c2", days_ago=2, added=10, deleted=5),
            _make_change("c.py", "c3", days_ago=3, added=5, deleted=2),  # extra for lift
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_coupling(repo, "/repo", 90, current_time=NOW)
        # a and b both coupled with lift > 1.0
        pain_map = {fp.file_path: fp for fp in report.file_pain}
        assert pain_map["a.py"].distance_raw > 0
        assert pain_map["b.py"].distance_raw > 0

    def test_size_normalized_relative_to_max(self):
        """Size (churn) normalized relative to max across files."""
        changes = [
            _make_change("big.py", "c1", days_ago=1, added=100, deleted=50),
            _make_change("small.py", "c1", days_ago=1, added=10, deleted=5),
            _make_change("big.py", "c2", days_ago=2, added=100, deleted=50),
            _make_change("small.py", "c2", days_ago=2, added=10, deleted=5),
            _make_change("x.py", "c3", days_ago=3, added=1, deleted=0),  # extra for lift
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_coupling(repo, "/repo", 90, current_time=NOW)
        pain_map = {fp.file_path: fp for fp in report.file_pain}
        assert pain_map["big.py"].size_normalized == 1.0
        assert pain_map["small.py"].size_normalized == round(30 / 300, 4)

    def test_volatility_normalized_relative_to_max(self):
        """Volatility (commit count) normalized relative to max."""
        changes = [
            _make_change("busy.py", "c1", days_ago=1),
            _make_change("busy.py", "c2", days_ago=2),
            _make_change("busy.py", "c3", days_ago=3),
            _make_change("quiet.py", "c1", days_ago=1),
            _make_change("quiet.py", "c2", days_ago=2),
            _make_change("x.py", "c4", days_ago=4),  # extra commit for lift
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_coupling(repo, "/repo", 90, current_time=NOW)
        pain_map = {fp.file_path: fp for fp in report.file_pain}
        assert pain_map["busy.py"].volatility_raw == 3
        assert pain_map["quiet.py"].volatility_raw == 2
        assert pain_map["busy.py"].volatility_normalized == 1.0
        assert pain_map["quiet.py"].volatility_normalized == round(2 / 3, 4)

    def test_empty_changes_no_pain(self):
        repo = FakeGitRepository(file_changes_val=[])
        report = analyze_coupling(repo, "/repo", 90, current_time=NOW)
        assert report.file_pain == []

    # Step 6: Distance dimension
    def test_distance_is_mean_coupling_strength(self):
        """Distance = mean coupling strength of filtered (lift>1) pairs involving the file."""
        # a in c1,c2,c3; b in c1,c2,c3; c in c1,c2,c4; d in c5 → total=5
        # a+b: shared=3, a_count=3, b_count=3, expected=(3/5)*(3/5)*5=1.8, lift=3/1.8≈1.67
        # a+c: shared=2, a_count=3, c_count=3, expected=1.8, lift=2/1.8≈1.11
        # b+c: shared=2, b_count=3, c_count=3, expected=1.8, lift≈1.11
        changes = [
            _make_change("a.py", "c1", days_ago=1),
            _make_change("b.py", "c1", days_ago=1),
            _make_change("c.py", "c1", days_ago=1),
            _make_change("a.py", "c2", days_ago=2),
            _make_change("b.py", "c2", days_ago=2),
            _make_change("c.py", "c2", days_ago=2),
            _make_change("a.py", "c3", days_ago=3),
            _make_change("b.py", "c3", days_ago=3),
            _make_change("c.py", "c4", days_ago=4),
            _make_change("d.py", "c5", days_ago=5),
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_coupling(repo, "/repo", 90, current_time=NOW)
        pain_map = {fp.file_path: fp for fp in report.file_pain}
        # a.py is involved in at least 2 coupling pairs
        assert pain_map["a.py"].distance_raw > 0

    def test_uncoupled_file_distance_zero(self):
        """File not in any coupling pair → distance=0, pain=0."""
        changes = [
            _make_change("a.py", "c1", days_ago=1),
            _make_change("b.py", "c1", days_ago=1),
            _make_change("a.py", "c2", days_ago=2),
            _make_change("b.py", "c2", days_ago=2),
            _make_change("lone.py", "c3", days_ago=3),  # uncoupled
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_coupling(repo, "/repo", 90, current_time=NOW)
        pain_map = {fp.file_path: fp for fp in report.file_pain}
        assert pain_map["lone.py"].distance_raw == 0.0
        assert pain_map["lone.py"].pain_score == 0.0

    def test_distance_normalized_relative_to_max(self):
        """Distance normalized relative to max distance across files."""
        # a+b in c1,c2; lone in c3 → total=3
        # expected_ab = (2/3)*(2/3)*3 = 4/3, lift=2/(4/3)=1.5 → kept
        changes = [
            _make_change("a.py", "c1", days_ago=1),
            _make_change("b.py", "c1", days_ago=1),
            _make_change("a.py", "c2", days_ago=2),
            _make_change("b.py", "c2", days_ago=2),
            _make_change("lone.py", "c3", days_ago=3),
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_coupling(repo, "/repo", 90, current_time=NOW)
        pain_map = {fp.file_path: fp for fp in report.file_pain}
        # a and b both coupled with lift>1, so they have distance>0
        assert pain_map["a.py"].distance_normalized == 1.0
        assert pain_map["b.py"].distance_normalized == 1.0
        assert pain_map["lone.py"].distance_normalized == 0.0

    def test_pain_is_product_of_normalized(self):
        """PAIN = product of normalized dimensions."""
        changes = [
            _make_change("a.py", "c1", days_ago=1, added=100, deleted=50),
            _make_change("b.py", "c1", days_ago=1, added=10, deleted=5),
            _make_change("a.py", "c2", days_ago=2, added=100, deleted=50),
            _make_change("b.py", "c2", days_ago=2, added=10, deleted=5),
            _make_change("x.py", "c3", days_ago=3, added=1, deleted=0),  # extra for lift
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_coupling(repo, "/repo", 90, current_time=NOW)
        for fp in report.file_pain:
            expected = round(fp.size_normalized * fp.volatility_normalized * fp.distance_normalized, 4)
            assert fp.pain_score == expected

    # Step 7: Sorting + edge cases
    def test_file_pain_sorted_by_score_descending(self):
        """file_pain sorted by pain_score descending."""
        changes = [
            _make_change("big.py", "c1", days_ago=1, added=100, deleted=50),
            _make_change("small.py", "c1", days_ago=1, added=10, deleted=5),
            _make_change("big.py", "c2", days_ago=2, added=100, deleted=50),
            _make_change("small.py", "c2", days_ago=2, added=10, deleted=5),
            _make_change("x.py", "c3", days_ago=3, added=1, deleted=0),  # extra for lift
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_coupling(repo, "/repo", 90, current_time=NOW)
        scores = [fp.pain_score for fp in report.file_pain]
        assert scores == sorted(scores, reverse=True)

    def test_all_files_appear_in_pain(self):
        """All files appear in file_pain, even uncoupled ones."""
        changes = [
            _make_change("a.py", "c1", days_ago=1),
            _make_change("b.py", "c1", days_ago=1),
            _make_change("a.py", "c2", days_ago=2),
            _make_change("b.py", "c2", days_ago=2),
            _make_change("lone.py", "c3", days_ago=3),
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_coupling(repo, "/repo", 90, current_time=NOW)
        paths = {fp.file_path for fp in report.file_pain}
        assert paths == {"a.py", "b.py", "lone.py"}

    def test_report_dates_and_total_commits(self):
        changes = [
            _make_change("a.py", "c1", days_ago=1),
            _make_change("b.py", "c1", days_ago=1),
            _make_change("a.py", "c2", days_ago=2),
            _make_change("b.py", "c2", days_ago=2),
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_coupling(repo, "/repo", 90, current_time=NOW)
        assert report.to_date == NOW
        assert report.from_date == NOW - timedelta(days=90)
        assert report.total_commits == 2
        assert report.window_days == 90

    # Step 8: Window filtering
    def test_old_changes_excluded_by_window(self):
        changes = [
            _make_change("a.py", "c1", days_ago=10),
            _make_change("b.py", "c1", days_ago=10),
            _make_change("a.py", "c2", days_ago=15),
            _make_change("b.py", "c2", days_ago=15),
            _make_change("x.py", "c4", days_ago=20),  # extra commit for lift>1
            _make_change("a.py", "c3", days_ago=60),  # outside 30d window
            _make_change("b.py", "c3", days_ago=60),
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_coupling(repo, "/repo", 30, current_time=NOW)
        assert report.total_commits == 3
        assert len(report.coupling_pairs) == 1
        assert report.coupling_pairs[0].shared_commits == 2

    def test_current_time_default_works(self):
        changes = [_make_change("a.py", "c1", days_ago=1)]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_coupling(repo, "/repo", 90)
        assert report.to_date is not None


class TestFakeGitRepositoryResolveRef:
    def test_known_ref_returns_datetime(self):
        dt = datetime(2024, 3, 1, tzinfo=timezone.utc)
        repo = FakeGitRepository(ref_dates={"v1.0": dt})
        assert repo.resolve_ref("v1.0") == dt

    def test_unknown_ref_raises_value_error(self):
        repo = FakeGitRepository(ref_dates={})
        with pytest.raises(ValueError, match="Unknown ref"):
            repo.resolve_ref("nonexistent")

    def test_existing_tests_unaffected(self):
        """FakeGitRepository still works without ref_dates param."""
        repo = FakeGitRepository(commit_count_val=5)
        assert repo.commit_count() == 5


class TestResolveRefToDatetime:
    def test_iso_datetime_parsed(self):
        result = _resolve_ref_to_datetime("2024-03-01T12:00:00+00:00", None)
        assert result == datetime(2024, 3, 1, 12, 0, 0, tzinfo=timezone.utc)

    def test_iso_date_only_parsed(self):
        result = _resolve_ref_to_datetime("2024-03-01", None)
        assert result == datetime(2024, 3, 1, tzinfo=timezone.utc)

    def test_non_date_delegates_to_repo(self):
        dt = datetime(2024, 6, 1, tzinfo=timezone.utc)
        repo = FakeGitRepository(ref_dates={"v1.0": dt})
        result = _resolve_ref_to_datetime("v1.0", repo)
        assert result == dt

    def test_unknown_ref_raises_value_error(self):
        repo = FakeGitRepository(ref_dates={})
        with pytest.raises(ValueError):
            _resolve_ref_to_datetime("nonexistent", repo)


class TestCompareHotspots:
    """Steps 3-5: compare_hotspots basic cases, sorting/counts, metadata."""

    # Step 3: Basic cases

    def test_identical_snapshots_all_unchanged(self):
        """Same data at both refs → all zero deltas, status 'unchanged'."""
        dt = datetime(2024, 6, 1, tzinfo=timezone.utc)
        changes = [
            _make_change("a.py", "c1", days_ago=10, added=20, deleted=5),
            _make_change("b.py", "c2", days_ago=5, added=10, deleted=3),
        ]
        repo = FakeGitRepository(
            file_changes_val=changes,
            ref_dates={"v1.0": dt, "v2.0": dt},
        )
        report = compare_hotspots(repo, "/repo", 90, "v1.0", "v2.0")
        for f in report.files:
            assert f.score_delta == 0.0
            assert f.status == "unchanged"

    def test_new_file_in_to_snapshot(self):
        """File only in 'to' → status 'new', from values 0."""
        from_date = datetime(2024, 3, 1, tzinfo=timezone.utc)
        to_date = datetime(2024, 6, 1, tzinfo=timezone.utc)
        changes = [
            # This change is only within the "to" window (within 90d of to_date)
            _make_change("new_file.py", "c1", days_ago=10, added=30, deleted=5),
        ]
        repo = FakeGitRepository(
            file_changes_val=changes,
            ref_dates={"from": from_date, "to": to_date},
        )
        report = compare_hotspots(repo, "/repo", 90, "from", "to")
        new_files = [f for f in report.files if f.status == "new"]
        assert len(new_files) >= 1
        for f in new_files:
            assert f.from_score == 0.0
            assert f.from_churn == 0
            assert f.from_frequency == 0

    def test_removed_file_in_from_snapshot(self):
        """File only in 'from' → status 'removed', to values 0."""
        from_date = datetime(2024, 6, 1, tzinfo=timezone.utc)
        to_date = datetime(2024, 9, 1, tzinfo=timezone.utc)
        # Change is within "from" window but not "to" window
        changes = [
            _make_change("old_file.py", "c1",
                         days_ago=0, added=20, deleted=5),
        ]
        # Manually set the date to be within from_date window but not to_date window
        old_change = FileChange(
            commit_hash="c1",
            date=datetime(2024, 5, 15, tzinfo=timezone.utc),
            file_path="old_file.py",
            lines_added=20, lines_deleted=5,
            author_name="Test User", author_email="test@example.com",
        )
        repo = FakeGitRepository(
            file_changes_val=[old_change],
            ref_dates={"from": from_date, "to": to_date},
        )
        report = compare_hotspots(repo, "/repo", 90, "from", "to")
        removed = [f for f in report.files if f.status == "removed"]
        assert len(removed) == 1
        assert removed[0].to_score == 0.0
        assert removed[0].to_churn == 0
        assert removed[0].to_frequency == 0

    def test_degraded_file_score_increases(self):
        """Higher score in 'to' → 'degraded'."""
        from_date = datetime(2024, 3, 1, tzinfo=timezone.utc)
        to_date = datetime(2024, 6, 1, tzinfo=timezone.utc)
        # a.py: small churn in "from", large churn in "to"
        # b.py: baseline present in both windows with constant churn
        changes = [
            # "from" window: a.py small, b.py large
            FileChange(commit_hash="c1", date=datetime(2024, 2, 15, tzinfo=timezone.utc),
                       file_path="a.py", lines_added=5, lines_deleted=2,
                       author_name="Test", author_email="test@test.com"),
            FileChange(commit_hash="c1", date=datetime(2024, 2, 15, tzinfo=timezone.utc),
                       file_path="b.py", lines_added=100, lines_deleted=50,
                       author_name="Test", author_email="test@test.com"),
            # "to" window: a.py large (2 commits), b.py small
            FileChange(commit_hash="c2", date=datetime(2024, 5, 15, tzinfo=timezone.utc),
                       file_path="a.py", lines_added=100, lines_deleted=50,
                       author_name="Test", author_email="test@test.com"),
            FileChange(commit_hash="c3", date=datetime(2024, 5, 20, tzinfo=timezone.utc),
                       file_path="a.py", lines_added=80, lines_deleted=40,
                       author_name="Test", author_email="test@test.com"),
            FileChange(commit_hash="c2", date=datetime(2024, 5, 15, tzinfo=timezone.utc),
                       file_path="b.py", lines_added=5, lines_deleted=2,
                       author_name="Test", author_email="test@test.com"),
        ]
        repo = FakeGitRepository(
            file_changes_val=changes,
            ref_dates={"from": from_date, "to": to_date},
        )
        report = compare_hotspots(repo, "/repo", 90, "from", "to")
        a_file = [f for f in report.files if f.file_path == "a.py"][0]
        assert a_file.score_delta > 0
        assert a_file.status == "degraded"

    def test_improved_file_score_decreases(self):
        """Lower score in 'to' → 'improved'."""
        from_date = datetime(2024, 3, 1, tzinfo=timezone.utc)
        to_date = datetime(2024, 6, 1, tzinfo=timezone.utc)
        # a.py: large churn in "from", small in "to"
        # b.py: baseline present in both windows
        changes = [
            # "from" window: a.py large (2 commits), b.py small
            FileChange(commit_hash="c1", date=datetime(2024, 2, 15, tzinfo=timezone.utc),
                       file_path="a.py", lines_added=100, lines_deleted=50,
                       author_name="Test", author_email="test@test.com"),
            FileChange(commit_hash="c2", date=datetime(2024, 2, 20, tzinfo=timezone.utc),
                       file_path="a.py", lines_added=80, lines_deleted=40,
                       author_name="Test", author_email="test@test.com"),
            FileChange(commit_hash="c1", date=datetime(2024, 2, 15, tzinfo=timezone.utc),
                       file_path="b.py", lines_added=5, lines_deleted=2,
                       author_name="Test", author_email="test@test.com"),
            # "to" window: a.py small, b.py large
            FileChange(commit_hash="c3", date=datetime(2024, 5, 15, tzinfo=timezone.utc),
                       file_path="a.py", lines_added=5, lines_deleted=2,
                       author_name="Test", author_email="test@test.com"),
            FileChange(commit_hash="c3", date=datetime(2024, 5, 15, tzinfo=timezone.utc),
                       file_path="b.py", lines_added=100, lines_deleted=50,
                       author_name="Test", author_email="test@test.com"),
        ]
        repo = FakeGitRepository(
            file_changes_val=changes,
            ref_dates={"from": from_date, "to": to_date},
        )
        report = compare_hotspots(repo, "/repo", 90, "from", "to")
        a_file = [f for f in report.files if f.file_path == "a.py"][0]
        assert a_file.score_delta < 0
        assert a_file.status == "improved"

    def test_churn_delta_computed(self):
        """churn_delta = to_churn - from_churn."""
        dt = datetime(2024, 6, 1, tzinfo=timezone.utc)
        changes = [
            _make_change("a.py", "c1", days_ago=10, added=20, deleted=10),
        ]
        repo = FakeGitRepository(
            file_changes_val=changes,
            ref_dates={"v1": dt, "v2": dt},
        )
        report = compare_hotspots(repo, "/repo", 90, "v1", "v2")
        f = report.files[0]
        assert f.churn_delta == f.to_churn - f.from_churn

    def test_frequency_delta_computed(self):
        """frequency_delta = to_frequency - from_frequency."""
        dt = datetime(2024, 6, 1, tzinfo=timezone.utc)
        changes = [
            _make_change("a.py", "c1", days_ago=10),
            _make_change("a.py", "c2", days_ago=5),
        ]
        repo = FakeGitRepository(
            file_changes_val=changes,
            ref_dates={"v1": dt, "v2": dt},
        )
        report = compare_hotspots(repo, "/repo", 90, "v1", "v2")
        f = report.files[0]
        assert f.frequency_delta == f.to_frequency - f.from_frequency

    def test_empty_both_snapshots(self):
        """No changes in either → empty files list."""
        from_date = datetime(2024, 3, 1, tzinfo=timezone.utc)
        to_date = datetime(2024, 6, 1, tzinfo=timezone.utc)
        repo = FakeGitRepository(
            file_changes_val=[],
            ref_dates={"from": from_date, "to": to_date},
        )
        report = compare_hotspots(repo, "/repo", 90, "from", "to")
        assert report.files == []

    # Step 4: Sorting and counts

    def test_sorted_by_abs_score_delta_descending(self):
        from_date = datetime(2024, 3, 1, tzinfo=timezone.utc)
        to_date = datetime(2024, 6, 1, tzinfo=timezone.utc)
        changes = [
            # File with large delta
            FileChange(commit_hash="c1", date=datetime(2024, 5, 15, tzinfo=timezone.utc),
                       file_path="big_change.py", lines_added=100, lines_deleted=50,
                       author_name="Test", author_email="test@test.com"),
            FileChange(commit_hash="c2", date=datetime(2024, 5, 20, tzinfo=timezone.utc),
                       file_path="big_change.py", lines_added=80, lines_deleted=40,
                       author_name="Test", author_email="test@test.com"),
            # File with small delta
            FileChange(commit_hash="c3", date=datetime(2024, 5, 25, tzinfo=timezone.utc),
                       file_path="small_change.py", lines_added=5, lines_deleted=2,
                       author_name="Test", author_email="test@test.com"),
        ]
        repo = FakeGitRepository(
            file_changes_val=changes,
            ref_dates={"from": from_date, "to": to_date},
        )
        report = compare_hotspots(repo, "/repo", 90, "from", "to")
        abs_deltas = [abs(f.score_delta) for f in report.files]
        assert abs_deltas == sorted(abs_deltas, reverse=True)

    def test_new_hotspot_count(self):
        from_date = datetime(2024, 3, 1, tzinfo=timezone.utc)
        to_date = datetime(2024, 6, 1, tzinfo=timezone.utc)
        changes = [
            FileChange(commit_hash="c1", date=datetime(2024, 5, 15, tzinfo=timezone.utc),
                       file_path="new1.py", lines_added=10, lines_deleted=5,
                       author_name="Test", author_email="test@test.com"),
            FileChange(commit_hash="c2", date=datetime(2024, 5, 20, tzinfo=timezone.utc),
                       file_path="new2.py", lines_added=20, lines_deleted=10,
                       author_name="Test", author_email="test@test.com"),
        ]
        repo = FakeGitRepository(
            file_changes_val=changes,
            ref_dates={"from": from_date, "to": to_date},
        )
        report = compare_hotspots(repo, "/repo", 90, "from", "to")
        assert report.new_hotspot_count == 2

    def test_removed_hotspot_count(self):
        from_date = datetime(2024, 6, 1, tzinfo=timezone.utc)
        to_date = datetime(2024, 9, 1, tzinfo=timezone.utc)
        changes = [
            FileChange(commit_hash="c1", date=datetime(2024, 5, 15, tzinfo=timezone.utc),
                       file_path="old1.py", lines_added=10, lines_deleted=5,
                       author_name="Test", author_email="test@test.com"),
            FileChange(commit_hash="c2", date=datetime(2024, 4, 20, tzinfo=timezone.utc),
                       file_path="old2.py", lines_added=20, lines_deleted=10,
                       author_name="Test", author_email="test@test.com"),
        ]
        repo = FakeGitRepository(
            file_changes_val=changes,
            ref_dates={"from": from_date, "to": to_date},
        )
        report = compare_hotspots(repo, "/repo", 90, "from", "to")
        assert report.removed_hotspot_count == 2

    def test_improved_count(self):
        from_date = datetime(2024, 3, 1, tzinfo=timezone.utc)
        to_date = datetime(2024, 6, 1, tzinfo=timezone.utc)
        changes = [
            # "from" window: a.py high churn, b.py low
            FileChange(commit_hash="c1", date=datetime(2024, 2, 15, tzinfo=timezone.utc),
                       file_path="a.py", lines_added=100, lines_deleted=50,
                       author_name="Test", author_email="test@test.com"),
            FileChange(commit_hash="c2", date=datetime(2024, 2, 20, tzinfo=timezone.utc),
                       file_path="a.py", lines_added=80, lines_deleted=40,
                       author_name="Test", author_email="test@test.com"),
            FileChange(commit_hash="c1", date=datetime(2024, 2, 15, tzinfo=timezone.utc),
                       file_path="b.py", lines_added=5, lines_deleted=2,
                       author_name="Test", author_email="test@test.com"),
            # "to" window: a.py low, b.py high
            FileChange(commit_hash="c3", date=datetime(2024, 5, 15, tzinfo=timezone.utc),
                       file_path="a.py", lines_added=5, lines_deleted=2,
                       author_name="Test", author_email="test@test.com"),
            FileChange(commit_hash="c3", date=datetime(2024, 5, 15, tzinfo=timezone.utc),
                       file_path="b.py", lines_added=100, lines_deleted=50,
                       author_name="Test", author_email="test@test.com"),
        ]
        repo = FakeGitRepository(
            file_changes_val=changes,
            ref_dates={"from": from_date, "to": to_date},
        )
        report = compare_hotspots(repo, "/repo", 90, "from", "to")
        assert report.improved_count >= 1

    def test_degraded_count(self):
        from_date = datetime(2024, 3, 1, tzinfo=timezone.utc)
        to_date = datetime(2024, 6, 1, tzinfo=timezone.utc)
        changes = [
            # "from" window: a.py low, b.py high
            FileChange(commit_hash="c1", date=datetime(2024, 2, 15, tzinfo=timezone.utc),
                       file_path="a.py", lines_added=5, lines_deleted=2,
                       author_name="Test", author_email="test@test.com"),
            FileChange(commit_hash="c1", date=datetime(2024, 2, 15, tzinfo=timezone.utc),
                       file_path="b.py", lines_added=100, lines_deleted=50,
                       author_name="Test", author_email="test@test.com"),
            # "to" window: a.py high (2 commits), b.py low
            FileChange(commit_hash="c2", date=datetime(2024, 5, 15, tzinfo=timezone.utc),
                       file_path="a.py", lines_added=100, lines_deleted=50,
                       author_name="Test", author_email="test@test.com"),
            FileChange(commit_hash="c3", date=datetime(2024, 5, 20, tzinfo=timezone.utc),
                       file_path="a.py", lines_added=80, lines_deleted=40,
                       author_name="Test", author_email="test@test.com"),
            FileChange(commit_hash="c2", date=datetime(2024, 5, 15, tzinfo=timezone.utc),
                       file_path="b.py", lines_added=5, lines_deleted=2,
                       author_name="Test", author_email="test@test.com"),
        ]
        repo = FakeGitRepository(
            file_changes_val=changes,
            ref_dates={"from": from_date, "to": to_date},
        )
        report = compare_hotspots(repo, "/repo", 90, "from", "to")
        assert report.degraded_count >= 1

    # Step 5: Report metadata

    def test_report_contains_ref_strings(self):
        dt = datetime(2024, 6, 1, tzinfo=timezone.utc)
        repo = FakeGitRepository(
            file_changes_val=[],
            ref_dates={"v1.0": dt, "v2.0": dt},
        )
        report = compare_hotspots(repo, "/repo", 90, "v1.0", "v2.0")
        assert report.from_ref == "v1.0"
        assert report.to_ref == "v2.0"

    def test_report_contains_resolved_dates(self):
        dt1 = datetime(2024, 3, 1, tzinfo=timezone.utc)
        dt2 = datetime(2024, 6, 1, tzinfo=timezone.utc)
        repo = FakeGitRepository(
            file_changes_val=[],
            ref_dates={"v1.0": dt1, "v2.0": dt2},
        )
        report = compare_hotspots(repo, "/repo", 90, "v1.0", "v2.0")
        assert report.from_date == dt1
        assert report.to_date == dt2

    def test_report_contains_commit_counts(self):
        dt = datetime(2024, 6, 1, tzinfo=timezone.utc)
        changes = [
            _make_change("a.py", "c1", days_ago=10),
            _make_change("b.py", "c2", days_ago=5),
        ]
        repo = FakeGitRepository(
            file_changes_val=changes,
            ref_dates={"v1": dt, "v2": dt},
        )
        report = compare_hotspots(repo, "/repo", 90, "v1", "v2")
        assert report.from_total_commits == report.to_total_commits
        assert report.from_total_commits == 2


class TestFakeSourceCodeReader:
    def test_list_python_files_returns_sorted_keys(self):
        reader = FakeSourceCodeReader({
            "b.py": "pass",
            "a.py": "pass",
            "c.py": "pass",
        })
        assert reader.list_python_files() == ["a.py", "b.py", "c.py"]

    def test_read_file_returns_content(self):
        reader = FakeSourceCodeReader({"models.py": "class Foo: pass"})
        assert reader.read_file("models.py") == "class Foo: pass"

    def test_read_file_missing_raises(self):
        reader = FakeSourceCodeReader({})
        with pytest.raises(FileNotFoundError):
            reader.read_file("missing.py")

    def test_empty_reader(self):
        reader = FakeSourceCodeReader({})
        assert reader.list_python_files() == []


class TestAnalyzeAnemia:
    def test_empty_repo_returns_empty_report(self):
        reader = FakeSourceCodeReader({})
        report = analyze_anemia(reader, "/repo")
        assert report.total_files == 0
        assert report.total_classes == 0
        assert report.anemic_count == 0
        assert report.files == []

    def test_single_anemic_class_detected(self):
        source = (
            "class UserDTO:\n"
            "    name = ''\n"
            "    email = ''\n"
            "    age = 0\n"
            "    def get_name(self):\n"
            "        return self.name\n"
        )
        reader = FakeSourceCodeReader({"models.py": source})
        report = analyze_anemia(reader, "/repo")
        assert report.total_classes == 1
        assert report.anemic_count == 1

    def test_healthy_class_not_flagged(self):
        source = (
            "class Service:\n"
            "    def process(self):\n"
            "        if True:\n"
            "            return 42\n"
        )
        reader = FakeSourceCodeReader({"svc.py": source})
        report = analyze_anemia(reader, "/repo")
        assert report.total_classes == 1
        assert report.anemic_count == 0

    def test_multiple_files_aggregated(self):
        anemic = (
            "class DTO:\n"
            "    x = 1\n"
            "    y = 2\n"
            "    def get_x(self): return self.x\n"
        )
        healthy = (
            "class Svc:\n"
            "    def run(self):\n"
            "        if True: pass\n"
        )
        reader = FakeSourceCodeReader({"dto.py": anemic, "svc.py": healthy})
        report = analyze_anemia(reader, "/repo")
        assert report.total_files == 2
        assert report.total_classes == 2

    def test_files_sorted_by_worst_ams_desc(self):
        highly_anemic = (
            "class HighAnemic:\n"
            "    x = 1\n"
            "    y = 2\n"
            "    z = 3\n"
            "    def get_x(self): return self.x\n"
            "    def get_y(self): return self.y\n"
        )
        slightly_anemic = (
            "class SlightAnemic:\n"
            "    x = 1\n"
            "    def get(self): return self.x\n"
            "    def check(self):\n"
            "        if True: pass\n"
        )
        reader = FakeSourceCodeReader({
            "high.py": highly_anemic,
            "slight.py": slightly_anemic,
        })
        report = analyze_anemia(reader, "/repo")
        assert report.files[0].worst_ams >= report.files[1].worst_ams

    def test_anemic_percentage_computed(self):
        anemic = "class DTO:\n    x = 1\n    def get(self): return self.x\n"
        healthy = "class Svc:\n    def run(self):\n        if True: pass\n"
        reader = FakeSourceCodeReader({"dto.py": anemic, "svc.py": healthy})
        report = analyze_anemia(reader, "/repo")
        # 1 anemic out of 2 = 50%
        assert report.anemic_percentage == 50.0

    def test_average_ams_computed(self):
        source = (
            "class A:\n"
            "    x = 1\n"
            "    def get(self): return self.x\n"
            "\n"
            "class B:\n"
            "    def run(self):\n"
            "        if True: pass\n"
        )
        reader = FakeSourceCodeReader({"mixed.py": source})
        report = analyze_anemia(reader, "/repo")
        # Average of A.ams and B.ams
        all_ams = [c.ams for f in report.files for c in f.classes]
        expected = round(sum(all_ams) / len(all_ams), 4)
        assert report.average_ams == expected

    def test_touch_count_integrated(self):
        models = "class Foo:\n    x = 1\n"
        app = "import models\nclass Bar:\n    def run(self):\n        if True: pass\n"
        reader = FakeSourceCodeReader({"models.py": models, "app.py": app})
        report = analyze_anemia(reader, "/repo")
        file_map = {f.file_path: f for f in report.files}
        assert file_map["models.py"].touch_count == 1
        assert file_map["app.py"].touch_count == 0


class TestAnalyzeComplexity:
    def test_empty_repo_returns_empty_report(self):
        reader = FakeSourceCodeReader({})
        report = analyze_complexity(reader, "/repo")
        assert report.total_files == 0
        assert report.total_functions == 0
        assert report.files == []

    def test_single_file_single_function(self):
        source = "def process(x):\n    if x > 0:\n        return x\n    return 0\n"
        reader = FakeSourceCodeReader({"svc.py": source})
        report = analyze_complexity(reader, "/repo")
        assert report.total_files == 1
        assert report.total_functions == 1
        assert report.max_complexity == 2
        assert report.files[0].worst_function == "process"

    def test_multiple_files_aggregated(self):
        simple = "def a():\n    pass\n"
        complex_src = "def b(x):\n    if x:\n        for i in x:\n            print(i)\n"
        reader = FakeSourceCodeReader({"simple.py": simple, "complex.py": complex_src})
        report = analyze_complexity(reader, "/repo")
        assert report.total_files == 2
        assert report.total_functions == 2

    def test_files_sorted_by_max_complexity_desc(self):
        simple = "def a():\n    pass\n"
        complex_src = (
            "def b(x, y, z):\n"
            "    if x:\n"
            "        if y:\n"
            "            if z:\n"
            "                return 1\n"
            "    return 0\n"
        )
        reader = FakeSourceCodeReader({"simple.py": simple, "complex.py": complex_src})
        report = analyze_complexity(reader, "/repo")
        assert report.files[0].file_path == "complex.py"
        assert report.files[1].file_path == "simple.py"

    def test_high_complexity_count_uses_threshold(self):
        # CC=1 (below threshold 2)
        simple = "def a():\n    pass\n"
        # CC=3 (above threshold 2)
        complex_src = "def b(x, y):\n    if x:\n        if y:\n            return 1\n    return 0\n"
        reader = FakeSourceCodeReader({"simple.py": simple, "complex.py": complex_src})
        report = analyze_complexity(reader, "/repo", complexity_threshold=2)
        assert report.high_complexity_count == 1

    def test_ref_passed_to_source_reader(self):
        source = "def f(): pass\n"
        reader = FakeSourceCodeReader({"a.py": source})
        report = analyze_complexity(reader, "/repo", ref="v1.0")
        assert report.ref == "v1.0"
        assert report.total_functions == 1

    def test_custom_threshold(self):
        source = "def f(): pass\n"
        reader = FakeSourceCodeReader({"a.py": source})
        report = analyze_complexity(reader, "/repo", complexity_threshold=5)
        assert report.complexity_threshold == 5

    def test_file_not_found_skipped(self):
        """FakeSourceCodeReader lists a file but read raises FileNotFoundError."""
        class BrokenReader:
            def list_python_files(self, ref=None):
                return ["exists.py", "missing.py"]
            def read_file(self, file_path, ref=None):
                if file_path == "missing.py":
                    raise FileNotFoundError("not found")
                return "def f(): pass\n"

        report = analyze_complexity(BrokenReader(), "/repo")
        assert report.total_files == 1
        assert report.total_functions == 1


class TestAnalyzeChangeClusters:
    def test_empty_changes_returns_empty_report(self):
        repo = FakeGitRepository(file_changes_val=[])
        report = analyze_change_clusters(repo, "/repo", 90, current_time=NOW)
        assert report.total_commits == 0
        assert report.k == 0
        assert report.clusters == []
        assert report.drift == []

    def test_single_commit_returns_single_cluster(self):
        changes = [_make_change("a.py", "c1", days_ago=5, added=10, deleted=5)]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_change_clusters(repo, "/repo", 90, current_time=NOW)
        assert report.total_commits == 1
        assert report.k == 1
        assert len(report.clusters) == 1
        assert report.clusters[0].size == 1

    def test_distinct_patterns_form_clusters(self):
        """Feature-like (high add, high churn, many files) and bugfix-like
        (low churn, few files) should form separate clusters."""
        changes = []
        # 5 feature-like commits: many files, high add churn
        for i in range(5):
            for f in ["a.py", "b.py", "c.py", "d.py", "e.py"]:
                changes.append(_make_change(f, f"feat{i}", days_ago=i + 1, added=50, deleted=5))
        # 5 bugfix-like commits: 1 file, low churn
        for i in range(5):
            changes.append(_make_change("fix.py", f"fix{i}", days_ago=i + 10, added=2, deleted=1))
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_change_clusters(repo, "/repo", 90, current_time=NOW)
        assert report.total_commits == 10
        assert report.k >= 2
        assert len(report.clusters) >= 2

    def test_window_filtering_applied(self):
        changes = [
            _make_change("a.py", "c1", days_ago=5, added=10, deleted=5),    # inside 30d
            _make_change("b.py", "c2", days_ago=60, added=10, deleted=5),   # outside 30d
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_change_clusters(repo, "/repo", 30, current_time=NOW)
        assert report.total_commits == 1

    def test_report_metadata_correct(self):
        changes = [
            _make_change("a.py", "c1", days_ago=5),
            _make_change("b.py", "c2", days_ago=10),
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_change_clusters(repo, "/repo", 90, current_time=NOW)
        assert report.repo_path == "/repo"
        assert report.window_days == 90
        assert report.to_date == NOW
        assert report.from_date == NOW - timedelta(days=90)
        assert report.total_commits == 2


class TestAnalyzeEffort:
    # --- Empty / fallback ---

    def test_empty_changes_returns_empty_report(self):
        repo = FakeGitRepository(file_changes_val=[])
        report = analyze_effort(repo, "/repo", 90, current_time=NOW)
        assert report.total_files == 0
        assert report.model_r_squared == 0.0
        assert report.files == []

    def test_single_file_uses_fallback(self):
        changes = [_make_change("a.py", "c1", days_ago=5)]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_effort(repo, "/repo", 90, current_time=NOW)
        assert report.total_files == 1
        assert report.model_r_squared == 0.0
        # Fallback: equal weights (1/6 each for 6 features)
        assert len(report.coefficients) == 6
        assert all(abs(c - 1.0 / 6) < 1e-3 for c in report.coefficients)

    def test_two_files_uses_fallback(self):
        changes = [
            _make_change("a.py", "c1", days_ago=5, added=10, deleted=5),
            _make_change("b.py", "c2", days_ago=3, added=20, deleted=10),
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_effort(repo, "/repo", 90, current_time=NOW)
        assert report.total_files == 2
        assert report.model_r_squared == 0.0

    def test_report_metadata_effort(self):
        changes = [_make_change("a.py", "c1", days_ago=5)]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_effort(repo, "/repo", 90, current_time=NOW)
        assert report.repo_path == "/repo"
        assert report.window_days == 90
        assert report.to_date == NOW
        assert report.from_date == NOW - timedelta(days=90)

    def test_current_time_default_works(self):
        changes = [_make_change("a.py", "c1", days_ago=5)]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_effort(repo, "/repo", 90)
        assert report.total_files >= 0  # just verifies no exception

    # --- Full model (3+ files) ---

    def test_three_files_trains_model(self):
        changes = [
            _make_change("a.py", "c1", days_ago=5, added=100, deleted=50),
            _make_change("a.py", "c2", days_ago=10, added=80, deleted=40),
            _make_change("a.py", "c3", days_ago=15, added=60, deleted=30),
            _make_change("b.py", "c1", days_ago=5, added=20, deleted=10),
            _make_change("b.py", "c4", days_ago=20, added=15, deleted=5),
            _make_change("c.py", "c5", days_ago=3, added=5, deleted=2),
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_effort(repo, "/repo", 90, current_time=NOW)
        assert report.total_files == 3
        # With 3+ files, model should be trained (not equal weights)
        assert len(report.coefficients) == 6

    def test_rei_scores_in_zero_one(self):
        changes = [
            _make_change("a.py", "c1", days_ago=5, added=100, deleted=50),
            _make_change("a.py", "c2", days_ago=10, added=80, deleted=40),
            _make_change("b.py", "c1", days_ago=5, added=20, deleted=10),
            _make_change("c.py", "c3", days_ago=3, added=5, deleted=2),
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_effort(repo, "/repo", 90, current_time=NOW)
        for f in report.files:
            assert 0.0 <= f.rei_score <= 1.0

    def test_proxy_labels_in_zero_one(self):
        changes = [
            _make_change("a.py", "c1", days_ago=5, added=100, deleted=50),
            _make_change("a.py", "c2", days_ago=10, added=80, deleted=40),
            _make_change("b.py", "c1", days_ago=5, added=20, deleted=10),
            _make_change("c.py", "c3", days_ago=3, added=5, deleted=2),
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_effort(repo, "/repo", 90, current_time=NOW)
        for f in report.files:
            assert 0.0 <= f.proxy_label <= 1.0

    def test_feature_names_correct(self):
        changes = [
            _make_change("a.py", "c1", days_ago=5, added=100, deleted=50),
            _make_change("b.py", "c2", days_ago=10, added=20, deleted=10),
            _make_change("c.py", "c3", days_ago=3, added=5, deleted=2),
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_effort(repo, "/repo", 90, current_time=NOW)
        assert report.feature_names == [
            "code_churn", "change_frequency", "pain_score",
            "knowledge_concentration", "author_count", "knowledge_x_pain",
        ]

    def test_coefficients_length_six(self):
        changes = [
            _make_change("a.py", "c1", days_ago=5, added=100, deleted=50),
            _make_change("b.py", "c2", days_ago=10, added=20, deleted=10),
            _make_change("c.py", "c3", days_ago=3, added=5, deleted=2),
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_effort(repo, "/repo", 90, current_time=NOW)
        assert len(report.coefficients) == 6

    def test_files_sorted_by_rei_descending(self):
        changes = [
            _make_change("a.py", "c1", days_ago=5, added=100, deleted=50),
            _make_change("a.py", "c2", days_ago=10, added=80, deleted=40),
            _make_change("b.py", "c1", days_ago=5, added=20, deleted=10),
            _make_change("c.py", "c3", days_ago=3, added=5, deleted=2),
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_effort(repo, "/repo", 90, current_time=NOW)
        scores = [f.rei_score for f in report.files]
        assert scores == sorted(scores, reverse=True)

    def test_attributions_per_file(self):
        changes = [
            _make_change("a.py", "c1", days_ago=5, added=100, deleted=50),
            _make_change("b.py", "c2", days_ago=10, added=20, deleted=10),
            _make_change("c.py", "c3", days_ago=3, added=5, deleted=2),
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_effort(repo, "/repo", 90, current_time=NOW)
        for f in report.files:
            assert len(f.attributions) == 6

    def test_attributions_sorted_by_abs_contribution(self):
        changes = [
            _make_change("a.py", "c1", days_ago=5, added=100, deleted=50),
            _make_change("b.py", "c2", days_ago=10, added=20, deleted=10),
            _make_change("c.py", "c3", days_ago=3, added=5, deleted=2),
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_effort(repo, "/repo", 90, current_time=NOW)
        for f in report.files:
            abs_contribs = [abs(a.contribution) for a in f.attributions]
            assert abs_contribs == sorted(abs_contribs, reverse=True)

    # --- Window / time travel ---

    def test_window_filtering_excludes_old_changes(self):
        changes = [
            _make_change("a.py", "c1", days_ago=5, added=10, deleted=5),
            _make_change("b.py", "c2", days_ago=60, added=20, deleted=10),  # outside 30d
            _make_change("c.py", "c3", days_ago=10, added=5, deleted=2),
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_effort(repo, "/repo", 30, current_time=NOW)
        file_paths = [f.file_path for f in report.files]
        assert "b.py" not in file_paths

    def test_current_time_anchors_analysis(self):
        shifted = NOW - timedelta(days=100)
        changes = [
            _make_change("a.py", "c1", days_ago=105, added=10, deleted=5),
            _make_change("b.py", "c2", days_ago=110, added=20, deleted=10),
            _make_change("c.py", "c3", days_ago=95, added=5, deleted=2),
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_effort(repo, "/repo", 30, current_time=shifted)
        assert report.total_files >= 1

    def test_alpha_parameter_passed(self):
        changes = [
            _make_change("a.py", "c1", days_ago=5, added=100, deleted=50),
            _make_change("b.py", "c2", days_ago=10, added=20, deleted=10),
            _make_change("c.py", "c3", days_ago=3, added=5, deleted=2),
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        report_low = analyze_effort(repo, "/repo", 90, current_time=NOW, alpha=0.01)
        report_high = analyze_effort(repo, "/repo", 90, current_time=NOW, alpha=100.0)
        assert report_low.alpha == 0.01
        assert report_high.alpha == 100.0

    def test_alpha_auto_tune_picks_best(self):
        """Default alpha=None → auto-selects from grid search candidates."""
        changes = [
            _make_change("a.py", "c1", days_ago=5, added=100, deleted=50),
            _make_change("a.py", "c2", days_ago=10, added=80, deleted=40),
            _make_change("b.py", "c1", days_ago=5, added=20, deleted=10),
            _make_change("c.py", "c3", days_ago=3, added=5, deleted=2),
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_effort(repo, "/repo", 90, current_time=NOW)
        # Auto-tuned alpha should be from candidates
        assert report.alpha in [0.1, 0.5, 1.0, 2.0, 5.0]

    def test_high_churn_file_gets_high_rei(self):
        """File with extreme churn should get a high REI score."""
        changes = [
            _make_change("hot.py", "c1", days_ago=1, added=500, deleted=200),
            _make_change("hot.py", "c2", days_ago=2, added=400, deleted=150),
            _make_change("hot.py", "c3", days_ago=3, added=300, deleted=100),
            _make_change("med.py", "c1", days_ago=1, added=20, deleted=10),
            _make_change("cold.py", "c4", days_ago=30, added=5, deleted=2),
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        report = analyze_effort(repo, "/repo", 90, current_time=NOW)
        file_map = {f.file_path: f for f in report.files}
        assert file_map["hot.py"].rei_score >= file_map["cold.py"].rei_score


class TestAnalyzeDX:
    def test_empty_repo(self):
        repo = FakeGitRepository(file_changes_val=[])
        source_reader = FakeSourceCodeReader({})
        report = analyze_dx(repo, source_reader, "/repo", 90, current_time=NOW)
        assert report.total_commits == 0
        assert report.total_files == 0
        assert report.dx_score == 0.0
        assert report.metrics.throughput == 0.0
        assert report.metrics.feedback_delay == 0.0
        assert report.metrics.focus_ratio == 0.5
        assert report.metrics.cognitive_load == 0.0

    def test_basic_repo(self):
        changes = [
            _make_change("a.py", "c1", days_ago=5, added=50, deleted=20),
            _make_change("a.py", "c2", days_ago=10, added=30, deleted=10),
            _make_change("b.py", "c1", days_ago=5, added=20, deleted=5),
            _make_change("c.py", "c3", days_ago=15, added=10, deleted=3),
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        source_reader = FakeSourceCodeReader({
            "a.py": "def process(x):\n    if x:\n        return x\n    return 0\n",
            "b.py": "def simple(): pass\n",
            "c.py": "CONST = 1\n",
        })
        report = analyze_dx(repo, source_reader, "/repo", 90, current_time=NOW)
        assert report.total_commits >= 1
        assert report.total_files >= 1
        assert report.dx_score >= 0.0

    def test_metrics_in_range(self):
        changes = [
            _make_change("a.py", "c1", days_ago=5, added=50, deleted=20),
            _make_change("b.py", "c2", days_ago=10, added=30, deleted=10),
            _make_change("c.py", "c3", days_ago=15, added=10, deleted=3),
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        source_reader = FakeSourceCodeReader({
            "a.py": "def f(): pass\n",
        })
        report = analyze_dx(repo, source_reader, "/repo", 90, current_time=NOW)
        assert 0.0 <= report.metrics.throughput <= 1.0
        assert 0.0 <= report.metrics.feedback_delay <= 1.0
        assert 0.0 <= report.metrics.focus_ratio <= 1.0
        assert 0.0 <= report.metrics.cognitive_load <= 1.0

    def test_dx_score_in_range(self):
        changes = [
            _make_change("a.py", "c1", days_ago=5, added=50, deleted=20),
            _make_change("b.py", "c2", days_ago=10, added=30, deleted=10),
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        source_reader = FakeSourceCodeReader({})
        report = analyze_dx(repo, source_reader, "/repo", 90, current_time=NOW)
        assert 0.0 <= report.dx_score <= 1.0

    def test_custom_weights(self):
        changes = [
            _make_change("a.py", "c1", days_ago=5, added=50, deleted=20),
            _make_change("b.py", "c2", days_ago=10, added=30, deleted=10),
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        source_reader = FakeSourceCodeReader({})
        custom_weights = [0.4, 0.2, 0.2, 0.2]
        report = analyze_dx(
            repo, source_reader, "/repo", 90,
            current_time=NOW, weights=custom_weights,
        )
        assert report.weights == custom_weights

    def test_time_travel(self):
        shifted = NOW - timedelta(days=100)
        changes = [
            _make_change("a.py", "c1", days_ago=105, added=50, deleted=20),
            _make_change("b.py", "c2", days_ago=110, added=30, deleted=10),
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        source_reader = FakeSourceCodeReader({})
        report = analyze_dx(
            repo, source_reader, "/repo", 30, current_time=shifted,
        )
        assert report.total_commits >= 1

    def test_cognitive_files_sorted(self):
        changes = [
            _make_change("hot.py", "c1", days_ago=1, added=200, deleted=100),
            _make_change("hot.py", "c2", days_ago=2, added=150, deleted=80),
            _make_change("cold.py", "c3", days_ago=30, added=5, deleted=2),
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        source_reader = FakeSourceCodeReader({})
        report = analyze_dx(repo, source_reader, "/repo", 90, current_time=NOW)
        if len(report.cognitive_load_files) >= 2:
            loads = [f.composite_load for f in report.cognitive_load_files]
            assert loads == sorted(loads, reverse=True)

    def test_report_metadata(self):
        changes = [
            _make_change("a.py", "c1", days_ago=5, added=50, deleted=20),
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        source_reader = FakeSourceCodeReader({})
        report = analyze_dx(repo, source_reader, "/repo", 90, current_time=NOW)
        assert report.repo_path == "/repo"
        assert report.window_days == 90
        assert report.to_date == NOW
        assert report.from_date == NOW - timedelta(days=90)

    def test_high_throughput_scenario(self):
        # Many feature-like commits (high add ratio, many files)
        changes = []
        for i in range(20):
            for f in ["a.py", "b.py", "c.py"]:
                changes.append(_make_change(f, f"feat{i}", days_ago=i + 1, added=50, deleted=5))
        repo = FakeGitRepository(file_changes_val=changes)
        source_reader = FakeSourceCodeReader({})
        report = analyze_dx(repo, source_reader, "/repo", 90, current_time=NOW)
        # With many feature commits, throughput should be reasonable
        assert report.metrics.throughput > 0.0

    def test_window_filtering(self):
        changes = [
            _make_change("a.py", "c1", days_ago=5, added=50, deleted=20),
            _make_change("b.py", "c2", days_ago=60, added=30, deleted=10),  # outside 30d
        ]
        repo = FakeGitRepository(file_changes_val=changes)
        source_reader = FakeSourceCodeReader({})
        report = analyze_dx(repo, source_reader, "/repo", 30, current_time=NOW)
        assert report.total_commits == 1
