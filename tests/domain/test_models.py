import dataclasses
from datetime import datetime, timezone

from git_xrays.domain.models import (
    AuthorContribution,
    CouplingPair,
    CouplingReport,
    FileChange,
    FileKnowledge,
    FileMetrics,
    FilePain,
    HotspotReport,
    KnowledgeReport,
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
            author_name="Alice", author_email="alice@example.com",
        )
        assert fc.commit_hash == "abc123"
        assert fc.date == dt
        assert fc.file_path == "src/main.py"
        assert fc.lines_added == 10
        assert fc.lines_deleted == 3
        assert fc.author_name == "Alice"
        assert fc.author_email == "alice@example.com"

    def test_frozen(self):
        fc = FileChange(
            commit_hash="abc", date=datetime.now(timezone.utc),
            file_path="f.py", lines_added=1, lines_deleted=0,
            author_name="Alice", author_email="alice@example.com",
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


class TestAuthorContribution:
    def test_creation(self):
        ac = AuthorContribution(
            author_name="Alice", author_email="alice@example.com",
            change_count=5, total_churn=100, proportion=0.6,
            weighted_proportion=0.7,
        )
        assert ac.author_name == "Alice"
        assert ac.author_email == "alice@example.com"
        assert ac.change_count == 5
        assert ac.total_churn == 100
        assert ac.proportion == 0.6
        assert ac.weighted_proportion == 0.7

    def test_frozen(self):
        ac = AuthorContribution(
            author_name="Alice", author_email="alice@example.com",
            change_count=1, total_churn=10, proportion=1.0,
            weighted_proportion=1.0,
        )
        with __import__("pytest").raises(dataclasses.FrozenInstanceError):
            ac.proportion = 0.5  # type: ignore[misc]


class TestFileKnowledge:
    def test_creation(self):
        ac = AuthorContribution(
            author_name="Alice", author_email="alice@example.com",
            change_count=3, total_churn=60, proportion=0.8,
            weighted_proportion=0.85,
        )
        fk = FileKnowledge(
            file_path="main.py", knowledge_concentration=0.75,
            primary_author="Alice", primary_author_pct=0.8,
            is_knowledge_island=True, author_count=2, authors=[ac],
        )
        assert fk.file_path == "main.py"
        assert fk.knowledge_concentration == 0.75
        assert fk.primary_author == "Alice"
        assert fk.primary_author_pct == 0.8
        assert fk.is_knowledge_island is True
        assert fk.author_count == 2
        assert len(fk.authors) == 1

    def test_frozen(self):
        fk = FileKnowledge(
            file_path="a.py", knowledge_concentration=0.5,
            primary_author="Bob", primary_author_pct=0.5,
            is_knowledge_island=False, author_count=2, authors=[],
        )
        with __import__("pytest").raises(dataclasses.FrozenInstanceError):
            fk.knowledge_concentration = 0.0  # type: ignore[misc]


class TestKnowledgeReport:
    def test_creation(self):
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        report = KnowledgeReport(
            repo_path="/repo", window_days=90,
            from_date=dt, to_date=dt,
            total_commits=10, developer_risk_index=2,
            knowledge_island_count=1, files=[],
        )
        assert report.repo_path == "/repo"
        assert report.window_days == 90
        assert report.total_commits == 10
        assert report.developer_risk_index == 2
        assert report.knowledge_island_count == 1
        assert report.files == []

    def test_frozen(self):
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        report = KnowledgeReport(
            repo_path="/repo", window_days=90,
            from_date=dt, to_date=dt,
            total_commits=0, developer_risk_index=0,
            knowledge_island_count=0, files=[],
        )
        with __import__("pytest").raises(dataclasses.FrozenInstanceError):
            report.developer_risk_index = 5  # type: ignore[misc]


class TestCouplingPair:
    def test_creation(self):
        cp = CouplingPair(
            file_a="a.py", file_b="b.py",
            shared_commits=3, total_commits=10,
            coupling_strength=0.75, support=0.3,
        )
        assert cp.file_a == "a.py"
        assert cp.file_b == "b.py"
        assert cp.shared_commits == 3
        assert cp.total_commits == 10
        assert cp.coupling_strength == 0.75
        assert cp.support == 0.3

    def test_frozen(self):
        cp = CouplingPair(
            file_a="a.py", file_b="b.py",
            shared_commits=1, total_commits=5,
            coupling_strength=0.5, support=0.2,
        )
        with __import__("pytest").raises(dataclasses.FrozenInstanceError):
            cp.coupling_strength = 0.0  # type: ignore[misc]

    def test_perfect_coupling(self):
        cp = CouplingPair(
            file_a="x.py", file_b="y.py",
            shared_commits=5, total_commits=5,
            coupling_strength=1.0, support=1.0,
        )
        assert cp.coupling_strength == 1.0
        assert cp.support == 1.0


class TestFilePain:
    def test_creation(self):
        fp = FilePain(
            file_path="a.py",
            size_raw=100, size_normalized=0.8,
            volatility_raw=5, volatility_normalized=0.5,
            distance_raw=0.6, distance_normalized=0.75,
            pain_score=0.3,
        )
        assert fp.file_path == "a.py"
        assert fp.size_raw == 100
        assert fp.size_normalized == 0.8
        assert fp.volatility_raw == 5
        assert fp.volatility_normalized == 0.5
        assert fp.distance_raw == 0.6
        assert fp.distance_normalized == 0.75
        assert fp.pain_score == 0.3

    def test_frozen(self):
        fp = FilePain(
            file_path="a.py",
            size_raw=10, size_normalized=1.0,
            volatility_raw=1, volatility_normalized=1.0,
            distance_raw=0.5, distance_normalized=1.0,
            pain_score=1.0,
        )
        with __import__("pytest").raises(dataclasses.FrozenInstanceError):
            fp.pain_score = 0.0  # type: ignore[misc]

    def test_zero_pain_when_dimension_zero(self):
        fp = FilePain(
            file_path="a.py",
            size_raw=100, size_normalized=1.0,
            volatility_raw=5, volatility_normalized=1.0,
            distance_raw=0.0, distance_normalized=0.0,
            pain_score=0.0,
        )
        assert fp.pain_score == 0.0


class TestCouplingReport:
    def test_creation(self):
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        report = CouplingReport(
            repo_path="/repo", window_days=90,
            from_date=dt, to_date=dt,
            total_commits=10,
            coupling_pairs=[], file_pain=[],
        )
        assert report.repo_path == "/repo"
        assert report.window_days == 90
        assert report.total_commits == 10
        assert report.coupling_pairs == []
        assert report.file_pain == []

    def test_frozen(self):
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        report = CouplingReport(
            repo_path="/repo", window_days=90,
            from_date=dt, to_date=dt,
            total_commits=0,
            coupling_pairs=[], file_pain=[],
        )
        with __import__("pytest").raises(dataclasses.FrozenInstanceError):
            report.total_commits = 5  # type: ignore[misc]

    def test_with_pairs_and_pain(self):
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        pair = CouplingPair(
            file_a="a.py", file_b="b.py",
            shared_commits=3, total_commits=5,
            coupling_strength=0.75, support=0.6,
        )
        pain = FilePain(
            file_path="a.py",
            size_raw=100, size_normalized=1.0,
            volatility_raw=5, volatility_normalized=1.0,
            distance_raw=0.75, distance_normalized=1.0,
            pain_score=1.0,
        )
        report = CouplingReport(
            repo_path="/repo", window_days=90,
            from_date=dt, to_date=dt,
            total_commits=5,
            coupling_pairs=[pair], file_pain=[pain],
        )
        assert len(report.coupling_pairs) == 1
        assert len(report.file_pain) == 1
        assert report.coupling_pairs[0].coupling_strength == 0.75
        assert report.file_pain[0].pain_score == 1.0
