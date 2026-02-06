import dataclasses
from datetime import datetime, timezone

from git_xrays.domain.models import (
    AnemiaReport,
    AuthorContribution,
    ClassMetrics,
    ClusterDrift,
    ClusteringReport,
    ClusterSummary,
    CommitFeatures,
    ComparisonReport,
    ComplexityReport,
    CouplingPair,
    CouplingReport,
    EffortReport,
    FeatureAttribution,
    FileAnemia,
    FileChange,
    FileComplexity,
    FileEffort,
    FileHotspotDelta,
    FileKnowledge,
    FileMetrics,
    FilePain,
    FunctionComplexity,
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


class TestFileHotspotDelta:
    def test_creation(self):
        delta = FileHotspotDelta(
            file_path="src/main.py",
            from_score=0.85, to_score=0.92,
            score_delta=0.07,
            from_churn=100, to_churn=150, churn_delta=50,
            from_frequency=5, to_frequency=8, frequency_delta=3,
            status="degraded",
        )
        assert delta.file_path == "src/main.py"
        assert delta.from_score == 0.85
        assert delta.to_score == 0.92
        assert delta.score_delta == 0.07
        assert delta.from_churn == 100
        assert delta.to_churn == 150
        assert delta.churn_delta == 50
        assert delta.from_frequency == 5
        assert delta.to_frequency == 8
        assert delta.frequency_delta == 3
        assert delta.status == "degraded"

    def test_frozen(self):
        delta = FileHotspotDelta(
            file_path="a.py",
            from_score=0.5, to_score=0.5, score_delta=0.0,
            from_churn=10, to_churn=10, churn_delta=0,
            from_frequency=1, to_frequency=1, frequency_delta=0,
            status="unchanged",
        )
        with __import__("pytest").raises(dataclasses.FrozenInstanceError):
            delta.score_delta = 1.0  # type: ignore[misc]

    def test_status_values(self):
        for status in ["unchanged", "improved", "degraded", "new", "removed"]:
            delta = FileHotspotDelta(
                file_path="a.py",
                from_score=0.0, to_score=0.0, score_delta=0.0,
                from_churn=0, to_churn=0, churn_delta=0,
                from_frequency=0, to_frequency=0, frequency_delta=0,
                status=status,
            )
            assert delta.status == status


class TestComparisonReport:
    def test_creation(self):
        dt1 = datetime(2024, 3, 1, tzinfo=timezone.utc)
        dt2 = datetime(2024, 6, 1, tzinfo=timezone.utc)
        delta = FileHotspotDelta(
            file_path="a.py",
            from_score=0.5, to_score=0.8, score_delta=0.3,
            from_churn=50, to_churn=100, churn_delta=50,
            from_frequency=3, to_frequency=5, frequency_delta=2,
            status="degraded",
        )
        report = ComparisonReport(
            repo_path="/repo", from_ref="v1.0", to_ref="v2.0",
            from_date=dt1, to_date=dt2, window_days=90,
            from_total_commits=42, to_total_commits=58,
            files=[delta],
            new_hotspot_count=1, removed_hotspot_count=0,
            improved_count=0, degraded_count=1,
        )
        assert report.repo_path == "/repo"
        assert report.from_ref == "v1.0"
        assert report.to_ref == "v2.0"
        assert report.from_date == dt1
        assert report.to_date == dt2
        assert report.window_days == 90
        assert report.from_total_commits == 42
        assert report.to_total_commits == 58
        assert len(report.files) == 1
        assert report.new_hotspot_count == 1
        assert report.removed_hotspot_count == 0
        assert report.improved_count == 0
        assert report.degraded_count == 1

    def test_frozen(self):
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        report = ComparisonReport(
            repo_path="/repo", from_ref="a", to_ref="b",
            from_date=dt, to_date=dt, window_days=90,
            from_total_commits=0, to_total_commits=0,
            files=[], new_hotspot_count=0, removed_hotspot_count=0,
            improved_count=0, degraded_count=0,
        )
        with __import__("pytest").raises(dataclasses.FrozenInstanceError):
            report.degraded_count = 5  # type: ignore[misc]

    def test_empty_files(self):
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        report = ComparisonReport(
            repo_path="/repo", from_ref="a", to_ref="b",
            from_date=dt, to_date=dt, window_days=90,
            from_total_commits=0, to_total_commits=0,
            files=[], new_hotspot_count=0, removed_hotspot_count=0,
            improved_count=0, degraded_count=0,
        )
        assert report.files == []
        assert report.new_hotspot_count == 0


class TestClassMetrics:
    def test_creation(self):
        cm = ClassMetrics(
            class_name="UserDTO",
            file_path="models.py",
            field_count=5,
            method_count=3,
            behavior_method_count=1,
            dunder_method_count=1,
            property_count=1,
            dbsi=0.8333,
            logic_density=1.0,
            orchestration_pressure=0.0,
            ams=0.0,
        )
        assert cm.class_name == "UserDTO"
        assert cm.file_path == "models.py"
        assert cm.field_count == 5
        assert cm.method_count == 3
        assert cm.behavior_method_count == 1
        assert cm.dunder_method_count == 1
        assert cm.property_count == 1
        assert cm.dbsi == 0.8333
        assert cm.logic_density == 1.0
        assert cm.orchestration_pressure == 0.0
        assert cm.ams == 0.0

    def test_frozen(self):
        cm = ClassMetrics(
            class_name="X", file_path="x.py",
            field_count=1, method_count=0, behavior_method_count=0,
            dunder_method_count=0, property_count=0,
            dbsi=1.0, logic_density=0.0, orchestration_pressure=1.0, ams=1.0,
        )
        with __import__("pytest").raises(dataclasses.FrozenInstanceError):
            cm.field_count = 99  # type: ignore[misc]


class TestFileAnemia:
    def test_creation(self):
        cm = ClassMetrics(
            class_name="Foo", file_path="foo.py",
            field_count=3, method_count=1, behavior_method_count=0,
            dunder_method_count=1, property_count=0,
            dbsi=1.0, logic_density=0.0, orchestration_pressure=1.0, ams=1.0,
        )
        fa = FileAnemia(
            file_path="foo.py",
            class_count=1,
            anemic_class_count=1,
            worst_ams=1.0,
            classes=[cm],
            touch_count=3,
        )
        assert fa.file_path == "foo.py"
        assert fa.class_count == 1
        assert fa.anemic_class_count == 1
        assert fa.worst_ams == 1.0
        assert len(fa.classes) == 1
        assert fa.touch_count == 3

    def test_frozen(self):
        fa = FileAnemia(
            file_path="a.py", class_count=0, anemic_class_count=0,
            worst_ams=0.0, classes=[], touch_count=0,
        )
        with __import__("pytest").raises(dataclasses.FrozenInstanceError):
            fa.class_count = 5  # type: ignore[misc]


class TestAnemiaReport:
    def test_creation(self):
        report = AnemiaReport(
            repo_path="/repo",
            ref=None,
            total_files=10,
            total_classes=25,
            anemic_count=5,
            anemic_percentage=20.0,
            average_ams=0.35,
            ams_threshold=0.5,
            files=[],
        )
        assert report.repo_path == "/repo"
        assert report.ref is None
        assert report.total_files == 10
        assert report.total_classes == 25
        assert report.anemic_count == 5
        assert report.anemic_percentage == 20.0
        assert report.average_ams == 0.35
        assert report.ams_threshold == 0.5
        assert report.files == []


class TestFunctionComplexity:
    def test_creation(self):
        fc = FunctionComplexity(
            function_name="process_data",
            file_path="service.py",
            class_name=None,
            line_number=10,
            length=25,
            cyclomatic_complexity=8,
            max_nesting_depth=3,
            branch_count=4,
            exception_paths=1,
        )
        assert fc.function_name == "process_data"
        assert fc.file_path == "service.py"
        assert fc.class_name is None
        assert fc.line_number == 10
        assert fc.length == 25
        assert fc.cyclomatic_complexity == 8
        assert fc.max_nesting_depth == 3
        assert fc.branch_count == 4
        assert fc.exception_paths == 1

    def test_frozen(self):
        fc = FunctionComplexity(
            function_name="f", file_path="a.py", class_name=None,
            line_number=1, length=1, cyclomatic_complexity=1,
            max_nesting_depth=0, branch_count=0, exception_paths=0,
        )
        with __import__("pytest").raises(dataclasses.FrozenInstanceError):
            fc.cyclomatic_complexity = 99  # type: ignore[misc]

    def test_method_with_class_name(self):
        fc = FunctionComplexity(
            function_name="validate",
            file_path="models.py",
            class_name="User",
            line_number=15,
            length=10,
            cyclomatic_complexity=3,
            max_nesting_depth=1,
            branch_count=2,
            exception_paths=0,
        )
        assert fc.class_name == "User"
        assert fc.function_name == "validate"


class TestFileComplexity:
    def test_creation(self):
        func = FunctionComplexity(
            function_name="process", file_path="svc.py", class_name=None,
            line_number=1, length=10, cyclomatic_complexity=5,
            max_nesting_depth=2, branch_count=3, exception_paths=1,
        )
        fc = FileComplexity(
            file_path="svc.py",
            function_count=1,
            total_complexity=5,
            avg_complexity=5.0,
            max_complexity=5,
            worst_function="process",
            avg_length=10.0,
            max_length=10,
            avg_nesting=2.0,
            max_nesting=2,
            functions=[func],
        )
        assert fc.file_path == "svc.py"
        assert fc.function_count == 1
        assert fc.total_complexity == 5
        assert fc.avg_complexity == 5.0
        assert fc.max_complexity == 5
        assert fc.worst_function == "process"
        assert fc.avg_length == 10.0
        assert fc.max_length == 10
        assert fc.avg_nesting == 2.0
        assert fc.max_nesting == 2
        assert len(fc.functions) == 1

    def test_frozen(self):
        fc = FileComplexity(
            file_path="a.py", function_count=0, total_complexity=0,
            avg_complexity=0.0, max_complexity=0, worst_function="",
            avg_length=0.0, max_length=0, avg_nesting=0.0, max_nesting=0,
            functions=[],
        )
        with __import__("pytest").raises(dataclasses.FrozenInstanceError):
            fc.function_count = 5  # type: ignore[misc]

    def test_empty(self):
        fc = FileComplexity(
            file_path="empty.py", function_count=0, total_complexity=0,
            avg_complexity=0.0, max_complexity=0, worst_function="",
            avg_length=0.0, max_length=0, avg_nesting=0.0, max_nesting=0,
            functions=[],
        )
        assert fc.functions == []
        assert fc.worst_function == ""


class TestComplexityReport:
    def test_creation(self):
        report = ComplexityReport(
            repo_path="/repo",
            ref=None,
            total_files=5,
            total_functions=20,
            avg_complexity=4.5,
            max_complexity=15,
            high_complexity_count=3,
            complexity_threshold=10,
            avg_length=12.0,
            max_length=45,
            files=[],
        )
        assert report.repo_path == "/repo"
        assert report.ref is None
        assert report.total_files == 5
        assert report.total_functions == 20
        assert report.avg_complexity == 4.5
        assert report.max_complexity == 15
        assert report.high_complexity_count == 3
        assert report.complexity_threshold == 10
        assert report.avg_length == 12.0
        assert report.max_length == 45
        assert report.files == []

    def test_frozen(self):
        report = ComplexityReport(
            repo_path="/repo", ref=None, total_files=0, total_functions=0,
            avg_complexity=0.0, max_complexity=0, high_complexity_count=0,
            complexity_threshold=10, avg_length=0.0, max_length=0, files=[],
        )
        with __import__("pytest").raises(dataclasses.FrozenInstanceError):
            report.total_functions = 99  # type: ignore[misc]

    def test_with_ref(self):
        report = ComplexityReport(
            repo_path="/repo", ref="v1.0", total_files=0, total_functions=0,
            avg_complexity=0.0, max_complexity=0, high_complexity_count=0,
            complexity_threshold=10, avg_length=0.0, max_length=0, files=[],
        )
        assert report.ref == "v1.0"


class TestCommitFeatures:
    def test_creation(self):
        dt = datetime(2024, 6, 1, tzinfo=timezone.utc)
        cf = CommitFeatures(
            commit_hash="abc123", date=dt,
            file_count=3, total_churn=150, add_ratio=0.7,
        )
        assert cf.commit_hash == "abc123"
        assert cf.date == dt
        assert cf.file_count == 3
        assert cf.total_churn == 150
        assert cf.add_ratio == 0.7

    def test_frozen(self):
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        cf = CommitFeatures(
            commit_hash="abc", date=dt,
            file_count=1, total_churn=10, add_ratio=0.5,
        )
        with __import__("pytest").raises(dataclasses.FrozenInstanceError):
            cf.total_churn = 99  # type: ignore[misc]


class TestClusterSummary:
    def test_creation(self):
        dt = datetime(2024, 6, 1, tzinfo=timezone.utc)
        commit = CommitFeatures(
            commit_hash="abc", date=dt,
            file_count=2, total_churn=50, add_ratio=0.8,
        )
        cs = ClusterSummary(
            cluster_id=0, label="feature", size=5,
            centroid_file_count=3.0, centroid_total_churn=120.0,
            centroid_add_ratio=0.75, commits=[commit],
        )
        assert cs.cluster_id == 0
        assert cs.label == "feature"
        assert cs.size == 5
        assert cs.centroid_file_count == 3.0
        assert cs.centroid_total_churn == 120.0
        assert cs.centroid_add_ratio == 0.75
        assert len(cs.commits) == 1

    def test_frozen(self):
        cs = ClusterSummary(
            cluster_id=0, label="mixed", size=1,
            centroid_file_count=1.0, centroid_total_churn=10.0,
            centroid_add_ratio=0.5, commits=[],
        )
        with __import__("pytest").raises(dataclasses.FrozenInstanceError):
            cs.size = 99  # type: ignore[misc]


class TestClusterDrift:
    def test_creation(self):
        cd = ClusterDrift(
            cluster_label="feature",
            first_half_pct=60.0, second_half_pct=40.0,
            drift=-20.0, trend="shrinking",
        )
        assert cd.cluster_label == "feature"
        assert cd.first_half_pct == 60.0
        assert cd.second_half_pct == 40.0
        assert cd.drift == -20.0
        assert cd.trend == "shrinking"

    def test_frozen(self):
        cd = ClusterDrift(
            cluster_label="bugfix",
            first_half_pct=50.0, second_half_pct=50.0,
            drift=0.0, trend="stable",
        )
        with __import__("pytest").raises(dataclasses.FrozenInstanceError):
            cd.drift = 10.0  # type: ignore[misc]


class TestClusteringReport:
    def test_creation(self):
        dt1 = datetime(2024, 3, 1, tzinfo=timezone.utc)
        dt2 = datetime(2024, 6, 1, tzinfo=timezone.utc)
        report = ClusteringReport(
            repo_path="/repo", window_days=90,
            from_date=dt1, to_date=dt2,
            total_commits=42, k=3, silhouette_score=0.72,
            clusters=[], drift=[],
        )
        assert report.repo_path == "/repo"
        assert report.window_days == 90
        assert report.from_date == dt1
        assert report.to_date == dt2
        assert report.total_commits == 42
        assert report.k == 3
        assert report.silhouette_score == 0.72
        assert report.clusters == []
        assert report.drift == []

    def test_frozen(self):
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        report = ClusteringReport(
            repo_path="/repo", window_days=90,
            from_date=dt, to_date=dt,
            total_commits=0, k=1, silhouette_score=0.0,
            clusters=[], drift=[],
        )
        with __import__("pytest").raises(dataclasses.FrozenInstanceError):
            report.k = 5  # type: ignore[misc]


class TestFeatureAttribution:
    def test_creation(self):
        fa = FeatureAttribution(
            feature_name="code_churn",
            raw_value=150.0,
            weight=0.42,
            contribution=0.35,
        )
        assert fa.feature_name == "code_churn"
        assert fa.raw_value == 150.0
        assert fa.weight == 0.42
        assert fa.contribution == 0.35

    def test_frozen(self):
        fa = FeatureAttribution(
            feature_name="pain_score", raw_value=0.5,
            weight=0.3, contribution=0.15,
        )
        with __import__("pytest").raises(dataclasses.FrozenInstanceError):
            fa.weight = 0.0  # type: ignore[misc]


class TestFileEffort:
    def test_creation(self):
        attr = FeatureAttribution(
            feature_name="code_churn", raw_value=100.0,
            weight=0.5, contribution=0.25,
        )
        fe = FileEffort(
            file_path="service.py",
            rei_score=0.85,
            proxy_label=0.72,
            commit_density=0.5,
            rework_ratio=0.6,
            attributions=[attr],
        )
        assert fe.file_path == "service.py"
        assert fe.rei_score == 0.85
        assert fe.proxy_label == 0.72
        assert fe.commit_density == 0.5
        assert fe.rework_ratio == 0.6
        assert len(fe.attributions) == 1

    def test_frozen(self):
        fe = FileEffort(
            file_path="a.py", rei_score=0.5, proxy_label=0.3,
            commit_density=0.1, rework_ratio=0.2, attributions=[],
        )
        with __import__("pytest").raises(dataclasses.FrozenInstanceError):
            fe.rei_score = 0.0  # type: ignore[misc]

    def test_empty_attributions(self):
        fe = FileEffort(
            file_path="a.py", rei_score=0.0, proxy_label=0.0,
            commit_density=0.0, rework_ratio=0.0, attributions=[],
        )
        assert fe.attributions == []


class TestEffortReport:
    def test_creation(self):
        dt1 = datetime(2024, 3, 1, tzinfo=timezone.utc)
        dt2 = datetime(2024, 6, 1, tzinfo=timezone.utc)
        report = EffortReport(
            repo_path="/repo", window_days=90,
            from_date=dt1, to_date=dt2,
            total_files=5, model_r_squared=0.72,
            alpha=1.0,
            feature_names=["code_churn", "change_frequency", "pain_score",
                           "knowledge_concentration", "author_count"],
            coefficients=[0.3, 0.2, 0.15, 0.25, 0.1],
            files=[],
        )
        assert report.repo_path == "/repo"
        assert report.window_days == 90
        assert report.total_files == 5
        assert report.model_r_squared == 0.72
        assert report.alpha == 1.0
        assert len(report.feature_names) == 5
        assert len(report.coefficients) == 5
        assert report.files == []

    def test_frozen(self):
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        report = EffortReport(
            repo_path="/repo", window_days=90,
            from_date=dt, to_date=dt,
            total_files=0, model_r_squared=0.0,
            alpha=1.0, feature_names=[], coefficients=[],
            files=[],
        )
        with __import__("pytest").raises(dataclasses.FrozenInstanceError):
            report.model_r_squared = 0.5  # type: ignore[misc]

    def test_empty_files(self):
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        report = EffortReport(
            repo_path="/repo", window_days=90,
            from_date=dt, to_date=dt,
            total_files=0, model_r_squared=0.0,
            alpha=1.0, feature_names=[], coefficients=[],
            files=[],
        )
        assert report.files == []
        assert report.total_files == 0
