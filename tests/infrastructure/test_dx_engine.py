from datetime import datetime, timezone

from git_xrays.domain.models import (
    ClusterSummary,
    CommitFeatures,
    ComplexityReport,
    CouplingReport,
    FileComplexity,
    FileKnowledge,
    FileMetrics,
    FilePain,
    FunctionComplexity,
    KnowledgeReport,
)
from git_xrays.infrastructure.dx_engine import (
    compute_cognitive_load_per_file,
    compute_dx_score,
    compute_feedback_delay,
    compute_focus_ratio,
    compute_throughput,
)


def _make_cluster(label: str, size: int) -> ClusterSummary:
    return ClusterSummary(
        cluster_id=0, label=label, size=size,
        centroid_file_count=1.0, centroid_total_churn=10.0,
        centroid_add_ratio=0.5, commits=[],
    )


class TestComputeThroughput:
    def test_all_feature_commits(self):
        clusters = [_make_cluster("feature", 10)]
        result = compute_throughput(clusters, window_days=90)
        # 10 * 1.0 / (90 * 10) = 10/900 ≈ 0.0111
        assert 0.0 < result <= 1.0

    def test_all_bugfix_commits(self):
        clusters = [_make_cluster("bugfix", 10)]
        result = compute_throughput(clusters, window_days=90)
        # bugfix weight=0.5, so 10*0.5/(90*10) = 5/900
        assert result < compute_throughput(
            [_make_cluster("feature", 10)], window_days=90
        )

    def test_mixed_clusters(self):
        clusters = [
            _make_cluster("feature", 5),
            _make_cluster("bugfix", 5),
        ]
        result = compute_throughput(clusters, window_days=90)
        # (5*1.0 + 5*0.5) / (90*10) = 7.5/900
        assert 0.0 < result <= 1.0

    def test_empty_clusters(self):
        result = compute_throughput([], window_days=90)
        assert result == 0.0

    def test_high_commit_rate_clamps_to_one(self):
        clusters = [_make_cluster("feature", 10000)]
        result = compute_throughput(clusters, window_days=1)
        assert result == 1.0

    def test_low_commit_rate(self):
        clusters = [_make_cluster("feature", 1)]
        result = compute_throughput(clusters, window_days=90)
        assert result < 0.1

    def test_refactoring_weighted(self):
        refactor = compute_throughput(
            [_make_cluster("refactoring", 10)], window_days=90
        )
        feature = compute_throughput(
            [_make_cluster("feature", 10)], window_days=90
        )
        # refactoring=0.8 < feature=1.0
        assert refactor < feature

    def test_config_weighted(self):
        config = compute_throughput(
            [_make_cluster("config", 10)], window_days=90
        )
        bugfix = compute_throughput(
            [_make_cluster("bugfix", 10)], window_days=90
        )
        # config=0.3 < bugfix=0.5
        assert config < bugfix


class TestComputeFeedbackDelay:
    def test_high_density_low_rework(self):
        densities = [0.9, 0.8]
        rework_ratios = [0.1, 0.1]
        result = compute_feedback_delay(densities, rework_ratios)
        # mean(densities) * (1 - mean(rework)) = 0.85 * 0.9 = 0.765
        assert abs(result - 0.765) < 0.001

    def test_low_density_high_rework(self):
        densities = [0.1, 0.2]
        rework_ratios = [0.9, 0.8]
        result = compute_feedback_delay(densities, rework_ratios)
        # mean(0.1, 0.2) * (1 - mean(0.9, 0.8)) = 0.15 * 0.15 = 0.0225
        assert abs(result - 0.0225) < 0.001

    def test_empty_files(self):
        result = compute_feedback_delay([], [])
        assert result == 0.0

    def test_single_file_zero_rework(self):
        result = compute_feedback_delay([0.5], [0.0])
        # 0.5 * (1 - 0.0) = 0.5
        assert abs(result - 0.5) < 0.001

    def test_multiple_files_average(self):
        densities = [0.2, 0.4, 0.6]
        rework_ratios = [0.1, 0.2, 0.3]
        result = compute_feedback_delay(densities, rework_ratios)
        # mean(densities) = 0.4, mean(rework) = 0.2
        # 0.4 * (1 - 0.2) = 0.32
        assert abs(result - 0.32) < 0.001

    def test_all_zero_density(self):
        result = compute_feedback_delay([0.0, 0.0], [0.5, 0.5])
        assert result == 0.0

    def test_all_one_density_zero_rework(self):
        result = compute_feedback_delay([1.0, 1.0], [0.0, 0.0])
        assert abs(result - 1.0) < 0.001


class TestComputeFocusRatio:
    def test_all_feature(self):
        clusters = [_make_cluster("feature", 10)]
        result = compute_focus_ratio(clusters)
        assert result == 1.0

    def test_all_bugfix(self):
        clusters = [_make_cluster("bugfix", 10)]
        result = compute_focus_ratio(clusters)
        assert result == 0.0

    def test_equal_feature_bugfix(self):
        clusters = [
            _make_cluster("feature", 5),
            _make_cluster("bugfix", 5),
        ]
        result = compute_focus_ratio(clusters)
        assert abs(result - 0.5) < 0.001

    def test_feature_and_config(self):
        clusters = [
            _make_cluster("feature", 3),
            _make_cluster("config", 7),
        ]
        result = compute_focus_ratio(clusters)
        # 3 / (3 + 7) = 0.3
        assert abs(result - 0.3) < 0.001

    def test_mixed_excluded_from_denominator(self):
        clusters = [
            _make_cluster("feature", 5),
            _make_cluster("bugfix", 5),
            _make_cluster("mixed", 10),
        ]
        result = compute_focus_ratio(clusters)
        # mixed excluded: 5 / (5 + 5) = 0.5
        assert abs(result - 0.5) < 0.001

    def test_empty_clusters(self):
        result = compute_focus_ratio([])
        assert result == 0.5

    def test_refactoring_in_denominator(self):
        clusters = [
            _make_cluster("feature", 4),
            _make_cluster("refactoring", 6),
        ]
        result = compute_focus_ratio(clusters)
        # 4 / (4 + 6) = 0.4
        assert abs(result - 0.4) < 0.001


class TestComputeCognitiveLoadPerFile:
    def test_single_file_all_scores(self):
        hotspot_files = [
            FileMetrics(file_path="a.py", change_frequency=5,
                        code_churn=100, hotspot_score=1.0, rework_ratio=0.8, file_size=0),
        ]
        knowledge_files = [
            FileKnowledge(
                file_path="a.py", knowledge_concentration=0.7,
                primary_author="Alice", primary_author_pct=0.7,
                is_knowledge_island=False, author_count=2, authors=[],
            ),
        ]
        pain_files = [
            FilePain(
                file_path="a.py",
                size_raw=100, size_normalized=1.0,
                volatility_raw=5, volatility_normalized=1.0,
                distance_raw=0.6, distance_normalized=0.75,
                pain_score=0.75,
            ),
        ]
        complexity_files = [
            FileComplexity(
                file_path="a.py", function_count=2,
                total_complexity=10, avg_complexity=5.0,
                max_complexity=7, worst_function="process",
                avg_length=15.0, max_length=20,
                avg_nesting=2.0, max_nesting=3,
                avg_cognitive=0.0, max_cognitive=0, functions=[],
            ),
        ]
        result = compute_cognitive_load_per_file(
            hotspot_files, knowledge_files, pain_files, complexity_files,
        )
        assert len(result) == 1
        assert result[0].file_path == "a.py"
        assert 0.0 <= result[0].composite_load <= 1.0

    def test_normalization(self):
        hotspot_files = [
            FileMetrics(file_path="a.py", change_frequency=10,
                        code_churn=200, hotspot_score=1.0, rework_ratio=0.9, file_size=0),
            FileMetrics(file_path="b.py", change_frequency=1,
                        code_churn=10, hotspot_score=0.1, rework_ratio=0.0, file_size=0),
        ]
        result = compute_cognitive_load_per_file(hotspot_files, [], [], [])
        for f in result:
            assert 0.0 <= f.complexity_score <= 1.0
            assert 0.0 <= f.coordination_score <= 1.0
            assert 0.0 <= f.knowledge_score <= 1.0
            assert 0.0 <= f.change_rate_score <= 1.0

    def test_composite_is_mean(self):
        hotspot_files = [
            FileMetrics(file_path="a.py", change_frequency=5,
                        code_churn=50, hotspot_score=0.5, rework_ratio=0.5, file_size=0),
        ]
        knowledge_files = [
            FileKnowledge(
                file_path="a.py", knowledge_concentration=0.6,
                primary_author="Alice", primary_author_pct=0.6,
                is_knowledge_island=False, author_count=2, authors=[],
            ),
        ]
        result = compute_cognitive_load_per_file(
            hotspot_files, knowledge_files, [], [],
        )
        f = result[0]
        expected = (f.complexity_score + f.coordination_score
                    + f.knowledge_score + f.change_rate_score) / 4.0
        assert abs(f.composite_load - expected) < 0.0001

    def test_sorted_by_composite_desc(self):
        hotspot_files = [
            FileMetrics(file_path="low.py", change_frequency=1,
                        code_churn=5, hotspot_score=0.1, rework_ratio=0.0, file_size=0),
            FileMetrics(file_path="high.py", change_frequency=10,
                        code_churn=200, hotspot_score=1.0, rework_ratio=0.9, file_size=0),
        ]
        result = compute_cognitive_load_per_file(hotspot_files, [], [], [])
        assert result[0].file_path == "high.py"
        assert result[1].file_path == "low.py"
        assert result[0].composite_load >= result[1].composite_load

    def test_missing_complexity(self):
        hotspot_files = [
            FileMetrics(file_path="a.py", change_frequency=5,
                        code_churn=50, hotspot_score=0.5, rework_ratio=0.5, file_size=0),
        ]
        result = compute_cognitive_load_per_file(hotspot_files, [], [], [])
        assert result[0].complexity_score == 0.0

    def test_missing_coupling(self):
        hotspot_files = [
            FileMetrics(file_path="a.py", change_frequency=5,
                        code_churn=50, hotspot_score=0.5, rework_ratio=0.5, file_size=0),
        ]
        result = compute_cognitive_load_per_file(hotspot_files, [], [], [])
        assert result[0].coordination_score == 0.0

    def test_missing_knowledge(self):
        hotspot_files = [
            FileMetrics(file_path="a.py", change_frequency=5,
                        code_churn=50, hotspot_score=0.5, rework_ratio=0.5, file_size=0),
        ]
        result = compute_cognitive_load_per_file(hotspot_files, [], [], [])
        assert result[0].knowledge_score == 0.0

    def test_missing_commit_count(self):
        # A file in knowledge but not in hotspot → change_rate=0
        knowledge_files = [
            FileKnowledge(
                file_path="orphan.py", knowledge_concentration=0.5,
                primary_author="Bob", primary_author_pct=0.5,
                is_knowledge_island=False, author_count=2, authors=[],
            ),
        ]
        result = compute_cognitive_load_per_file([], knowledge_files, [], [])
        # orphan.py should appear with change_rate_score=0
        assert any(f.file_path == "orphan.py" for f in result)
        orphan = [f for f in result if f.file_path == "orphan.py"][0]
        assert orphan.change_rate_score == 0.0

    def test_empty_files(self):
        result = compute_cognitive_load_per_file([], [], [], [])
        assert result == []

    def test_multiple_files_different_loads(self):
        hotspot_files = [
            FileMetrics(file_path="a.py", change_frequency=10,
                        code_churn=200, hotspot_score=1.0, rework_ratio=0.9, file_size=0),
            FileMetrics(file_path="b.py", change_frequency=2,
                        code_churn=20, hotspot_score=0.2, rework_ratio=0.0, file_size=0),
            FileMetrics(file_path="c.py", change_frequency=5,
                        code_churn=80, hotspot_score=0.5, rework_ratio=0.4, file_size=0),
        ]
        knowledge_files = [
            FileKnowledge(
                file_path="a.py", knowledge_concentration=0.9,
                primary_author="Alice", primary_author_pct=0.9,
                is_knowledge_island=True, author_count=1, authors=[],
            ),
            FileKnowledge(
                file_path="b.py", knowledge_concentration=0.1,
                primary_author="Bob", primary_author_pct=0.5,
                is_knowledge_island=False, author_count=3, authors=[],
            ),
        ]
        result = compute_cognitive_load_per_file(
            hotspot_files, knowledge_files, [], [],
        )
        assert len(result) == 3
        loads = [f.composite_load for f in result]
        # Should be sorted descending
        assert loads == sorted(loads, reverse=True)


class TestComputeDXScore:
    def test_default_weights(self):
        throughput = 0.6
        feedback = 0.8
        focus = 0.7
        cognitive = 0.4
        weights = [0.3, 0.25, 0.25, 0.2]
        result = compute_dx_score(throughput, feedback, focus, cognitive, weights)
        expected = (
            0.3 * 0.6 + 0.25 * 0.8 + 0.25 * 0.7 + 0.2 * (1 - 0.4)
        )
        assert abs(result - expected) < 0.0001

    def test_custom_weights(self):
        weights = [0.4, 0.2, 0.2, 0.2]
        result = compute_dx_score(0.5, 0.5, 0.5, 0.5, weights)
        expected = 0.4 * 0.5 + 0.2 * 0.5 + 0.2 * 0.5 + 0.2 * 0.5
        assert abs(result - expected) < 0.0001

    def test_cognitive_load_inverted(self):
        # High cognitive load should reduce score
        low_load = compute_dx_score(0.5, 0.5, 0.5, 0.0, [0.25, 0.25, 0.25, 0.25])
        high_load = compute_dx_score(0.5, 0.5, 0.5, 1.0, [0.25, 0.25, 0.25, 0.25])
        assert low_load > high_load

    def test_all_ones(self):
        result = compute_dx_score(1.0, 1.0, 1.0, 0.0, [0.25, 0.25, 0.25, 0.25])
        assert abs(result - 1.0) < 0.0001

    def test_all_zeros(self):
        result = compute_dx_score(0.0, 0.0, 0.0, 1.0, [0.25, 0.25, 0.25, 0.25])
        assert abs(result - 0.0) < 0.0001
