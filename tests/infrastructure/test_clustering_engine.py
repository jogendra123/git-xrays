from datetime import datetime, timedelta, timezone

import pytest

from git_xrays.domain.models import CommitFeatures, FileChange
from git_xrays.infrastructure.clustering_engine import (
    _kmeans_plus_plus_init,
    auto_select_k,
    compute_cluster_drift,
    extract_commit_features,
    kmeans,
    label_cluster,
    min_max_normalize,
    silhouette_score,
)

NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def _fc(
    commit_hash: str, file_path: str,
    added: int = 10, deleted: int = 5, days_ago: int = 0,
) -> FileChange:
    return FileChange(
        commit_hash=commit_hash,
        date=NOW - timedelta(days=days_ago),
        file_path=file_path,
        lines_added=added,
        lines_deleted=deleted,
        author_name="Test",
        author_email="test@example.com",
    )


# --- Step 2: extract_commit_features ---

class TestExtractCommitFeatures:
    def test_single_commit_single_file(self):
        changes = [_fc("c1", "a.py", added=10, deleted=5)]
        features = extract_commit_features(changes)
        assert len(features) == 1
        assert features[0].commit_hash == "c1"
        assert features[0].file_count == 1
        assert features[0].total_churn == 15
        assert features[0].add_ratio == pytest.approx(10 / 15, abs=1e-3)

    def test_single_commit_multiple_files(self):
        changes = [
            _fc("c1", "a.py", added=10, deleted=5),
            _fc("c1", "b.py", added=20, deleted=10),
        ]
        features = extract_commit_features(changes)
        assert len(features) == 1
        assert features[0].file_count == 2
        assert features[0].total_churn == 45

    def test_multiple_commits_grouped_by_hash(self):
        changes = [
            _fc("c1", "a.py", added=10, deleted=5),
            _fc("c2", "b.py", added=20, deleted=10),
        ]
        features = extract_commit_features(changes)
        assert len(features) == 2
        hashes = {f.commit_hash for f in features}
        assert hashes == {"c1", "c2"}

    def test_add_ratio_pure_addition(self):
        changes = [_fc("c1", "a.py", added=10, deleted=0)]
        features = extract_commit_features(changes)
        assert features[0].add_ratio == pytest.approx(1.0)

    def test_add_ratio_pure_deletion(self):
        changes = [_fc("c1", "a.py", added=0, deleted=10)]
        features = extract_commit_features(changes)
        assert features[0].add_ratio == pytest.approx(0.0)

    def test_add_ratio_balanced(self):
        changes = [_fc("c1", "a.py", added=10, deleted=10)]
        features = extract_commit_features(changes)
        assert features[0].add_ratio == pytest.approx(0.5)

    def test_zero_churn_add_ratio_zero(self):
        changes = [_fc("c1", "a.py", added=0, deleted=0)]
        features = extract_commit_features(changes)
        assert features[0].add_ratio == 0.0

    def test_empty_changes_returns_empty(self):
        features = extract_commit_features([])
        assert features == []


# --- Step 3: min_max_normalize ---

class TestMinMaxNormalize:
    def test_basic_normalization(self):
        data = [[0, 0], [10, 20]]
        result = min_max_normalize(data)
        assert result == [[0.0, 0.0], [1.0, 1.0]]

    def test_midpoint_values(self):
        data = [[0], [5], [10]]
        result = min_max_normalize(data)
        assert result[0] == [0.0]
        assert result[1] == [pytest.approx(0.5)]
        assert result[2] == [1.0]

    def test_constant_dimension_becomes_zero(self):
        data = [[5, 0], [5, 10]]
        result = min_max_normalize(data)
        assert result[0][0] == 0.0
        assert result[1][0] == 0.0

    def test_single_point_all_zeros(self):
        data = [[3, 7, 1]]
        result = min_max_normalize(data)
        assert result == [[0.0, 0.0, 0.0]]

    def test_empty_input_returns_empty(self):
        result = min_max_normalize([])
        assert result == []


# --- K-Means++ Init ---

class TestKMeansPlusPlusInit:
    def test_basic_two_groups(self):
        """Two well-separated groups → centroids end up in different groups."""
        import random
        points = [[0, 0], [1, 1], [100, 100], [101, 101]]
        rng = random.Random(42)
        centroids = _kmeans_plus_plus_init(points, k=2, rng=rng)
        assert len(centroids) == 2
        # Check centroids are in different groups
        group_a = {tuple(p) for p in points[:2]}
        group_b = {tuple(p) for p in points[2:]}
        c_tuples = [tuple(c) for c in centroids]
        in_a = sum(1 for c in c_tuples if c in group_a)
        in_b = sum(1 for c in c_tuples if c in group_b)
        assert in_a >= 1 and in_b >= 1

    def test_deterministic_with_same_seed(self):
        """Same seed → same centroids."""
        import random
        points = [[0, 0], [5, 5], [10, 10], [15, 15]]
        c1 = _kmeans_plus_plus_init(points, k=2, rng=random.Random(42))
        c2 = _kmeans_plus_plus_init(points, k=2, rng=random.Random(42))
        assert c1 == c2

    def test_k_greater_than_n(self):
        """k=5, n=3 → 5 centroids with padding."""
        import random
        points = [[0, 0], [5, 5], [10, 10]]
        rng = random.Random(42)
        centroids = _kmeans_plus_plus_init(points, k=5, rng=rng)
        assert len(centroids) == 5


# --- Step 4: K-Means ---

class TestKMeans:
    def test_two_clear_clusters(self):
        points = [[0, 0], [0, 1], [10, 10], [10, 11]]
        centroids, assignments = kmeans(points, k=2, seed=42)
        # Points 0,1 should be in same cluster, 2,3 in same cluster
        assert assignments[0] == assignments[1]
        assert assignments[2] == assignments[3]
        assert assignments[0] != assignments[2]

    def test_k_equals_one_all_same_cluster(self):
        points = [[1, 2], [3, 4], [5, 6]]
        centroids, assignments = kmeans(points, k=1, seed=42)
        assert all(a == 0 for a in assignments)

    def test_k_equals_n_each_own_cluster(self):
        points = [[0, 0], [100, 100], [200, 200]]
        centroids, assignments = kmeans(points, k=3, seed=42)
        assert len(set(assignments)) == 3

    def test_returns_correct_number_of_centroids(self):
        points = [[1], [2], [3], [4], [5]]
        centroids, assignments = kmeans(points, k=3, seed=42)
        assert len(centroids) == 3

    def test_assignments_length_matches_points(self):
        points = [[1], [2], [3], [4]]
        centroids, assignments = kmeans(points, k=2, seed=42)
        assert len(assignments) == 4

    def test_deterministic_with_same_seed(self):
        points = [[0, 0], [1, 1], [10, 10], [11, 11]]
        _, a1 = kmeans(points, k=2, seed=42)
        _, a2 = kmeans(points, k=2, seed=42)
        assert a1 == a2

    def test_different_seed_may_differ(self):
        # Just verify the function accepts seed param and runs
        points = [[0, 0], [1, 1], [10, 10], [11, 11]]
        _, a1 = kmeans(points, k=2, seed=42)
        _, a2 = kmeans(points, k=2, seed=99)
        # Both should produce valid assignments regardless
        assert len(a1) == 4
        assert len(a2) == 4

    def test_single_point_k_one(self):
        points = [[5, 5]]
        centroids, assignments = kmeans(points, k=1, seed=42)
        assert assignments == [0]
        assert centroids[0] == pytest.approx([5.0, 5.0])

    def test_three_clusters_1d(self):
        points = [[0], [1], [50], [51], [100], [101]]
        centroids, assignments = kmeans(points, k=3, seed=42)
        # Points 0,1 in same cluster; 2,3 in same; 4,5 in same
        assert assignments[0] == assignments[1]
        assert assignments[2] == assignments[3]
        assert assignments[4] == assignments[5]
        assert len(set(assignments)) == 3


# --- Step 5: silhouette_score ---

class TestSilhouetteScore:
    def test_perfect_separation(self):
        points = [[0, 0], [0, 1], [100, 100], [100, 101]]
        assignments = [0, 0, 1, 1]
        score = silhouette_score(points, assignments)
        assert score > 0.9

    def test_single_cluster_returns_zero(self):
        points = [[0, 0], [1, 1], [2, 2]]
        assignments = [0, 0, 0]
        score = silhouette_score(points, assignments)
        assert score == 0.0

    def test_two_points_two_clusters(self):
        points = [[0], [10]]
        assignments = [0, 1]
        score = silhouette_score(points, assignments)
        assert score == 0.0

    def test_overlapping_clusters_low_score(self):
        points = [[0], [1], [1], [2]]
        assignments = [0, 0, 1, 1]
        score = silhouette_score(points, assignments)
        assert score < 0.5

    def test_empty_returns_zero(self):
        score = silhouette_score([], [])
        assert score == 0.0

    def test_score_between_minus_one_and_one(self):
        points = [[0, 0], [1, 1], [10, 10], [11, 11]]
        assignments = [0, 0, 1, 1]
        score = silhouette_score(points, assignments)
        assert -1.0 <= score <= 1.0


# --- Step 6: auto_select_k ---

class TestAutoSelectK:
    def test_finds_k_for_clear_clusters(self):
        # 3 well-separated groups with enough points to prevent degeneracy
        # but at varying intra-cluster distances to discourage further splitting
        points = (
            [[0, 0], [1, 0], [0, 1], [0.5, 0.5], [1, 1], [0, 0.5], [0.5, 0]] +
            [[50, 50], [51, 50], [50, 51], [50.5, 50.5], [51, 51], [50, 50.5], [50.5, 50]] +
            [[100, 100], [101, 100], [100, 101], [100.5, 100.5], [101, 101], [100, 100.5], [100.5, 100]]
        )
        k = auto_select_k(points, seed=42)
        assert k == 3

    def test_single_point_returns_k_min(self):
        points = [[5, 5]]
        k = auto_select_k(points, seed=42)
        assert k == 2

    def test_two_identical_points_returns_k_min(self):
        points = [[5, 5], [5, 5]]
        k = auto_select_k(points, seed=42)
        assert k == 2

    def test_respects_k_max(self):
        # Many groups but k_max=4
        points = [[i * 100, 0] for i in range(20)]
        k = auto_select_k(points, k_max=4, seed=42)
        assert k <= 4


# --- Step 7: label_cluster ---

class TestLabelCluster:
    def test_feature_work(self):
        # add_ratio >= 0.6 AND churn >= 0.5
        assert label_cluster(0.5, 0.8, 0.8) == "feature"

    def test_bugfix(self):
        # churn < 0.3 AND file_count < 0.3
        assert label_cluster(0.1, 0.1, 0.7) == "bugfix"

    def test_refactoring(self):
        # file_count >= 0.5 AND 0.3 <= add_ratio < 0.6
        assert label_cluster(0.7, 0.6, 0.45) == "refactoring"

    def test_config(self):
        # churn < 0.3 AND file_count < 0.3 AND add_ratio < 0.6
        assert label_cluster(0.1, 0.1, 0.3) == "config"

    def test_mixed(self):
        # file_count < 0.5, churn >= 0.3, add_ratio in [0.3, 0.6) → no rule matches
        assert label_cluster(0.4, 0.4, 0.5) == "mixed"

    def test_all_zeros(self):
        assert label_cluster(0.0, 0.0, 0.0) == "config"


# --- Step 8: compute_cluster_drift ---

class TestComputeClusterDrift:
    def test_all_first_half_shrinking(self):
        midpoint = NOW - timedelta(days=15)
        commits = [
            CommitFeatures("c1", NOW - timedelta(days=20), 1, 10, 0.5),
            CommitFeatures("c2", NOW - timedelta(days=25), 1, 10, 0.5),
        ]
        assignments = [0, 0]
        labels = {0: "feature"}
        drift = compute_cluster_drift(commits, assignments, labels, midpoint)
        assert len(drift) == 1
        assert drift[0].cluster_label == "feature"
        assert drift[0].first_half_pct == pytest.approx(100.0)
        assert drift[0].second_half_pct == pytest.approx(0.0)
        assert drift[0].trend == "shrinking"

    def test_all_second_half_growing(self):
        midpoint = NOW - timedelta(days=15)
        commits = [
            CommitFeatures("c1", NOW - timedelta(days=5), 1, 10, 0.5),
            CommitFeatures("c2", NOW - timedelta(days=10), 1, 10, 0.5),
        ]
        assignments = [0, 0]
        labels = {0: "feature"}
        drift = compute_cluster_drift(commits, assignments, labels, midpoint)
        assert len(drift) == 1
        assert drift[0].first_half_pct == pytest.approx(0.0)
        assert drift[0].second_half_pct == pytest.approx(100.0)
        assert drift[0].trend == "growing"

    def test_even_split_stable(self):
        midpoint = NOW - timedelta(days=15)
        commits = [
            CommitFeatures("c1", NOW - timedelta(days=20), 1, 10, 0.5),  # first half, cluster 0
            CommitFeatures("c2", NOW - timedelta(days=18), 2, 50, 0.8),  # first half, cluster 1
            CommitFeatures("c3", NOW - timedelta(days=10), 1, 10, 0.5),  # second half, cluster 0
            CommitFeatures("c4", NOW - timedelta(days=8), 2, 50, 0.8),   # second half, cluster 1
        ]
        assignments = [0, 1, 0, 1]
        labels = {0: "bugfix", 1: "feature"}
        drift = compute_cluster_drift(commits, assignments, labels, midpoint)
        label_map = {d.cluster_label: d for d in drift}
        assert label_map["bugfix"].first_half_pct == pytest.approx(50.0)
        assert label_map["bugfix"].second_half_pct == pytest.approx(50.0)
        assert label_map["bugfix"].trend == "stable"
        assert label_map["feature"].trend == "stable"

    def test_multiple_clusters_independent(self):
        midpoint = NOW - timedelta(days=15)
        commits = [
            CommitFeatures("c1", NOW - timedelta(days=20), 1, 10, 0.5),  # first half, cluster 0
            CommitFeatures("c2", NOW - timedelta(days=5), 2, 50, 0.8),   # second half, cluster 1
        ]
        assignments = [0, 1]
        labels = {0: "bugfix", 1: "feature"}
        drift = compute_cluster_drift(commits, assignments, labels, midpoint)
        assert len(drift) == 2
        label_map = {d.cluster_label: d for d in drift}
        assert label_map["bugfix"].trend == "shrinking"
        assert label_map["feature"].trend == "growing"

    def test_empty_commits_returns_empty(self):
        midpoint = NOW
        drift = compute_cluster_drift([], [], {}, midpoint)
        assert drift == []
