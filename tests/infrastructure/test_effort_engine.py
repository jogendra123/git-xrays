from datetime import datetime, timedelta, timezone

import pytest

from git_xrays.domain.models import (
    FileKnowledge,
    FileMetrics,
    FilePain,
)
from git_xrays.infrastructure.effort_engine import (
    build_feature_matrix,
    compute_commit_density,
    compute_effort_proxy,
    compute_rei_scores,
    r_squared,
    ridge_regression,
)


NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


# --- compute_commit_density ---

class TestComputeCommitDensity:
    def test_single_file_single_commit(self):
        """1 commit → no intervals → median=0 → density=1.0."""
        dates = {"a.py": [NOW]}
        result = compute_commit_density(dates)
        assert result["a.py"] == 1.0

    def test_single_file_regular_intervals(self):
        """Commits at days 0,10,20 → intervals=[10,10] → median=10 → density=1/11."""
        base = NOW - timedelta(days=20)
        dates = {"a.py": [base, base + timedelta(days=10), base + timedelta(days=20)]}
        result = compute_commit_density(dates)
        assert result["a.py"] == pytest.approx(1.0 / 11.0)

    def test_single_file_irregular_intervals(self):
        """Commits at days 0,1,30 → intervals=[1,29] → median=15 → density=1/16."""
        base = NOW - timedelta(days=30)
        dates = {"a.py": [base, base + timedelta(days=1), base + timedelta(days=30)]}
        result = compute_commit_density(dates)
        assert result["a.py"] == pytest.approx(1.0 / 16.0)

    def test_multiple_files_independent(self):
        """Two files get different densities based on their own intervals."""
        base = NOW - timedelta(days=20)
        dates = {
            "a.py": [base, base + timedelta(days=10), base + timedelta(days=20)],
            "b.py": [base],
        }
        result = compute_commit_density(dates)
        assert result["a.py"] == pytest.approx(1.0 / 11.0)
        assert result["b.py"] == 1.0

    def test_empty_dict_returns_empty(self):
        result = compute_commit_density({})
        assert result == {}

    def test_high_frequency_high_density(self):
        """Daily commits → intervals all 1 day → density=1/2=0.5."""
        base = NOW - timedelta(days=5)
        dates = {"a.py": [base + timedelta(days=i) for i in range(6)]}
        result = compute_commit_density(dates)
        assert result["a.py"] == pytest.approx(0.5)


# --- compute_effort_proxy ---

class TestComputeEffortProxy:
    def test_equal_density_and_rework(self):
        """Same normalized values → proxy = average."""
        density = {"a.py": 0.5, "b.py": 1.0}
        rework = {"a.py": 0.5, "b.py": 1.0}
        result = compute_effort_proxy(density, rework)
        # a.py: norm_d=0.0, norm_r=0.0 → 0.0; b.py: norm_d=1.0, norm_r=1.0 → 1.0
        assert result["a.py"] == pytest.approx(0.0)
        assert result["b.py"] == pytest.approx(1.0)

    def test_high_density_low_rework(self):
        """Different density/rework produce intermediate proxy."""
        density = {"a.py": 1.0, "b.py": 0.0}
        rework = {"a.py": 0.0, "b.py": 1.0}
        result = compute_effort_proxy(density, rework)
        assert result["a.py"] == pytest.approx(0.5)
        assert result["b.py"] == pytest.approx(0.5)

    def test_single_file_proxy_is_zero(self):
        """Single file → min==max → normalized to 0.0 → proxy=0.0."""
        density = {"a.py": 0.5}
        rework = {"a.py": 0.8}
        result = compute_effort_proxy(density, rework)
        assert result["a.py"] == pytest.approx(0.0)

    def test_empty_returns_empty(self):
        assert compute_effort_proxy({}, {}) == {}

    def test_proxy_values_in_zero_one(self):
        density = {"a.py": 0.1, "b.py": 0.5, "c.py": 0.9}
        rework = {"a.py": 0.2, "b.py": 0.6, "c.py": 0.8}
        result = compute_effort_proxy(density, rework)
        for v in result.values():
            assert 0.0 <= v <= 1.0


# --- build_feature_matrix ---

def _make_hotspot(fp: str, churn: int = 100, freq: int = 5, rework: float = 0.8) -> FileMetrics:
    return FileMetrics(file_path=fp, change_frequency=freq, code_churn=churn,
                       hotspot_score=0.5, rework_ratio=rework, file_size=0)


def _make_knowledge(fp: str, kdi: float = 0.5, author_count: int = 3) -> FileKnowledge:
    return FileKnowledge(file_path=fp, knowledge_concentration=kdi,
                         primary_author="Alice", primary_author_pct=0.6,
                         is_knowledge_island=False, author_count=author_count, authors=[])


def _make_pain(fp: str, pain: float = 0.4) -> FilePain:
    return FilePain(file_path=fp, size_raw=100, size_normalized=0.5,
                    volatility_raw=5, volatility_normalized=0.5,
                    distance_raw=0.3, distance_normalized=0.3, pain_score=pain)


class TestBuildFeatureMatrix:
    def test_all_reports_have_file(self):
        """Correct values extracted from all 3 reports."""
        hotspots = [_make_hotspot("a.py", churn=200, freq=8)]
        knowledge = [_make_knowledge("a.py", kdi=0.7, author_count=4)]
        pain = [_make_pain("a.py", pain=0.6)]
        files, matrix = build_feature_matrix(["a.py"], hotspots, knowledge, pain)
        assert files == ["a.py"]
        assert len(matrix) == 1
        assert matrix[0] == [200, 8, 0.6, 0.7, 4]

    def test_missing_from_pain_uses_defaults(self):
        """File missing from pain → pain_score defaults to 0.0."""
        hotspots = [_make_hotspot("a.py")]
        knowledge = [_make_knowledge("a.py")]
        files, matrix = build_feature_matrix(["a.py"], hotspots, knowledge, [])
        assert matrix[0][2] == 0.0  # pain_score

    def test_missing_from_all_uses_defaults(self):
        """File not in any report → safe defaults."""
        files, matrix = build_feature_matrix(["a.py"], [], [], [])
        assert matrix[0] == [0, 0, 0.0, 0.0, 1]

    def test_feature_order_correct(self):
        """Features: [code_churn, change_frequency, pain_score, knowledge_concentration, author_count]."""
        hotspots = [_make_hotspot("a.py", churn=150, freq=6)]
        knowledge = [_make_knowledge("a.py", kdi=0.3, author_count=2)]
        pain = [_make_pain("a.py", pain=0.5)]
        _, matrix = build_feature_matrix(["a.py"], hotspots, knowledge, pain)
        row = matrix[0]
        assert row[0] == 150    # code_churn
        assert row[1] == 6      # change_frequency
        assert row[2] == 0.5    # pain_score
        assert row[3] == 0.3    # knowledge_concentration
        assert row[4] == 2      # author_count

    def test_multiple_files_rows_match_order(self):
        hotspots = [_make_hotspot("a.py", churn=100), _make_hotspot("b.py", churn=200)]
        knowledge = [_make_knowledge("a.py"), _make_knowledge("b.py")]
        pain = [_make_pain("a.py"), _make_pain("b.py")]
        files, matrix = build_feature_matrix(["a.py", "b.py"], hotspots, knowledge, pain)
        assert files == ["a.py", "b.py"]
        assert len(matrix) == 2
        assert matrix[0][0] == 100  # a.py churn
        assert matrix[1][0] == 200  # b.py churn


# --- ridge_regression ---

class TestRidgeRegression:
    def test_identity_regression(self):
        """X=identity, y=[1,0,0] → coefficients close to expected."""
        X = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        y = [1.0, 0.0, 0.0]
        coeffs = ridge_regression(X, y, alpha=0.01)
        assert len(coeffs) == 3
        assert coeffs[0] > coeffs[1]
        assert coeffs[0] > coeffs[2]

    def test_simple_linear_relationship(self):
        """y ≈ 2*x1 + 3*x2 → coefficients ≈ [2, 3]."""
        X = [[1, 0], [0, 1], [1, 1], [2, 1], [1, 2]]
        y = [2.0, 3.0, 5.0, 7.0, 8.0]
        coeffs = ridge_regression(X, y, alpha=0.001)
        assert coeffs[0] == pytest.approx(2.0, abs=0.1)
        assert coeffs[1] == pytest.approx(3.0, abs=0.1)

    def test_alpha_shrinks_coefficients(self):
        """Higher alpha → smaller magnitude coefficients."""
        X = [[1, 0], [0, 1], [1, 1]]
        y = [2.0, 3.0, 5.0]
        low_alpha = ridge_regression(X, y, alpha=0.01)
        high_alpha = ridge_regression(X, y, alpha=100.0)
        low_mag = sum(abs(c) for c in low_alpha)
        high_mag = sum(abs(c) for c in high_alpha)
        assert high_mag < low_mag

    def test_single_feature(self):
        """1D regression: y = 3*x."""
        X = [[1], [2], [3], [4]]
        y = [3.0, 6.0, 9.0, 12.0]
        coeffs = ridge_regression(X, y, alpha=0.001)
        assert len(coeffs) == 1
        assert coeffs[0] == pytest.approx(3.0, abs=0.1)

    def test_five_features(self):
        """5D regression works (our actual feature count)."""
        X = [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ]
        y = [1.0, 2.0, 3.0, 4.0, 5.0, 15.0]
        coeffs = ridge_regression(X, y, alpha=0.01)
        assert len(coeffs) == 5


# --- r_squared ---

class TestRSquared:
    def test_perfect_fit_r_squared_one(self):
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.0, 2.0, 3.0]
        assert r_squared(y_true, y_pred) == pytest.approx(1.0)

    def test_constant_prediction_r_squared_zero(self):
        """Predicting mean → R²=0."""
        y_true = [1.0, 2.0, 3.0]
        mean = 2.0
        y_pred = [mean, mean, mean]
        assert r_squared(y_true, y_pred) == pytest.approx(0.0)

    def test_constant_y_returns_zero(self):
        """All y same → SS_tot=0 → R²=0.0."""
        y_true = [5.0, 5.0, 5.0]
        y_pred = [5.0, 5.0, 5.0]
        assert r_squared(y_true, y_pred) == pytest.approx(0.0)


# --- compute_rei_scores ---

class TestComputeReiScores:
    def test_single_file_rei_zero(self):
        """Single file → min-max has no range → 0.0."""
        matrix = [[1.0, 2.0, 3.0]]
        coeffs = [1.0, 1.0, 1.0]
        scores = compute_rei_scores(matrix, coeffs)
        assert scores == [0.0]

    def test_two_files_extremes(self):
        """Different dot products → min=0.0, max=1.0."""
        matrix = [[0.0, 0.0], [1.0, 1.0]]
        coeffs = [1.0, 1.0]
        scores = compute_rei_scores(matrix, coeffs)
        assert scores[0] == 0.0
        assert scores[1] == 1.0

    def test_equal_features_equal_rei(self):
        """Identical rows → all 0.0 (no range)."""
        matrix = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
        coeffs = [0.5, 0.5]
        scores = compute_rei_scores(matrix, coeffs)
        assert all(s == 0.0 for s in scores)

    def test_rei_values_in_zero_one(self):
        matrix = [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]
        coeffs = [0.3, 0.7]
        scores = compute_rei_scores(matrix, coeffs)
        for s in scores:
            assert 0.0 <= s <= 1.0
