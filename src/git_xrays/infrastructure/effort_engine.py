"""Pure-Python effort modeling engine — zero external dependencies.

Functions:
- compute_commit_density: per-file commit frequency metric
- compute_effort_proxy: training label for ridge regression
- build_feature_matrix: assemble features from hotspot/knowledge/coupling reports
- ridge_regression: (X^T X + alpha*I)^{-1} X^T y via Gauss-Jordan
- r_squared: goodness-of-fit metric
- compute_rei_scores: dot product + min-max normalize → REI in [0,1]
"""

from __future__ import annotations

from datetime import datetime

from git_xrays.domain.models import FileKnowledge, FileMetrics, FilePain


# ---------------------------------------------------------------------------
# 1. Commit density
# ---------------------------------------------------------------------------

def compute_commit_density(
    file_dates: dict[str, list[datetime]],
) -> dict[str, float]:
    """Compute commit density per file: 1 / (1 + median_interval_days).

    Args:
        file_dates: mapping of file_path → sorted list of commit datetimes.

    Returns:
        mapping of file_path → density in (0, 1].
    """
    result: dict[str, float] = {}
    for fp, dates in file_dates.items():
        if len(dates) <= 1:
            result[fp] = 1.0
            continue
        sorted_dates = sorted(dates)
        intervals = [
            (sorted_dates[i + 1] - sorted_dates[i]).total_seconds() / 86400.0
            for i in range(len(sorted_dates) - 1)
        ]
        intervals.sort()
        n = len(intervals)
        if n % 2 == 1:
            median = intervals[n // 2]
        else:
            median = (intervals[n // 2 - 1] + intervals[n // 2]) / 2.0
        result[fp] = 1.0 / (1.0 + median)
    return result


# ---------------------------------------------------------------------------
# 2. Effort proxy label
# ---------------------------------------------------------------------------

def _min_max_normalize(values: dict[str, float]) -> dict[str, float]:
    """Min-max normalize a dict of floats to [0, 1]."""
    if not values:
        return {}
    vals = list(values.values())
    lo, hi = min(vals), max(vals)
    rng = hi - lo
    if rng == 0:
        return {k: 0.0 for k in values}
    return {k: (v - lo) / rng for k, v in values.items()}


def compute_effort_proxy(
    density: dict[str, float],
    rework: dict[str, float],
) -> dict[str, float]:
    """Effort proxy = 0.5 * norm(density) + 0.5 * norm(rework).

    Returns dict of file_path → proxy in [0, 1].
    """
    if not density:
        return {}
    norm_d = _min_max_normalize(density)
    norm_r = _min_max_normalize(rework)
    return {
        fp: 0.5 * norm_d.get(fp, 0.0) + 0.5 * norm_r.get(fp, 0.0)
        for fp in density
    }


# ---------------------------------------------------------------------------
# 3. Feature matrix builder
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "code_churn",
    "change_frequency",
    "pain_score",
    "knowledge_concentration",
    "author_count",
]


def build_feature_matrix(
    file_paths: list[str],
    hotspots: list[FileMetrics],
    knowledge: list[FileKnowledge],
    pain: list[FilePain],
) -> tuple[list[str], list[list[float]]]:
    """Build feature matrix from report data.

    Returns (file_paths, matrix) where each row corresponds to a file.
    Feature order: [code_churn, change_frequency, pain_score,
                    knowledge_concentration, author_count]
    """
    hot_map = {f.file_path: f for f in hotspots}
    know_map = {f.file_path: f for f in knowledge}
    pain_map = {f.file_path: f for f in pain}

    matrix: list[list[float]] = []
    for fp in file_paths:
        h = hot_map.get(fp)
        k = know_map.get(fp)
        p = pain_map.get(fp)
        row = [
            h.code_churn if h else 0,
            h.change_frequency if h else 0,
            p.pain_score if p else 0.0,
            k.knowledge_concentration if k else 0.0,
            k.author_count if k else 1,
        ]
        matrix.append(row)
    return file_paths, matrix


# ---------------------------------------------------------------------------
# 4. Ridge regression via Gauss-Jordan elimination
# ---------------------------------------------------------------------------

def ridge_regression(
    X: list[list[float]],
    y: list[float],
    alpha: float = 1.0,
) -> list[float]:
    """Solve beta = (X^T X + alpha*I)^{-1} X^T y via Gauss-Jordan.

    Args:
        X: n x p feature matrix.
        y: n-length target vector.
        alpha: regularization strength (>0 guarantees invertibility).

    Returns:
        p-length coefficient vector.
    """
    n = len(X)
    p = len(X[0])

    # Compute X^T X + alpha*I  (p x p)
    xtx = [[0.0] * p for _ in range(p)]
    for i in range(p):
        for j in range(p):
            s = 0.0
            for k in range(n):
                s += X[k][i] * X[k][j]
            xtx[i][j] = s
        xtx[i][i] += alpha

    # Compute X^T y  (p-length)
    xty = [0.0] * p
    for i in range(p):
        s = 0.0
        for k in range(n):
            s += X[k][i] * y[k]
        xty[i] = s

    # Gauss-Jordan elimination on augmented matrix [xtx | xty]
    aug = [xtx[i][:] + [xty[i]] for i in range(p)]

    for col in range(p):
        # Partial pivoting
        max_row = col
        max_val = abs(aug[col][col])
        for row in range(col + 1, p):
            if abs(aug[row][col]) > max_val:
                max_val = abs(aug[row][col])
                max_row = row
        aug[col], aug[max_row] = aug[max_row], aug[col]

        pivot = aug[col][col]
        for j in range(col, p + 1):
            aug[col][j] /= pivot

        for row in range(p):
            if row == col:
                continue
            factor = aug[row][col]
            for j in range(col, p + 1):
                aug[row][j] -= factor * aug[col][j]

    return [aug[i][p] for i in range(p)]


# ---------------------------------------------------------------------------
# 5. R-squared
# ---------------------------------------------------------------------------

def r_squared(y_true: list[float], y_pred: list[float]) -> float:
    """Compute coefficient of determination R².

    Returns 0.0 if SS_tot == 0 (constant y).
    """
    n = len(y_true)
    mean_y = sum(y_true) / n
    ss_tot = sum((yt - mean_y) ** 2 for yt in y_true)
    if ss_tot == 0:
        return 0.0
    ss_res = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))
    return 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# 6. REI scores
# ---------------------------------------------------------------------------

def compute_rei_scores(
    matrix: list[list[float]],
    coefficients: list[float],
) -> list[float]:
    """Compute Relative Effort Index via dot product + min-max normalize.

    Returns list of REI scores in [0, 1], same order as matrix rows.
    """
    raw = [
        sum(row[j] * coefficients[j] for j in range(len(coefficients)))
        for row in matrix
    ]
    lo, hi = min(raw), max(raw)
    rng = hi - lo
    if rng == 0:
        return [0.0] * len(raw)
    return [(v - lo) / rng for v in raw]
