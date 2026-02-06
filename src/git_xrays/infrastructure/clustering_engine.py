"""Pure-Python change clustering engine: K-Means, silhouette, auto-k, labeling, drift."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from datetime import datetime

from git_xrays.domain.models import ClusterDrift, CommitFeatures, FileChange


def extract_commit_features(changes: list[FileChange]) -> list[CommitFeatures]:
    """Group FileChanges by commit_hash and compute per-commit feature vectors."""
    if not changes:
        return []

    commit_data: dict[str, dict] = {}
    for c in changes:
        if c.commit_hash not in commit_data:
            commit_data[c.commit_hash] = {
                "date": c.date,
                "files": set(),
                "added": 0,
                "deleted": 0,
            }
        d = commit_data[c.commit_hash]
        d["files"].add(c.file_path)
        d["added"] += c.lines_added
        d["deleted"] += c.lines_deleted

    result: list[CommitFeatures] = []
    for commit_hash, d in commit_data.items():
        total_churn = d["added"] + d["deleted"]
        add_ratio = d["added"] / total_churn if total_churn > 0 else 0.0
        result.append(CommitFeatures(
            commit_hash=commit_hash,
            date=d["date"],
            file_count=len(d["files"]),
            total_churn=total_churn,
            add_ratio=round(add_ratio, 4),
        ))

    return result


def min_max_normalize(data: list[list[float]]) -> list[list[float]]:
    """Min-max normalize each dimension to [0, 1]."""
    if not data:
        return []

    dims = len(data[0])
    mins = [min(row[d] for row in data) for d in range(dims)]
    maxs = [max(row[d] for row in data) for d in range(dims)]
    ranges = [maxs[d] - mins[d] for d in range(dims)]

    result: list[list[float]] = []
    for row in data:
        normalized = []
        for d in range(dims):
            if ranges[d] == 0:
                normalized.append(0.0)
            else:
                normalized.append((row[d] - mins[d]) / ranges[d])
        result.append(normalized)

    return result


def _euclidean_dist(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def kmeans(
    points: list[list[float]], k: int, seed: int = 42, max_iter: int = 100,
) -> tuple[list[list[float]], list[int]]:
    """Lloyd's K-Means algorithm. Returns (centroids, assignments)."""
    n = len(points)
    dims = len(points[0])

    rng = random.Random(seed)
    indices = rng.sample(range(n), min(k, n))
    centroids = [list(points[i]) for i in indices]

    # Pad if k > n
    while len(centroids) < k:
        centroids.append(list(centroids[-1]))

    assignments = [0] * n

    for _ in range(max_iter):
        # Assign each point to nearest centroid
        new_assignments = []
        for p in points:
            dists = [_euclidean_dist(p, c) for c in centroids]
            new_assignments.append(dists.index(min(dists)))

        if new_assignments == assignments and _ > 0:
            break
        assignments = new_assignments

        # Recompute centroids
        for ci in range(k):
            members = [points[i] for i in range(n) if assignments[i] == ci]
            if members:
                centroids[ci] = [
                    sum(m[d] for m in members) / len(members)
                    for d in range(dims)
                ]

    return centroids, assignments


def silhouette_score(points: list[list[float]], assignments: list[int]) -> float:
    """Compute mean silhouette coefficient."""
    if not points:
        return 0.0

    n = len(points)
    clusters = set(assignments)
    if len(clusters) <= 1:
        return 0.0

    # Group indices by cluster
    cluster_indices: defaultdict[int, list[int]] = defaultdict(list)
    for i, a in enumerate(assignments):
        cluster_indices[a].append(i)

    # Check: if any cluster has only 1 member and there's only 2 clusters with 1 each
    if all(len(v) == 1 for v in cluster_indices.values()):
        return 0.0

    scores: list[float] = []
    for i in range(n):
        own_cluster = assignments[i]
        own_members = cluster_indices[own_cluster]

        # a(i): mean distance to same-cluster members
        if len(own_members) <= 1:
            a_i = 0.0
        else:
            a_i = sum(
                _euclidean_dist(points[i], points[j])
                for j in own_members if j != i
            ) / (len(own_members) - 1)

        # b(i): min mean distance to other clusters
        b_i = float("inf")
        for c, members in cluster_indices.items():
            if c == own_cluster:
                continue
            mean_dist = sum(
                _euclidean_dist(points[i], points[j]) for j in members
            ) / len(members)
            if mean_dist < b_i:
                b_i = mean_dist

        if b_i == float("inf"):
            b_i = 0.0

        denom = max(a_i, b_i)
        if denom == 0:
            scores.append(0.0)
        else:
            scores.append((b_i - a_i) / denom)

    return sum(scores) / len(scores)


def auto_select_k(
    points: list[list[float]], k_min: int = 2, k_max: int = 8, seed: int = 42,
) -> int:
    """Try k=k_min..k_max, return k with highest silhouette score."""
    n = len(points)
    if n <= k_min:
        return k_min

    best_k = k_min
    best_score = -2.0

    for k in range(k_min, min(k_max, n) + 1):
        _, assignments = kmeans(points, k=k, seed=seed)
        score = silhouette_score(points, assignments)
        if score > best_score:
            best_score = score
            best_k = k

    return best_k


def label_cluster(
    norm_file_count: float, norm_churn: float, norm_add_ratio: float,
) -> str:
    """Heuristic label based on normalized centroid values."""
    if norm_add_ratio >= 0.6 and norm_churn >= 0.5:
        return "feature"
    if norm_churn < 0.3 and norm_file_count < 0.3:
        if norm_add_ratio < 0.6:
            return "config"
        return "bugfix"
    if norm_file_count >= 0.5 and 0.3 <= norm_add_ratio < 0.6:
        return "refactoring"
    return "mixed"


def compute_cluster_drift(
    commits: list[CommitFeatures],
    assignments: list[int],
    labels: dict[int, str],
    midpoint: datetime,
) -> list[ClusterDrift]:
    """Compute distribution drift between first and second half of window."""
    if not commits:
        return []

    first_half: defaultdict[int, int] = defaultdict(int)
    second_half: defaultdict[int, int] = defaultdict(int)

    for i, commit in enumerate(commits):
        cluster = assignments[i]
        if commit.date < midpoint:
            first_half[cluster] += 1
        else:
            second_half[cluster] += 1

    total_first = sum(first_half.values())
    total_second = sum(second_half.values())

    all_clusters = sorted(set(assignments))
    result: list[ClusterDrift] = []

    for c in all_clusters:
        f_pct = (first_half[c] / total_first * 100) if total_first > 0 else 0.0
        s_pct = (second_half[c] / total_second * 100) if total_second > 0 else 0.0
        drift = round(s_pct - f_pct, 1)

        if abs(drift) < 5.0:
            trend = "stable"
        elif drift >= 5.0:
            trend = "growing"
        else:
            trend = "shrinking"

        result.append(ClusterDrift(
            cluster_label=labels.get(c, "mixed"),
            first_half_pct=round(f_pct, 1),
            second_half_pct=round(s_pct, 1),
            drift=drift,
            trend=trend,
        ))

    result.sort(key=lambda d: abs(d.drift), reverse=True)
    return result
