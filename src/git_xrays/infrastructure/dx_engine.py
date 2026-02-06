"""Pure-Python DX Core 4 engine — zero external dependencies.

Functions:
- compute_throughput: weighted commit rate metric
- compute_feedback_delay: iteration speed metric
- compute_focus_ratio: feature vs toil ratio
- compute_cognitive_load_per_file: per-file complexity burden
- compute_dx_score: weighted composite DX score
"""

from __future__ import annotations

from git_xrays.domain.models import (
    ClusterSummary,
    FileCognitiveLoad,
    FileComplexity,
    FileKnowledge,
    FileMetrics,
    FilePain,
)

# Label weights for throughput calculation
_LABEL_WEIGHTS: dict[str, float] = {
    "feature": 1.0,
    "refactoring": 0.8,
    "bugfix": 0.5,
    "mixed": 0.5,
    "config": 0.3,
}

_MAX_DAILY_RATE = 10.0


def compute_throughput(
    clusters: list[ClusterSummary],
    window_days: int,
) -> float:
    """Weighted commit rate. Feature commits weighted higher than bugfix/config.

    Returns a value in [0, 1].
    """
    if not clusters or window_days <= 0:
        return 0.0

    weighted_commits = sum(
        c.size * _LABEL_WEIGHTS.get(c.label, 0.5)
        for c in clusters
    )
    raw = weighted_commits / (window_days * _MAX_DAILY_RATE)
    return min(max(raw, 0.0), 1.0)


def compute_feedback_delay(
    densities: list[float],
    rework_ratios: list[float],
) -> float:
    """Fast iteration signal. High density + low rework = fast feedback.

    Returns a value in [0, 1].
    """
    if not densities:
        return 0.0

    mean_density = sum(densities) / len(densities)
    mean_rework = sum(rework_ratios) / len(rework_ratios) if rework_ratios else 0.0
    return mean_density * (1.0 - mean_rework)


def compute_focus_ratio(clusters: list[ClusterSummary]) -> float:
    """Feature work vs maintenance/toil. Mixed commits excluded from denominator.

    Returns a value in [0, 1]. Empty → 0.5 (neutral).
    """
    if not clusters:
        return 0.5

    feature_count = 0
    denominator = 0
    for c in clusters:
        if c.label == "mixed":
            continue
        denominator += c.size
        if c.label == "feature":
            feature_count += c.size

    if denominator == 0:
        return 0.5

    return feature_count / denominator


def compute_cognitive_load_per_file(
    hotspot_files: list[FileMetrics],
    knowledge_files: list[FileKnowledge],
    pain_files: list[FilePain],
    complexity_files: list[FileComplexity],
) -> list[FileCognitiveLoad]:
    """Compute cognitive load per file from 4 sub-signals.

    Returns list sorted by composite_load descending.
    """
    # Gather all file paths from all sources
    all_paths: set[str] = set()
    for f in hotspot_files:
        all_paths.add(f.file_path)
    for f in knowledge_files:
        all_paths.add(f.file_path)
    for f in pain_files:
        all_paths.add(f.file_path)
    for f in complexity_files:
        all_paths.add(f.file_path)

    if not all_paths:
        return []

    # Build lookup maps
    hot_map = {f.file_path: f for f in hotspot_files}
    know_map = {f.file_path: f for f in knowledge_files}
    pain_map = {f.file_path: f for f in pain_files}
    cx_map = {f.file_path: f for f in complexity_files}

    # Gather raw values for normalization
    raw_complexity: dict[str, float] = {}
    raw_coordination: dict[str, float] = {}
    raw_knowledge: dict[str, float] = {}
    raw_change_rate: dict[str, float] = {}

    for fp in all_paths:
        # Complexity: avg_complexity from ComplexityReport
        cx = cx_map.get(fp)
        raw_complexity[fp] = cx.avg_complexity if cx else 0.0

        # Coordination: distance_normalized from FilePain
        p = pain_map.get(fp)
        raw_coordination[fp] = p.distance_normalized if p else 0.0

        # Knowledge: knowledge_concentration from KnowledgeReport
        k = know_map.get(fp)
        raw_knowledge[fp] = k.knowledge_concentration if k else 0.0

        # Change rate: change_frequency from hotspot
        h = hot_map.get(fp)
        raw_change_rate[fp] = float(h.change_frequency) if h else 0.0

    # Min-max normalize each dimension
    norm_complexity = _min_max_norm(raw_complexity)
    norm_coordination = _min_max_norm(raw_coordination)
    norm_knowledge = _min_max_norm(raw_knowledge)
    norm_change_rate = _min_max_norm(raw_change_rate)

    result: list[FileCognitiveLoad] = []
    for fp in sorted(all_paths):
        cs = norm_complexity[fp]
        co = norm_coordination[fp]
        ks = norm_knowledge[fp]
        cr = norm_change_rate[fp]
        composite = (cs + co + ks + cr) / 4.0

        result.append(FileCognitiveLoad(
            file_path=fp,
            complexity_score=round(cs, 4),
            coordination_score=round(co, 4),
            knowledge_score=round(ks, 4),
            change_rate_score=round(cr, 4),
            composite_load=round(composite, 4),
        ))

    result.sort(key=lambda f: f.composite_load, reverse=True)
    return result


def compute_dx_score(
    throughput: float,
    feedback_delay: float,
    focus_ratio: float,
    cognitive_load: float,
    weights: list[float],
) -> float:
    """Weighted composite DX score. Cognitive load is inverted (lower is better).

    Returns a value in [0, 1].
    """
    return (
        weights[0] * throughput
        + weights[1] * feedback_delay
        + weights[2] * focus_ratio
        + weights[3] * (1.0 - cognitive_load)
    )


def _min_max_norm(values: dict[str, float]) -> dict[str, float]:
    """Min-max normalize dict values to [0, 1]."""
    if not values:
        return {}
    vals = list(values.values())
    lo, hi = min(vals), max(vals)
    rng = hi - lo
    if rng == 0:
        return {k: 0.0 for k in values}
    return {k: (v - lo) / rng for k, v in values.items()}
