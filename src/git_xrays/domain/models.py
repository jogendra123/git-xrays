from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class RepoSummary:
    repo_path: str
    commit_count: int
    first_commit_date: datetime | None
    last_commit_date: datetime | None


@dataclass(frozen=True)
class FileChange:
    """A single file's change within one commit."""

    commit_hash: str
    date: datetime
    file_path: str
    lines_added: int
    lines_deleted: int
    author_name: str
    author_email: str


@dataclass(frozen=True)
class FileMetrics:
    """Behavioral metrics for a single file within a time window."""

    file_path: str
    change_frequency: int  # number of commits that touched this file
    code_churn: int  # total lines added + deleted
    hotspot_score: float  # normalized(churn) * normalized(frequency)
    rework_ratio: float  # (frequency - 1) / frequency


@dataclass(frozen=True)
class HotspotReport:
    """Behavioral metrics report across all files in a time window."""

    repo_path: str
    window_days: int
    from_date: datetime
    to_date: datetime
    total_commits: int
    files: list[FileMetrics]  # sorted by hotspot_score descending


@dataclass(frozen=True)
class AuthorContribution:
    """A single author's contribution to a file."""

    author_name: str
    author_email: str
    change_count: int
    total_churn: int
    proportion: float
    weighted_proportion: float


@dataclass(frozen=True)
class FileKnowledge:
    """Knowledge distribution metrics for a single file."""

    file_path: str
    knowledge_concentration: float  # KDI: 1.0 = single author, 0.0 = evenly distributed
    primary_author: str
    primary_author_pct: float
    is_knowledge_island: bool
    author_count: int
    authors: list[AuthorContribution]


@dataclass(frozen=True)
class KnowledgeReport:
    """Knowledge distribution report across all files in a time window."""

    repo_path: str
    window_days: int
    from_date: datetime
    to_date: datetime
    total_commits: int
    developer_risk_index: int
    knowledge_island_count: int
    files: list[FileKnowledge]  # sorted by knowledge_concentration descending


@dataclass(frozen=True)
class CouplingPair:
    """Temporal coupling between two files based on co-change frequency."""

    file_a: str  # alphabetically first
    file_b: str  # alphabetically second
    shared_commits: int
    total_commits: int  # total unique commits in repo window
    coupling_strength: float  # Jaccard: shared / union
    support: float  # shared / total_commits


@dataclass(frozen=True)
class FilePain:
    """PAIN metric for a single file: Size x Distance x Volatility."""

    file_path: str
    size_raw: int
    size_normalized: float
    volatility_raw: int
    volatility_normalized: float
    distance_raw: float
    distance_normalized: float
    pain_score: float


@dataclass(frozen=True)
class CouplingReport:
    """Temporal coupling and PAIN analysis report."""

    repo_path: str
    window_days: int
    from_date: datetime
    to_date: datetime
    total_commits: int
    coupling_pairs: list[CouplingPair]  # sorted by coupling_strength descending
    file_pain: list[FilePain]  # sorted by pain_score descending


@dataclass(frozen=True)
class FileHotspotDelta:
    """Per-file hotspot change between two snapshots."""

    file_path: str
    from_score: float        # 0.0 if new
    to_score: float          # 0.0 if removed
    score_delta: float       # to_score - from_score
    from_churn: int          # 0 if new
    to_churn: int            # 0 if removed
    churn_delta: int         # to_churn - from_churn
    from_frequency: int      # 0 if new
    to_frequency: int        # 0 if removed
    frequency_delta: int     # to_frequency - from_frequency
    status: str              # "unchanged"|"improved"|"degraded"|"new"|"removed"


@dataclass(frozen=True)
class ComparisonReport:
    """Hotspot comparison between two points in time."""

    repo_path: str
    from_ref: str
    to_ref: str
    from_date: datetime
    to_date: datetime
    window_days: int
    from_total_commits: int
    to_total_commits: int
    files: list[FileHotspotDelta]  # sorted by abs(score_delta) desc
    new_hotspot_count: int
    removed_hotspot_count: int
    improved_count: int
    degraded_count: int


@dataclass(frozen=True)
class ClassMetrics:
    """AST-derived metrics for a single class."""

    class_name: str
    file_path: str
    field_count: int
    method_count: int              # all methods incl dunders
    behavior_method_count: int     # non-dunder, non-property with logic
    dunder_method_count: int
    property_count: int
    dbsi: float                    # field_count / (field_count + behavior_method_count)
    logic_density: float           # methods_with_logic / non_dunder_non_property_methods
    orchestration_pressure: float  # 1 - logic_density
    ams: float                     # dbsi * orchestration_pressure


@dataclass(frozen=True)
class FileAnemic:
    """Anemia metrics for a single file."""

    file_path: str
    class_count: int
    anemic_class_count: int
    worst_ams: float               # highest AMS (0.0 if no classes)
    classes: list[ClassMetrics]    # sorted by ams desc
    touch_count: int               # other .py files importing from this file


@dataclass(frozen=True)
class AnemicReport:
    """Anemia analysis report across all Python files."""

    repo_path: str
    ref: str | None                # git ref analyzed (None = HEAD)
    total_files: int
    total_classes: int
    anemic_count: int
    anemic_percentage: float       # anemic_count / total_classes * 100
    average_ams: float
    ams_threshold: float           # default 0.5
    files: list[FileAnemic]        # sorted by worst_ams desc


@dataclass(frozen=True)
class FunctionComplexity:
    """Complexity metrics for a single function or method."""

    function_name: str
    file_path: str
    class_name: str | None        # None if top-level function
    line_number: int
    length: int                    # end_lineno - lineno + 1
    cyclomatic_complexity: int     # 1 + decision points
    max_nesting_depth: int
    branch_count: int              # ast.If node count
    exception_paths: int           # ast.ExceptHandler count


@dataclass(frozen=True)
class FileComplexity:
    """Complexity metrics aggregated for a single file."""

    file_path: str
    function_count: int
    total_complexity: int
    avg_complexity: float
    max_complexity: int
    worst_function: str            # name of highest complexity function ("" if none)
    avg_length: float
    max_length: int
    avg_nesting: float
    max_nesting: int
    functions: list[FunctionComplexity]  # sorted by cyclomatic_complexity desc


@dataclass(frozen=True)
class ComplexityReport:
    """Complexity analysis report across all Python files."""

    repo_path: str
    ref: str | None
    total_files: int
    total_functions: int
    avg_complexity: float
    max_complexity: int
    high_complexity_count: int     # functions above threshold
    complexity_threshold: int      # default 10
    avg_length: float
    max_length: int
    files: list[FileComplexity]    # sorted by max_complexity desc


@dataclass(frozen=True)
class CommitFeatures:
    """Feature vector for a single commit."""

    commit_hash: str
    date: datetime
    file_count: int          # number of files touched
    total_churn: int         # lines_added + lines_deleted
    add_ratio: float         # lines_added / total_churn (0.0 if churn==0)


@dataclass(frozen=True)
class ClusterSummary:
    """Summary of a single cluster of commits."""

    cluster_id: int
    label: str               # "feature"|"bugfix"|"refactoring"|"config"|"mixed"
    size: int                # commits in cluster
    centroid_file_count: float
    centroid_total_churn: float
    centroid_add_ratio: float
    commits: list[CommitFeatures]


@dataclass(frozen=True)
class ClusterDrift:
    """Drift of a cluster between first and second half of window."""

    cluster_label: str
    first_half_pct: float
    second_half_pct: float
    drift: float             # second - first (positive = growing)
    trend: str               # "growing"|"shrinking"|"stable"


@dataclass(frozen=True)
class ClusteringReport:
    """Change clustering analysis report."""

    repo_path: str
    window_days: int
    from_date: datetime
    to_date: datetime
    total_commits: int
    k: int
    silhouette_score: float
    clusters: list[ClusterSummary]   # sorted by size desc
    drift: list[ClusterDrift]        # sorted by abs(drift) desc


@dataclass(frozen=True)
class FeatureAttribution:
    """Explains one feature's contribution to a file's REI score."""

    feature_name: str       # e.g. "code_churn", "pain_score"
    raw_value: float        # original feature value
    weight: float           # learned coefficient
    contribution: float     # weight * normalized_value


@dataclass(frozen=True)
class FileEffort:
    """Effort model results for a single file."""

    file_path: str
    rei_score: float        # Relative Effort Index in [0,1]
    proxy_label: float      # effort proxy label in [0,1]
    commit_density: float   # 1 / (1 + median_interval_days)
    rework_ratio: float     # from hotspot analysis
    attributions: list[FeatureAttribution]  # sorted by abs(contribution) desc


@dataclass(frozen=True)
class EffortReport:
    """Effort modeling report across all files in a time window."""

    repo_path: str
    window_days: int
    from_date: datetime
    to_date: datetime
    total_files: int
    model_r_squared: float  # goodness of fit (0.0 if fallback)
    alpha: float            # ridge regularization parameter
    feature_names: list[str]   # 5 feature names in order
    coefficients: list[float]  # 5 learned coefficients
    files: list[FileEffort]    # sorted by rei_score desc


@dataclass(frozen=True)
class FileCognitiveLoad:
    """Cognitive load breakdown for a single file."""

    file_path: str
    complexity_score: float       # normalized from ComplexityReport
    coordination_score: float     # normalized coupling distance
    knowledge_score: float        # knowledge_concentration
    change_rate_score: float      # normalized file commit count
    composite_load: float         # mean of 4 sub-scores in [0,1]


@dataclass(frozen=True)
class DXMetrics:
    """The four DX Core metrics."""

    throughput: float             # [0,1]
    feedback_delay: float         # [0,1]
    focus_ratio: float            # [0,1]
    cognitive_load: float         # [0,1]


@dataclass(frozen=True)
class DXReport:
    """Developer Experience analysis report."""

    repo_path: str
    window_days: int
    from_date: datetime
    to_date: datetime
    total_commits: int
    total_files: int
    dx_score: float               # composite in [0,1]
    weights: list[float]          # 4 weights [throughput, feedback, focus, cognitive]
    metrics: DXMetrics
    cognitive_load_files: list[FileCognitiveLoad]  # sorted by composite_load desc
