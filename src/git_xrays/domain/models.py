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
