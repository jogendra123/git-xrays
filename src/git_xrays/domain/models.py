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
