from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class RepoSummary:
    repo_path: str
    commit_count: int
    first_commit_date: datetime | None
    last_commit_date: datetime | None
