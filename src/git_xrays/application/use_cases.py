from collections import defaultdict
from datetime import datetime, timedelta, timezone

from git_xrays.domain.models import (
    FileChange,
    FileMetrics,
    HotspotReport,
    RepoSummary,
)
from git_xrays.domain.ports import GitRepository


def get_repo_summary(repo: GitRepository, repo_path: str) -> RepoSummary:
    return RepoSummary(
        repo_path=repo_path,
        commit_count=repo.commit_count(),
        first_commit_date=repo.first_commit_date(),
        last_commit_date=repo.last_commit_date(),
    )


def analyze_hotspots(
    repo: GitRepository, repo_path: str, window_days: int,
    current_time: datetime | None = None,
) -> HotspotReport:
    now = current_time or datetime.now(timezone.utc)
    since = now - timedelta(days=window_days)

    changes = repo.file_changes(since=since, until=now)

    # Aggregate per file
    freq: defaultdict[str, int] = defaultdict(int)  # commit count per file
    churn: defaultdict[str, int] = defaultdict(int)  # lines added+deleted
    commits_seen: defaultdict[str, set[str]] = defaultdict(set)

    for c in changes:
        commits_seen[c.file_path].add(c.commit_hash)
        churn[c.file_path] += c.lines_added + c.lines_deleted

    for path, hashes in commits_seen.items():
        freq[path] = len(hashes)

    all_commit_hashes = set()
    for c in changes:
        all_commit_hashes.add(c.commit_hash)
    total_commits = len(all_commit_hashes)

    # Compute normalized hotspot score
    max_freq = max(freq.values()) if freq else 1
    max_churn = max(churn.values()) if churn else 1

    files: list[FileMetrics] = []
    for path in freq:
        f = freq[path]
        ch = churn[path]
        norm_freq = f / max_freq
        norm_churn = ch / max_churn
        hotspot = norm_freq * norm_churn
        rework = (f - 1) / f if f > 1 else 0.0
        files.append(
            FileMetrics(
                file_path=path,
                change_frequency=f,
                code_churn=ch,
                hotspot_score=round(hotspot, 4),
                rework_ratio=round(rework, 4),
            )
        )

    files.sort(key=lambda m: m.hotspot_score, reverse=True)

    return HotspotReport(
        repo_path=repo_path,
        window_days=window_days,
        from_date=since,
        to_date=now,
        total_commits=total_commits,
        files=files,
    )
