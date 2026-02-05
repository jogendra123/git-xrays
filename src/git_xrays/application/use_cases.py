import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from git_xrays.domain.models import (
    AuthorContribution,
    CouplingPair,
    CouplingReport,
    FileChange,
    FileKnowledge,
    FileMetrics,
    FilePain,
    HotspotReport,
    KnowledgeReport,
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


def _shannon_entropy(proportions: list[float]) -> float:
    """Compute Shannon entropy of a probability distribution."""
    return -sum(p * math.log2(p) for p in proportions if p > 0)


def analyze_knowledge(
    repo: GitRepository, repo_path: str, window_days: int,
    current_time: datetime | None = None,
    island_threshold: float = 0.8,
    half_life: float = 90.0,
) -> KnowledgeReport:
    now = current_time or datetime.now(timezone.utc)
    since = now - timedelta(days=window_days)

    changes = repo.file_changes(since=since, until=now)

    if not changes:
        return KnowledgeReport(
            repo_path=repo_path, window_days=window_days,
            from_date=since, to_date=now,
            total_commits=0, developer_risk_index=0,
            knowledge_island_count=0, files=[],
        )

    total_commits = len({c.commit_hash for c in changes})

    # Aggregate per file per author: raw churn and weighted churn
    # Key: (file_path, author_email)
    raw_churn: defaultdict[tuple[str, str], int] = defaultdict(int)
    weighted_churn: defaultdict[tuple[str, str], float] = defaultdict(float)
    change_count: defaultdict[tuple[str, str], int] = defaultdict(int)
    author_names: dict[str, str] = {}  # email -> name

    for c in changes:
        key = (c.file_path, c.author_email)
        churn = c.lines_added + c.lines_deleted
        raw_churn[key] += churn
        change_count[key] += 1
        author_names[c.author_email] = c.author_name

        age_days = (now - c.date).total_seconds() / 86400
        weight = 2 ** (-age_days / half_life)
        weighted_churn[key] += churn * weight

    # Group by file
    file_paths = sorted(set(fp for fp, _ in raw_churn))

    knowledge_files: list[FileKnowledge] = []
    for fp in file_paths:
        # Gather authors for this file
        authors_for_file: dict[str, tuple[int, int, float]] = {}  # email -> (count, raw, weighted)
        for (path, email), rc in raw_churn.items():
            if path != fp:
                continue
            authors_for_file[email] = (
                change_count[(path, email)],
                rc,
                weighted_churn[(path, email)],
            )

        total_raw = sum(v[1] for v in authors_for_file.values())
        total_weighted = sum(v[2] for v in authors_for_file.values())

        contributions: list[AuthorContribution] = []
        for email, (cnt, rc, wc) in authors_for_file.items():
            proportion = rc / total_raw if total_raw > 0 else 0.0
            w_proportion = wc / total_weighted if total_weighted > 0 else 0.0
            contributions.append(AuthorContribution(
                author_name=author_names[email],
                author_email=email,
                change_count=cnt,
                total_churn=rc,
                proportion=round(proportion, 4),
                weighted_proportion=round(w_proportion, 4),
            ))

        # Sort by proportion descending
        contributions.sort(key=lambda a: a.proportion, reverse=True)

        primary = contributions[0]
        n_authors = len(contributions)

        # KDI = 1 - normalized_shannon_entropy
        proportions = [a.proportion for a in contributions]
        if n_authors <= 1:
            kdi = 1.0
        else:
            max_entropy = math.log2(n_authors)
            entropy = _shannon_entropy(proportions)
            kdi = round(1 - entropy / max_entropy, 4) if max_entropy > 0 else 1.0

        is_island = primary.proportion > island_threshold

        knowledge_files.append(FileKnowledge(
            file_path=fp,
            knowledge_concentration=kdi,
            primary_author=primary.author_name,
            primary_author_pct=primary.proportion,
            is_knowledge_island=is_island,
            author_count=n_authors,
            authors=contributions,
        ))

    knowledge_files.sort(key=lambda f: f.knowledge_concentration, reverse=True)

    island_count = sum(1 for f in knowledge_files if f.is_knowledge_island)
    dri = _compute_dri(changes)

    return KnowledgeReport(
        repo_path=repo_path,
        window_days=window_days,
        from_date=since,
        to_date=now,
        total_commits=total_commits,
        developer_risk_index=dri,
        knowledge_island_count=island_count,
        files=knowledge_files,
    )


def _compute_dri(changes: list[FileChange]) -> int:
    """Compute Developer Risk Index (bus factor).

    Minimum number of authors accounting for >50% of total churn.
    """
    if not changes:
        return 0

    author_churn: defaultdict[str, int] = defaultdict(int)
    for c in changes:
        author_churn[c.author_email] += c.lines_added + c.lines_deleted

    total = sum(author_churn.values())
    if total == 0:
        return 0

    sorted_authors = sorted(author_churn.values(), reverse=True)
    cumulative = 0
    for i, churn in enumerate(sorted_authors):
        cumulative += churn
        if cumulative > total * 0.5:
            return i + 1

    return len(sorted_authors)


def analyze_coupling(
    repo: GitRepository, repo_path: str, window_days: int,
    current_time: datetime | None = None,
    min_shared_commits: int = 2,
) -> CouplingReport:
    now = current_time or datetime.now(timezone.utc)
    since = now - timedelta(days=window_days)

    changes = repo.file_changes(since=since, until=now)

    if not changes:
        return CouplingReport(
            repo_path=repo_path, window_days=window_days,
            from_date=since, to_date=now,
            total_commits=0, coupling_pairs=[], file_pain=[],
        )

    # Build commit → set of files
    commit_files: defaultdict[str, set[str]] = defaultdict(set)
    for c in changes:
        commit_files[c.commit_hash].add(c.file_path)

    total_commits = len(commit_files)

    # Count co-changes for each file pair and individual commit counts
    file_commit_count: defaultdict[str, int] = defaultdict(int)
    pair_shared: defaultdict[tuple[str, str], int] = defaultdict(int)

    for files in commit_files.values():
        sorted_files = sorted(files)
        for f in sorted_files:
            file_commit_count[f] += 1
        for i in range(len(sorted_files)):
            for j in range(i + 1, len(sorted_files)):
                pair_shared[(sorted_files[i], sorted_files[j])] += 1

    # Build coupling pairs with Jaccard filtering
    coupling_pairs: list[CouplingPair] = []
    for (fa, fb), shared in pair_shared.items():
        if shared < min_shared_commits:
            continue
        union = file_commit_count[fa] + file_commit_count[fb] - shared
        strength = round(shared / union, 4) if union > 0 else 0.0
        support = round(shared / total_commits, 4) if total_commits > 0 else 0.0
        coupling_pairs.append(CouplingPair(
            file_a=fa, file_b=fb,
            shared_commits=shared, total_commits=total_commits,
            coupling_strength=strength, support=support,
        ))

    coupling_pairs.sort(key=lambda p: p.coupling_strength, reverse=True)

    # Compute per-file metrics for PAIN
    all_files = sorted(set(c.file_path for c in changes))

    # Size: total churn per file
    size_raw: defaultdict[str, int] = defaultdict(int)
    volatility_raw: defaultdict[str, set[str]] = defaultdict(set)
    for c in changes:
        size_raw[c.file_path] += c.lines_added + c.lines_deleted
        volatility_raw[c.file_path].add(c.commit_hash)

    # Distance: mean coupling strength of filtered pairs involving the file
    file_strengths: defaultdict[str, list[float]] = defaultdict(list)
    for p in coupling_pairs:
        file_strengths[p.file_a].append(p.coupling_strength)
        file_strengths[p.file_b].append(p.coupling_strength)

    distance_raw: dict[str, float] = {}
    for f in all_files:
        if f in file_strengths:
            distance_raw[f] = round(sum(file_strengths[f]) / len(file_strengths[f]), 4)
        else:
            distance_raw[f] = 0.0

    # Normalize
    max_size = max(size_raw[f] for f in all_files) if all_files else 1
    max_vol = max(len(volatility_raw[f]) for f in all_files) if all_files else 1
    max_dist = max(distance_raw[f] for f in all_files) if all_files else 1.0

    file_pain: list[FilePain] = []
    for f in all_files:
        s_raw = size_raw[f]
        v_raw = len(volatility_raw[f])
        d_raw = distance_raw[f]

        s_norm = round(s_raw / max_size, 4) if max_size > 0 else 0.0
        v_norm = round(v_raw / max_vol, 4) if max_vol > 0 else 0.0
        d_norm = round(d_raw / max_dist, 4) if max_dist > 0 else 0.0

        pain = round(s_norm * v_norm * d_norm, 4)

        file_pain.append(FilePain(
            file_path=f,
            size_raw=s_raw, size_normalized=s_norm,
            volatility_raw=v_raw, volatility_normalized=v_norm,
            distance_raw=d_raw, distance_normalized=d_norm,
            pain_score=pain,
        ))

    file_pain.sort(key=lambda fp: fp.pain_score, reverse=True)

    return CouplingReport(
        repo_path=repo_path, window_days=window_days,
        from_date=since, to_date=now,
        total_commits=total_commits,
        coupling_pairs=coupling_pairs, file_pain=file_pain,
    )
