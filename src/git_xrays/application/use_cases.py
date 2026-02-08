import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from git_xrays.domain.models import (
    AnemicReport,
    AuthorContribution,
    ClusteringReport,
    ClusterSummary,
    ComparisonReport,
    ComplexityReport,
    CouplingPair,
    CouplingReport,
    DXMetrics,
    DXReport,
    EffortReport,
    FeatureAttribution,
    FileAnemic,
    FileChange,
    FileComplexity,
    FileEffort,
    FileHotspotDelta,
    FileKnowledge,
    FileMetrics,
    FilePain,
    HotspotReport,
    KnowledgeReport,
    RepoSummary,
)
from git_xrays.domain.ports import GitRepository, SourceCodeReader
from git_xrays.infrastructure.ast_analyzer import (
    analyze_file,
    compute_touch_counts,
)
from git_xrays.infrastructure.clustering_engine import (
    auto_select_k,
    compute_cluster_drift,
    extract_commit_features,
    kmeans,
    label_cluster,
    min_max_normalize,
    silhouette_score as compute_silhouette,
)
from git_xrays.infrastructure.complexity_analyzer import analyze_file_complexity
from git_xrays.infrastructure.dx_engine import (
    compute_cognitive_load_per_file,
    compute_dx_score,
    compute_feedback_delay,
    compute_focus_ratio,
    compute_throughput,
)
from git_xrays.infrastructure.effort_engine import (
    FEATURE_NAMES,
    build_feature_matrix,
    compute_commit_density,
    compute_effort_proxy,
    compute_rei_scores,
    grid_search_alpha,
    r_squared as compute_r_squared,
    ridge_regression,
)


def get_repo_summary(repo: GitRepository, repo_path: str) -> RepoSummary:
    return RepoSummary(
        repo_path=repo_path,
        commit_count=repo.commit_count(),
        first_commit_date=repo.first_commit_date(),
        last_commit_date=repo.last_commit_date(),
    )


def _compute_rework_ratio(dates: list[datetime], window_days: int = 14) -> float:
    """Fraction of commits re-touching a file within *window_days* of a prior touch.

    Single-commit files return 0.0.
    """
    if len(dates) <= 1:
        return 0.0
    sorted_dates = sorted(dates)
    rework_count = sum(
        1
        for i in range(1, len(sorted_dates))
        if (sorted_dates[i] - sorted_dates[i - 1]).total_seconds() / 86400 <= window_days
    )
    return rework_count / len(sorted_dates)


HOTSPOT_HALF_LIFE = 30.0


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
    commit_dates: defaultdict[str, list[datetime]] = defaultdict(list)

    # Decay-weighted aggregates for hotspot scoring
    weighted_freq: defaultdict[str, float] = defaultdict(float)
    weighted_churn: defaultdict[str, float] = defaultdict(float)

    for c in changes:
        age_days = (now - c.date).total_seconds() / 86400
        weight = 2 ** (-age_days / HOTSPOT_HALF_LIFE)

        if c.commit_hash not in commits_seen[c.file_path]:
            commit_dates[c.file_path].append(c.date)
            weighted_freq[c.file_path] += weight
        commits_seen[c.file_path].add(c.commit_hash)
        file_churn = c.lines_added + c.lines_deleted
        churn[c.file_path] += file_churn
        weighted_churn[c.file_path] += file_churn * weight

    for path, hashes in commits_seen.items():
        freq[path] = len(hashes)

    all_commit_hashes = set()
    for c in changes:
        all_commit_hashes.add(c.commit_hash)
    total_commits = len(all_commit_hashes)

    # Get file sizes for relative churn
    sizes = repo.file_sizes()

    # Compute weighted relative churn (weighted_churn / file_size)
    weighted_relative_churn: dict[str, float] = {}
    for path in freq:
        wc = weighted_churn[path]
        sz = sizes.get(path, 0)
        weighted_relative_churn[path] = wc / sz if sz > 0 else wc

    # Compute normalized hotspot score using decay-weighted values
    max_wfreq = max(weighted_freq.values()) if weighted_freq else 1.0
    max_wrel_churn = max(weighted_relative_churn.values()) if weighted_relative_churn else 1.0

    files: list[FileMetrics] = []
    for path in freq:
        f = freq[path]
        ch = churn[path]
        sz = sizes.get(path, 0)
        norm_freq = weighted_freq[path] / max_wfreq if max_wfreq > 0 else 0.0
        norm_rel_churn = weighted_relative_churn[path] / max_wrel_churn if max_wrel_churn > 0 else 0.0
        hotspot = norm_freq * norm_rel_churn
        rework = _compute_rework_ratio(commit_dates[path])
        files.append(
            FileMetrics(
                file_path=path,
                change_frequency=f,
                code_churn=ch,
                hotspot_score=round(hotspot, 4),
                rework_ratio=round(rework, 4),
                file_size=sz,
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
            total_commits=0, developer_risk_index=0.0,
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
        proportions = [a.weighted_proportion for a in contributions]
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
    gini = _compute_gini(changes)

    return KnowledgeReport(
        repo_path=repo_path,
        window_days=window_days,
        from_date=since,
        to_date=now,
        total_commits=total_commits,
        developer_risk_index=gini,
        knowledge_island_count=island_count,
        files=knowledge_files,
    )


def _compute_gini(changes: list[FileChange]) -> float:
    """Compute Gini coefficient of author churn distribution.

    Returns a float in [0, 1]. 0.0 = perfectly equal, 1.0 = one person.
    """
    if not changes:
        return 0.0

    author_churn: defaultdict[str, int] = defaultdict(int)
    for c in changes:
        author_churn[c.author_email] += c.lines_added + c.lines_deleted

    values = sorted(author_churn.values())
    n = len(values)
    if n == 0:
        return 0.0
    total = sum(values)
    if total == 0:
        return 0.0
    # Standard Gini: (2 * sum(i*x_i) - (n+1) * sum(x_i)) / (n * sum(x_i))
    numerator = 2.0 * sum((i + 1) * v for i, v in enumerate(values)) - (n + 1) * total
    denominator = n * total
    return round(numerator / denominator, 4)


COUPLING_HALF_LIFE = 30.0


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

    # Build commit → set of files + commit date
    commit_files_map: defaultdict[str, set[str]] = defaultdict(set)
    commit_date_map: dict[str, datetime] = {}
    for c in changes:
        commit_files_map[c.commit_hash].add(c.file_path)
        commit_date_map[c.commit_hash] = c.date

    total_commits = len(commit_files_map)

    # Count co-changes for each file pair and individual commit counts
    # Raw counts (for min_shared filter and lift)
    file_commit_count: defaultdict[str, int] = defaultdict(int)
    pair_shared: defaultdict[tuple[str, str], int] = defaultdict(int)
    # Temporal decay-weighted counts (for coupling_strength)
    file_commit_weight: defaultdict[str, float] = defaultdict(float)
    pair_weighted_shared: defaultdict[tuple[str, str], float] = defaultdict(float)

    for commit_hash, files in commit_files_map.items():
        age_days = (now - commit_date_map[commit_hash]).total_seconds() / 86400
        weight = 2 ** (-age_days / COUPLING_HALF_LIFE)

        sorted_files = sorted(files)
        for f in sorted_files:
            file_commit_count[f] += 1
            file_commit_weight[f] += weight
        for i in range(len(sorted_files)):
            for j in range(i + 1, len(sorted_files)):
                pair_key = (sorted_files[i], sorted_files[j])
                pair_shared[pair_key] += 1
                pair_weighted_shared[pair_key] += weight

    # Build coupling pairs with temporal Jaccard + lift filtering
    coupling_pairs: list[CouplingPair] = []
    for (fa, fb), shared in pair_shared.items():
        if shared < min_shared_commits:
            continue
        # Temporal Jaccard: weighted_shared / weighted_union
        w_shared = pair_weighted_shared[(fa, fb)]
        w_union = file_commit_weight[fa] + file_commit_weight[fb] - w_shared
        strength = round(w_shared / w_union, 4) if w_union > 0 else 0.0
        support = round(shared / total_commits, 4) if total_commits > 0 else 0.0
        # Lift: uses raw counts (not weighted)
        expected = (file_commit_count[fa] / total_commits) * (file_commit_count[fb] / total_commits) * total_commits if total_commits > 0 else 0.0
        lift = round(shared / expected, 4) if expected > 0 else 0.0
        if lift <= 1.0:
            continue
        coupling_pairs.append(CouplingPair(
            file_a=fa, file_b=fb,
            shared_commits=shared, total_commits=total_commits,
            coupling_strength=strength, support=support,
            expected_cochange=round(expected, 4), lift=lift,
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


def _resolve_ref_to_datetime(
    ref: str, repo: GitRepository | None,
) -> datetime:
    """Resolve a ref string to a datetime.

    Tries ISO date parse first, then delegates to repo.resolve_ref().
    """
    # Try ISO datetime/date parse first
    try:
        dt = datetime.fromisoformat(ref)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        pass

    # Delegate to repo
    if repo is None:
        raise ValueError(f"Cannot resolve ref without a repository: {ref}")
    return repo.resolve_ref(ref)


def compare_hotspots(
    repo: GitRepository, repo_path: str, window_days: int,
    from_ref: str, to_ref: str,
) -> ComparisonReport:
    """Compare hotspot analysis between two points in time."""
    from_date = _resolve_ref_to_datetime(from_ref, repo)
    to_date = _resolve_ref_to_datetime(to_ref, repo)

    from_report = analyze_hotspots(repo, repo_path, window_days, current_time=from_date)
    to_report = analyze_hotspots(repo, repo_path, window_days, current_time=to_date)

    # Build lookup maps
    from_map = {f.file_path: f for f in from_report.files}
    to_map = {f.file_path: f for f in to_report.files}

    all_files = sorted(set(from_map) | set(to_map))

    deltas: list[FileHotspotDelta] = []
    for fp in all_files:
        f_from = from_map.get(fp)
        f_to = to_map.get(fp)

        from_score = f_from.hotspot_score if f_from else 0.0
        to_score = f_to.hotspot_score if f_to else 0.0
        score_delta = round(to_score - from_score, 4)

        from_churn = f_from.code_churn if f_from else 0
        to_churn = f_to.code_churn if f_to else 0

        from_freq = f_from.change_frequency if f_from else 0
        to_freq = f_to.change_frequency if f_to else 0

        if f_from is None:
            status = "new"
        elif f_to is None:
            status = "removed"
        elif score_delta > 0:
            status = "degraded"
        elif score_delta < 0:
            status = "improved"
        else:
            status = "unchanged"

        deltas.append(FileHotspotDelta(
            file_path=fp,
            from_score=from_score, to_score=to_score, score_delta=score_delta,
            from_churn=from_churn, to_churn=to_churn, churn_delta=to_churn - from_churn,
            from_frequency=from_freq, to_frequency=to_freq,
            frequency_delta=to_freq - from_freq,
            status=status,
        ))

    deltas.sort(key=lambda d: abs(d.score_delta), reverse=True)

    return ComparisonReport(
        repo_path=repo_path,
        from_ref=from_ref, to_ref=to_ref,
        from_date=from_date, to_date=to_date,
        window_days=window_days,
        from_total_commits=from_report.total_commits,
        to_total_commits=to_report.total_commits,
        files=deltas,
        new_hotspot_count=sum(1 for d in deltas if d.status == "new"),
        removed_hotspot_count=sum(1 for d in deltas if d.status == "removed"),
        improved_count=sum(1 for d in deltas if d.status == "improved"),
        degraded_count=sum(1 for d in deltas if d.status == "degraded"),
    )


def analyze_anemia(
    source_reader: SourceCodeReader,
    repo_path: str,
    ref: str | None = None,
    ams_threshold: float = 0.5,
) -> AnemicReport:
    """Analyze Python files for anemic domain model patterns."""
    py_files = source_reader.list_python_files(ref=ref)

    if not py_files:
        return AnemicReport(
            repo_path=repo_path, ref=ref,
            total_files=0, total_classes=0,
            anemic_count=0, anemic_percentage=0.0,
            average_ams=0.0, ams_threshold=ams_threshold,
            files=[],
        )

    # Read all sources for touch count analysis
    file_sources: dict[str, str] = {}
    for fp in py_files:
        try:
            file_sources[fp] = source_reader.read_file(fp, ref=ref)
        except FileNotFoundError:
            continue

    touch_counts = compute_touch_counts(file_sources)

    # Analyze each file
    file_results: list[FileAnemic] = []
    for fp, source in file_sources.items():
        fa = analyze_file(source, fp, ams_threshold=ams_threshold)
        # Replace touch_count with computed value
        fa = FileAnemic(
            file_path=fa.file_path,
            class_count=fa.class_count,
            anemic_class_count=fa.anemic_class_count,
            worst_ams=fa.worst_ams,
            classes=fa.classes,
            touch_count=touch_counts.get(fp, 0),
        )
        file_results.append(fa)

    file_results.sort(key=lambda f: f.worst_ams, reverse=True)

    total_classes = sum(f.class_count for f in file_results)
    anemic_count = sum(f.anemic_class_count for f in file_results)
    all_ams = [c.ams for f in file_results for c in f.classes]
    avg_ams = round(sum(all_ams) / len(all_ams), 4) if all_ams else 0.0
    pct = round(anemic_count / total_classes * 100, 1) if total_classes > 0 else 0.0

    return AnemicReport(
        repo_path=repo_path,
        ref=ref,
        total_files=len(file_results),
        total_classes=total_classes,
        anemic_count=anemic_count,
        anemic_percentage=pct,
        average_ams=avg_ams,
        ams_threshold=ams_threshold,
        files=file_results,
    )


def analyze_complexity(
    source_reader: SourceCodeReader,
    repo_path: str,
    ref: str | None = None,
    complexity_threshold: int = 10,
) -> ComplexityReport:
    """Analyze Python files for function-level cyclomatic complexity."""
    py_files = source_reader.list_python_files(ref=ref)

    if not py_files:
        return ComplexityReport(
            repo_path=repo_path, ref=ref,
            total_files=0, total_functions=0,
            avg_complexity=0.0, max_complexity=0,
            high_complexity_count=0, complexity_threshold=complexity_threshold,
            avg_length=0.0, max_length=0,
            avg_cognitive=0.0, max_cognitive=0,
            files=[],
        )

    file_results: list[FileComplexity] = []
    for fp in py_files:
        try:
            source = source_reader.read_file(fp, ref=ref)
        except FileNotFoundError:
            continue
        fc = analyze_file_complexity(source, fp)
        if fc.function_count > 0:
            file_results.append(fc)

    file_results.sort(key=lambda f: f.max_complexity, reverse=True)

    all_funcs = [fn for f in file_results for fn in f.functions]
    total_functions = len(all_funcs)

    if total_functions == 0:
        return ComplexityReport(
            repo_path=repo_path, ref=ref,
            total_files=len(file_results), total_functions=0,
            avg_complexity=0.0, max_complexity=0,
            high_complexity_count=0, complexity_threshold=complexity_threshold,
            avg_length=0.0, max_length=0,
            avg_cognitive=0.0, max_cognitive=0,
            files=file_results,
        )

    all_cc = [fn.cyclomatic_complexity for fn in all_funcs]
    all_cog = [fn.cognitive_complexity for fn in all_funcs]
    all_len = [fn.length for fn in all_funcs]
    avg_cc = round(sum(all_cc) / len(all_cc), 2)
    max_cc = max(all_cc)
    high_count = sum(1 for cc in all_cc if cc > complexity_threshold)
    avg_len = round(sum(all_len) / len(all_len), 2)
    max_len = max(all_len)
    avg_cog = round(sum(all_cog) / len(all_cog), 2)
    max_cog = max(all_cog)

    return ComplexityReport(
        repo_path=repo_path,
        ref=ref,
        total_files=len(file_results),
        total_functions=total_functions,
        avg_complexity=avg_cc,
        max_complexity=max_cc,
        high_complexity_count=high_count,
        complexity_threshold=complexity_threshold,
        avg_length=avg_len,
        max_length=max_len,
        avg_cognitive=avg_cog,
        max_cognitive=max_cog,
        files=file_results,
    )


def analyze_change_clusters(
    repo: GitRepository, repo_path: str, window_days: int,
    current_time: datetime | None = None,
    k: int | None = None,
) -> ClusteringReport:
    """Cluster commits by behavioral patterns."""
    now = current_time or datetime.now(timezone.utc)
    since = now - timedelta(days=window_days)

    changes = repo.file_changes(since=since, until=now)

    if not changes:
        return ClusteringReport(
            repo_path=repo_path, window_days=window_days,
            from_date=since, to_date=now,
            total_commits=0, k=0, silhouette_score=0.0,
            clusters=[], drift=[],
        )

    features = extract_commit_features(changes)
    total_commits = len(features)

    if total_commits == 1:
        # Single commit → single cluster, no silhouette
        cf = features[0]
        cluster = ClusterSummary(
            cluster_id=0,
            label=label_cluster(0.0, 0.0, 0.0),
            size=1,
            centroid_file_count=float(cf.file_count),
            centroid_total_churn=float(cf.total_churn),
            centroid_add_ratio=cf.add_ratio,
            commits=[cf],
        )
        return ClusteringReport(
            repo_path=repo_path, window_days=window_days,
            from_date=since, to_date=now,
            total_commits=1, k=1, silhouette_score=0.0,
            clusters=[cluster], drift=[],
        )

    # Build feature matrix
    raw_points = [
        [float(f.file_count), float(f.total_churn), f.add_ratio]
        for f in features
    ]
    norm_points = min_max_normalize(raw_points)

    # Select k
    chosen_k = k if k is not None else auto_select_k(norm_points, seed=42)

    # Run k-means
    centroids, assignments = kmeans(norm_points, k=chosen_k, seed=42)
    sil_score = compute_silhouette(norm_points, assignments)

    # Normalize centroids for labeling
    centroid_norm = min_max_normalize(centroids) if len(centroids) > 1 else [[0.0] * 3] * len(centroids)

    # Build cluster summaries
    cluster_commits: defaultdict[int, list] = defaultdict(list)
    for i, a in enumerate(assignments):
        cluster_commits[a].append(features[i])

    labels: dict[int, str] = {}
    summaries: list[ClusterSummary] = []
    for ci in range(chosen_k):
        commits = cluster_commits[ci]
        if not commits:
            continue
        cn = centroid_norm[ci] if ci < len(centroid_norm) else [0.0, 0.0, 0.0]
        lbl = label_cluster(cn[0], cn[1], cn[2])
        labels[ci] = lbl
        summaries.append(ClusterSummary(
            cluster_id=ci,
            label=lbl,
            size=len(commits),
            centroid_file_count=centroids[ci][0],
            centroid_total_churn=centroids[ci][1],
            centroid_add_ratio=centroids[ci][2],
            commits=commits,
        ))

    summaries.sort(key=lambda s: s.size, reverse=True)

    # Compute drift
    midpoint = since + (now - since) / 2
    drift = compute_cluster_drift(features, assignments, labels, midpoint)

    return ClusteringReport(
        repo_path=repo_path, window_days=window_days,
        from_date=since, to_date=now,
        total_commits=total_commits,
        k=chosen_k,
        silhouette_score=round(sil_score, 4),
        clusters=summaries,
        drift=drift,
    )


def analyze_effort(
    repo: GitRepository, repo_path: str, window_days: int,
    current_time: datetime | None = None,
    alpha: float | None = None,
) -> EffortReport:
    """Compute Relative Effort Index per file using ridge regression.

    Internally runs hotspot, knowledge, and coupling analyses to gather
    features, then trains a ridge regression model on an effort proxy label.
    Falls back to equal weights when fewer than 3 files are present.

    When alpha is None (default), auto-tunes via grid search over
    [0.1, 0.5, 1.0, 2.0, 5.0]. Pass a float to fix alpha explicitly.
    """
    now = current_time or datetime.now(timezone.utc)
    since = now - timedelta(days=window_days)
    n_features = len(FEATURE_NAMES)

    # Run sub-analyses
    hotspot_report = analyze_hotspots(repo, repo_path, window_days, current_time=now)
    knowledge_report = analyze_knowledge(repo, repo_path, window_days, current_time=now)
    coupling_report = analyze_coupling(repo, repo_path, window_days, current_time=now)

    # Gather file paths from hotspot report
    file_paths = sorted(set(f.file_path for f in hotspot_report.files))
    n_files = len(file_paths)

    chosen_alpha = alpha if alpha is not None else 1.0

    if n_files == 0:
        return EffortReport(
            repo_path=repo_path, window_days=window_days,
            from_date=since, to_date=now,
            total_files=0, model_r_squared=0.0, alpha=chosen_alpha,
            feature_names=FEATURE_NAMES,
            coefficients=[1.0 / n_features] * n_features,
            files=[],
        )

    # Build feature matrix
    _, matrix = build_feature_matrix(
        file_paths, hotspot_report.files,
        knowledge_report.files, coupling_report.file_pain,
    )

    # Compute commit density per file
    changes = repo.file_changes(since=since, until=now)
    file_dates: defaultdict[str, list[datetime]] = defaultdict(list)
    for c in changes:
        file_dates[c.file_path].append(c.date)
    density = compute_commit_density(dict(file_dates))

    # Compute rework map from hotspot data
    rework_map = {f.file_path: f.rework_ratio for f in hotspot_report.files}

    # Compute effort proxy labels
    proxy = compute_effort_proxy(
        {fp: density.get(fp, 1.0) for fp in file_paths},
        {fp: rework_map.get(fp, 0.0) for fp in file_paths},
    )

    use_fallback = n_files < 3
    if use_fallback:
        coefficients = [1.0 / n_features] * n_features
        r2 = 0.0
    else:
        # Normalize features for regression
        norm_matrix = _normalize_matrix(matrix)
        y = [proxy.get(fp, 0.0) for fp in file_paths]
        if alpha is None:
            # Auto-tune alpha via grid search
            chosen_alpha, coefficients, r2 = grid_search_alpha(norm_matrix, y)
        else:
            chosen_alpha = alpha
            coefficients = ridge_regression(norm_matrix, y, alpha=alpha)
            # Compute predictions for R²
            y_pred = [
                sum(norm_matrix[i][j] * coefficients[j] for j in range(n_features))
                for i in range(n_files)
            ]
            r2 = compute_r_squared(y, y_pred)

    # Compute REI scores
    norm_matrix = _normalize_matrix(matrix)
    rei_scores = compute_rei_scores(norm_matrix, coefficients)

    # Build file results with attributions
    files: list[FileEffort] = []
    for i, fp in enumerate(file_paths):
        attribs = []
        for j, fname in enumerate(FEATURE_NAMES):
            raw_val = float(matrix[i][j])
            norm_val = norm_matrix[i][j]
            w = coefficients[j]
            contribution = w * norm_val
            attribs.append(FeatureAttribution(
                feature_name=fname,
                raw_value=raw_val,
                weight=round(w, 6),
                contribution=round(contribution, 6),
            ))
        # Sort attributions by abs(contribution) descending
        attribs.sort(key=lambda a: abs(a.contribution), reverse=True)

        files.append(FileEffort(
            file_path=fp,
            rei_score=round(rei_scores[i], 4),
            proxy_label=round(proxy.get(fp, 0.0), 4),
            commit_density=round(density.get(fp, 1.0), 4),
            rework_ratio=round(rework_map.get(fp, 0.0), 4),
            attributions=attribs,
        ))

    # Sort by REI descending
    files.sort(key=lambda f: f.rei_score, reverse=True)

    return EffortReport(
        repo_path=repo_path, window_days=window_days,
        from_date=since, to_date=now,
        total_files=n_files,
        model_r_squared=round(r2, 4),
        alpha=chosen_alpha,
        feature_names=FEATURE_NAMES,
        coefficients=[round(c, 6) for c in coefficients],
        files=files,
    )


def _normalize_matrix(matrix: list[list[float]]) -> list[list[float]]:
    """Min-max normalize each column of the matrix to [0, 1]."""
    if not matrix:
        return []
    n_cols = len(matrix[0])
    n_rows = len(matrix)
    result = [[0.0] * n_cols for _ in range(n_rows)]
    for j in range(n_cols):
        col = [matrix[i][j] for i in range(n_rows)]
        lo, hi = min(col), max(col)
        rng = hi - lo
        for i in range(n_rows):
            result[i][j] = (matrix[i][j] - lo) / rng if rng != 0 else 0.0
    return result


_DEFAULT_DX_WEIGHTS = [0.3, 0.25, 0.25, 0.2]


def analyze_dx(
    repo: GitRepository,
    source_reader: SourceCodeReader,
    repo_path: str,
    window_days: int,
    current_time: datetime | None = None,
    weights: list[float] | None = None,
) -> DXReport:
    """Compute Developer Experience metrics from existing analyses."""
    now = current_time or datetime.now(timezone.utc)
    since = now - timedelta(days=window_days)
    w = weights or _DEFAULT_DX_WEIGHTS

    # Run sub-analyses
    hotspot_report = analyze_hotspots(repo, repo_path, window_days, current_time=now)
    knowledge_report = analyze_knowledge(repo, repo_path, window_days, current_time=now)
    coupling_report = analyze_coupling(repo, repo_path, window_days, current_time=now)
    clustering_report = analyze_change_clusters(repo, repo_path, window_days, current_time=now)
    complexity_report = analyze_complexity(source_reader, repo_path)

    total_commits = hotspot_report.total_commits
    total_files = len(hotspot_report.files)

    if total_commits == 0:
        metrics = DXMetrics(
            throughput=0.0,
            feedback_delay=0.0,
            focus_ratio=0.5,
            cognitive_load=0.0,
        )
        return DXReport(
            repo_path=repo_path,
            window_days=window_days,
            from_date=since,
            to_date=now,
            total_commits=0,
            total_files=0,
            dx_score=0.0,
            weights=w,
            metrics=metrics,
            cognitive_load_files=[],
        )

    # 1. Throughput
    throughput = compute_throughput(clustering_report.clusters, window_days)

    # 2. Feedback Delay
    changes = repo.file_changes(since=since, until=now)
    file_dates: defaultdict[str, list[datetime]] = defaultdict(list)
    for c in changes:
        file_dates[c.file_path].append(c.date)
    density = compute_commit_density(dict(file_dates))

    densities = [density.get(f.file_path, 1.0) for f in hotspot_report.files]
    rework_ratios = [f.rework_ratio for f in hotspot_report.files]
    feedback = compute_feedback_delay(densities, rework_ratios)

    # 3. Focus Ratio
    focus = compute_focus_ratio(clustering_report.clusters)

    # 4. Cognitive Load
    cognitive_files = compute_cognitive_load_per_file(
        hotspot_report.files,
        knowledge_report.files,
        coupling_report.file_pain,
        complexity_report.files,
    )
    repo_cognitive_load = (
        sum(f.composite_load for f in cognitive_files) / len(cognitive_files)
        if cognitive_files
        else 0.0
    )

    metrics = DXMetrics(
        throughput=round(throughput, 4),
        feedback_delay=round(feedback, 4),
        focus_ratio=round(focus, 4),
        cognitive_load=round(repo_cognitive_load, 4),
    )

    dx_score = compute_dx_score(
        metrics.throughput, metrics.feedback_delay,
        metrics.focus_ratio, metrics.cognitive_load, w,
    )

    return DXReport(
        repo_path=repo_path,
        window_days=window_days,
        from_date=since,
        to_date=now,
        total_commits=total_commits,
        total_files=total_files,
        dx_score=round(dx_score, 4),
        weights=w,
        metrics=metrics,
        cognitive_load_files=cognitive_files,
    )
