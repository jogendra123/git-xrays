import argparse
import re
import sys

from git_xrays.application.use_cases import (
    _resolve_ref_to_datetime,
    analyze_anemia,
    analyze_change_clusters,
    analyze_complexity,
    analyze_coupling,
    analyze_dx,
    analyze_effort,
    analyze_hotspots,
    analyze_knowledge,
    compare_hotspots,
    get_repo_summary,
)
from git_xrays.infrastructure.git_cli_reader import GitCliReader
from git_xrays.infrastructure.git_source_reader import GitSourceReader


def _parse_window(value: str) -> int:
    """Parse a window string like '90d' into days."""
    match = re.fullmatch(r"(\d+)d", value)
    if not match:
        raise argparse.ArgumentTypeError(
            f"Invalid window format '{value}'. Use <number>d, e.g. 90d"
        )
    return int(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="analyze-repo",
        description="Behavioral & Architectural Code Intelligence",
    )
    parser.add_argument(
        "repo_path", nargs="?", default=None,
        help="Path to a local git repository",
    )
    parser.add_argument(
        "--window",
        type=_parse_window,
        default=90,
        metavar="DAYS",
        help="Analysis window in days, e.g. 90d (default: 90d)",
    )
    parser.add_argument(
        "--knowledge",
        action="store_true",
        help="Show knowledge distribution analysis",
    )
    parser.add_argument(
        "--coupling",
        action="store_true",
        help="Show temporal coupling and PAIN analysis",
    )
    parser.add_argument(
        "--anemia",
        action="store_true",
        help="Show anemic domain model analysis",
    )
    parser.add_argument(
        "--complexity",
        action="store_true",
        help="Show function-level complexity analysis",
    )
    parser.add_argument(
        "--clustering",
        action="store_true",
        help="Show change clustering analysis",
    )
    parser.add_argument(
        "--effort",
        action="store_true",
        help="Show effort modeling analysis (REI scores)",
    )
    parser.add_argument(
        "--dx",
        action="store_true",
        help="Show Developer Experience (DX Core 4) analysis",
    )
    parser.add_argument(
        "--at",
        metavar="REF",
        help="Anchor analysis at a commit, tag, branch, or date",
    )
    parser.add_argument(
        "--from",
        dest="from_ref",
        metavar="REF",
        help="Start ref for comparison (requires --to)",
    )
    parser.add_argument(
        "--to",
        dest="to_ref",
        metavar="REF",
        help="End ref for comparison (requires --from)",
    )
    parser.add_argument(
        "--all",
        dest="run_all",
        action="store_true",
        help="Run all analyses, print all output, and store results in DuckDB",
    )
    parser.add_argument(
        "--db",
        metavar="PATH",
        default=None,
        help="DuckDB file path (default: ~/.git-xrays/runs.db)",
    )
    parser.add_argument(
        "--list-runs",
        action="store_true",
        help="Show past runs from DuckDB, then exit",
    )
    args = parser.parse_args()

    # Handle --list-runs early (no repo_path required)
    if args.list_runs:
        from git_xrays.infrastructure.run_store import RunStore

        store = RunStore(db_path=args.db)
        runs = store.list_runs()
        store.close()
        _print_runs(runs)
        return

    # repo_path is required for all other paths
    if args.repo_path is None:
        print("Error: repo_path is required", file=sys.stderr)
        sys.exit(1)

    # Validate mutual exclusivity
    if getattr(args, "at", None) and (args.from_ref or args.to_ref):
        print("Error: --at cannot be combined with --from/--to", file=sys.stderr)
        sys.exit(1)
    if args.from_ref and not args.to_ref:
        print("Error: --from requires --to", file=sys.stderr)
        sys.exit(1)
    if args.to_ref and not args.from_ref:
        print("Error: --to requires --from", file=sys.stderr)
        sys.exit(1)
    if args.run_all and (args.from_ref or args.to_ref):
        print("Error: --all cannot be combined with --from/--to", file=sys.stderr)
        sys.exit(1)

    try:
        git_reader = GitCliReader(args.repo_path)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Handle --from/--to comparison mode
    if args.from_ref and args.to_ref:
        try:
            report = compare_hotspots(
                git_reader, args.repo_path, args.window,
                args.from_ref, args.to_ref,
            )
        except (ValueError, RuntimeError) as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        _print_comparison(report)
        return

    # Resolve --at to current_time
    current_time = None
    snapshot_info = None
    if args.at:
        try:
            current_time = _resolve_ref_to_datetime(args.at, git_reader)
            snapshot_info = (args.at, current_time)
        except (ValueError, RuntimeError) as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    try:
        summary = get_repo_summary(git_reader, args.repo_path)
    except Exception as e:
        print(f"Error reading repository: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Repository:   {summary.repo_path}")
    if snapshot_info:
        ref_name, ref_date = snapshot_info
        print(f"Snapshot at:  {ref_name} ({ref_date.strftime('%Y-%m-%d %H:%M:%S %z')})")
    print(f"Commits:      {summary.commit_count}")
    if summary.first_commit_date:
        print(f"First commit: {summary.first_commit_date.strftime('%Y-%m-%d %H:%M:%S %z')}")
        print(f"Last commit:  {summary.last_commit_date.strftime('%Y-%m-%d %H:%M:%S %z')}")
    else:
        print("No commits found.")
        return

    # Handle --all: run all 8 analyses, print everything, store in DuckDB
    if args.run_all:
        _run_all(args, git_reader, summary, current_time)
        return

    # Hotspot analysis
    report = analyze_hotspots(git_reader, args.repo_path, args.window, current_time=current_time)

    print(f"\n--- Hotspot Analysis (last {report.window_days} days, {report.total_commits} commits) ---\n")

    if not report.files:
        print("No file changes found in this window.")
    else:
        _print_hotspots(report)
        print()
        _print_effort_distribution(report)

    if args.knowledge:
        knowledge_report = analyze_knowledge(git_reader, args.repo_path, args.window, current_time=current_time)
        print()
        _print_knowledge(knowledge_report)

    if args.coupling:
        coupling_report = analyze_coupling(git_reader, args.repo_path, args.window, current_time=current_time)
        print()
        _print_coupling(coupling_report)
        print()
        _print_pain(coupling_report)

    if args.anemia:
        source_reader = GitSourceReader(args.repo_path)
        ref = args.at if args.at else None
        anemia_report = analyze_anemia(source_reader, args.repo_path, ref=ref)
        print()
        _print_anemia(anemia_report)

    if args.complexity:
        source_reader = GitSourceReader(args.repo_path)
        ref = args.at if args.at else None
        complexity_report = analyze_complexity(source_reader, args.repo_path, ref=ref)
        print()
        _print_complexity(complexity_report)

    if args.clustering:
        clustering_report = analyze_change_clusters(
            git_reader, args.repo_path, args.window, current_time=current_time,
        )
        print()
        _print_clustering(clustering_report)

    if args.effort:
        effort_report = analyze_effort(
            git_reader, args.repo_path, args.window, current_time=current_time,
        )
        print()
        _print_effort(effort_report)

    if args.dx:
        source_reader = GitSourceReader(args.repo_path)
        dx_report = analyze_dx(
            git_reader, source_reader, args.repo_path, args.window,
            current_time=current_time,
        )
        print()
        _print_dx(dx_report)


def _run_all(args, git_reader, summary, current_time) -> None:
    """Run all 8 analyses, print all output, and store in DuckDB."""
    import uuid

    from git_xrays.infrastructure.run_store import RunStore

    source_reader = GitSourceReader(args.repo_path)
    ref = args.at if args.at else None

    # 1. Hotspots
    hotspot_report = analyze_hotspots(
        git_reader, args.repo_path, args.window, current_time=current_time,
    )
    print(f"\n--- Hotspot Analysis (last {hotspot_report.window_days} days, {hotspot_report.total_commits} commits) ---\n")
    if not hotspot_report.files:
        print("No file changes found in this window.")
    else:
        _print_hotspots(hotspot_report)
        print()
        _print_effort_distribution(hotspot_report)

    # 2. Knowledge
    knowledge_report = analyze_knowledge(
        git_reader, args.repo_path, args.window, current_time=current_time,
    )
    print()
    _print_knowledge(knowledge_report)

    # 3. Coupling
    coupling_report = analyze_coupling(
        git_reader, args.repo_path, args.window, current_time=current_time,
    )
    print()
    _print_coupling(coupling_report)
    print()
    _print_pain(coupling_report)

    # 4. Anemia
    anemia_report = analyze_anemia(source_reader, args.repo_path, ref=ref)
    print()
    _print_anemia(anemia_report)

    # 5. Complexity
    complexity_report = analyze_complexity(source_reader, args.repo_path, ref=ref)
    print()
    _print_complexity(complexity_report)

    # 6. Clustering
    clustering_report = analyze_change_clusters(
        git_reader, args.repo_path, args.window, current_time=current_time,
    )
    print()
    _print_clustering(clustering_report)

    # 7. Effort
    effort_report = analyze_effort(
        git_reader, args.repo_path, args.window, current_time=current_time,
    )
    print()
    _print_effort(effort_report)

    # 8. DX
    dx_report = analyze_dx(
        git_reader, source_reader, args.repo_path, args.window,
        current_time=current_time,
    )
    print()
    _print_dx(dx_report)

    # Store results
    run_id = str(uuid.uuid4())
    store = RunStore(db_path=args.db)
    store.save_run(
        run_id, args.repo_path, args.window, summary,
        hotspot_report, knowledge_report, coupling_report,
        anemia_report, complexity_report, clustering_report,
        effort_report, dx_report,
    )
    store.close()
    print(f"\nRun stored: {run_id}")


def _print_runs(runs: list[dict]) -> None:
    """Print past runs in a table."""
    if not runs:
        print("No runs found.")
        return

    header = (
        f"{'Run ID':<36}  "
        f"{'Repository':<30}  "
        f"{'Date':<19}  "
        f"{'Window':>6}  "
        f"{'Commits':>7}  "
        f"{'Files':>5}  "
        f"{'DX':>6}"
    )
    print(header)
    print("-" * len(header))

    for r in runs:
        created = r["created_at"]
        if hasattr(created, "strftime"):
            date_str = created.strftime("%Y-%m-%d %H:%M:%S")
        else:
            date_str = str(created)[:19]
        repo = r["repo_path"]
        if len(repo) > 30:
            repo = "..." + repo[-27:]
        print(
            f"{r['run_id']:<36}  "
            f"{repo:<30}  "
            f"{date_str:<19}  "
            f"{r['window_days']:>6}  "
            f"{r['total_commits']:>7}  "
            f"{r['hotspot_file_count']:>5}  "
            f"{r['dx_score']:>6.4f}"
        )


def _print_hotspots(report) -> None:
    # Column widths
    path_width = min(max(len(f.file_path) for f in report.files), 60)
    header = (
        f"{'File':<{path_width}}  "
        f"{'Freq':>4}  "
        f"{'Churn':>6}  "
        f"{'Hotspot':>7}  "
        f"{'Rework':>6}"
    )
    print(header)
    print("-" * len(header))

    for f in report.files[:20]:
        path = f.file_path
        if len(path) > path_width:
            path = "..." + path[-(path_width - 3):]
        print(
            f"{path:<{path_width}}  "
            f"{f.change_frequency:>4}  "
            f"{f.code_churn:>6}  "
            f"{f.hotspot_score:>7.4f}  "
            f"{f.rework_ratio:>6.2f}"
        )

    if len(report.files) > 20:
        print(f"  ... and {len(report.files) - 20} more files")


def _print_effort_distribution(report) -> None:
    total_churn = sum(f.code_churn for f in report.files)
    if total_churn == 0:
        return

    print("Effort Distribution:")
    cumulative = 0
    thresholds = [50, 80, 90]
    threshold_idx = 0

    for i, f in enumerate(report.files):
        cumulative += f.code_churn
        pct = cumulative / total_churn * 100
        while threshold_idx < len(thresholds) and pct >= thresholds[threshold_idx]:
            file_pct = (i + 1) / len(report.files) * 100
            print(
                f"  {thresholds[threshold_idx]}% of churn is in "
                f"{i + 1}/{len(report.files)} files ({file_pct:.0f}%)"
            )
            threshold_idx += 1

    if threshold_idx == 0:
        print(f"  All churn spread across {len(report.files)} files")


def _print_knowledge(report) -> None:
    print(f"--- Knowledge Analysis (last {report.window_days} days, {report.total_commits} commits) ---\n")

    print(f"Developer Risk Index: {report.developer_risk_index}")
    print(f"Knowledge Islands:    {report.knowledge_island_count}")
    print()

    if not report.files:
        print("No file changes found in this window.")
        return

    path_width = min(max(len(f.file_path) for f in report.files), 60)
    header = (
        f"{'File':<{path_width}}  "
        f"{'Concentration':>13}  "
        f"{'Primary Author':<20}  "
        f"{'Pct':>5}  "
        f"{'Authors':>7}  "
        f"{'Island':>6}"
    )
    print(header)
    print("-" * len(header))

    for f in report.files[:20]:
        path = f.file_path
        if len(path) > path_width:
            path = "..." + path[-(path_width - 3):]
        island_str = "Yes" if f.is_knowledge_island else "No"
        author = f.primary_author
        if len(author) > 20:
            author = author[:17] + "..."
        print(
            f"{path:<{path_width}}  "
            f"{f.knowledge_concentration:>13.4f}  "
            f"{author:<20}  "
            f"{f.primary_author_pct:>5.0%}  "
            f"{f.author_count:>7}  "
            f"{island_str:>6}"
        )

    if len(report.files) > 20:
        print(f"  ... and {len(report.files) - 20} more files")


def _print_coupling(report) -> None:
    print(f"--- Coupling Analysis (last {report.window_days} days, {report.total_commits} commits) ---\n")

    if not report.coupling_pairs:
        print("No coupling pairs found in this window.")
        return

    fa_width = min(max(len(p.file_a) for p in report.coupling_pairs), 40)
    fb_width = min(max(len(p.file_b) for p in report.coupling_pairs), 40)
    header = (
        f"{'File A':<{fa_width}}  "
        f"{'File B':<{fb_width}}  "
        f"{'Shared':>6}  "
        f"{'Strength':>8}  "
        f"{'Support':>7}"
    )
    print(header)
    print("-" * len(header))

    for p in report.coupling_pairs[:20]:
        fa = p.file_a
        if len(fa) > fa_width:
            fa = "..." + fa[-(fa_width - 3):]
        fb = p.file_b
        if len(fb) > fb_width:
            fb = "..." + fb[-(fb_width - 3):]
        print(
            f"{fa:<{fa_width}}  "
            f"{fb:<{fb_width}}  "
            f"{p.shared_commits:>6}  "
            f"{p.coupling_strength:>8.4f}  "
            f"{p.support:>7.4f}"
        )

    if len(report.coupling_pairs) > 20:
        print(f"  ... and {len(report.coupling_pairs) - 20} more pairs")


def _print_pain(report) -> None:
    print("--- PAIN Scores (Size x Distance x Volatility) ---\n")

    if not report.file_pain:
        print("No files found in this window.")
        return

    path_width = min(max(len(fp.file_path) for fp in report.file_pain), 60)
    header = (
        f"{'File':<{path_width}}  "
        f"{'Size':>6}  "
        f"{'Volatility':>10}  "
        f"{'Distance':>8}  "
        f"{'PAIN':>6}"
    )
    print(header)
    print("-" * len(header))

    for fp in report.file_pain[:20]:
        path = fp.file_path
        if len(path) > path_width:
            path = "..." + path[-(path_width - 3):]
        print(
            f"{path:<{path_width}}  "
            f"{fp.size_normalized:>6.4f}  "
            f"{fp.volatility_normalized:>10.4f}  "
            f"{fp.distance_normalized:>8.4f}  "
            f"{fp.pain_score:>6.4f}"
        )

    if len(report.file_pain) > 20:
        print(f"  ... and {len(report.file_pain) - 20} more files")


def _print_comparison(report) -> None:
    print(
        f"--- Hotspot Comparison: {report.from_ref} → {report.to_ref} "
        f"({report.window_days}-day window) ---"
    )
    print()
    print(
        f"From: {report.from_ref} "
        f"({report.from_date.strftime('%Y-%m-%d %H:%M:%S %z')}) "
        f"— {report.from_total_commits} commits"
    )
    print(
        f"To:   {report.to_ref} "
        f"({report.to_date.strftime('%Y-%m-%d %H:%M:%S %z')}) "
        f"— {report.to_total_commits} commits"
    )
    print()
    print(
        f"Summary: {report.new_hotspot_count} new, "
        f"{report.removed_hotspot_count} removed, "
        f"{report.improved_count} improved, "
        f"{report.degraded_count} degraded"
    )
    print()

    if not report.files:
        print("No file changes found in either snapshot.")
        return

    path_width = min(max(len(f.file_path) for f in report.files), 60)
    header = (
        f"{'File':<{path_width}}  "
        f"{'From':>6}  "
        f"{'To':>6}  "
        f"{'Delta':>7}  "
        f"{'Status':<10}"
    )
    print(header)
    print("-" * len(header))

    for f in report.files[:20]:
        path = f.file_path
        if len(path) > path_width:
            path = "..." + path[-(path_width - 3):]
        delta_str = f"{f.score_delta:+.4f}" if f.score_delta != 0 else " 0.0000"
        print(
            f"{path:<{path_width}}  "
            f"{f.from_score:>6.4f}  "
            f"{f.to_score:>6.4f}  "
            f"{delta_str:>7}  "
            f"{f.status:<10}"
        )

    if len(report.files) > 20:
        print(f"  ... and {len(report.files) - 20} more files")


def _print_anemia(report) -> None:
    print(
        f"--- Anemia Analysis "
        f"({report.total_classes} classes in {report.total_files} files) ---\n"
    )

    print(f"Total classes:       {report.total_classes}")
    print(f"Anemic classes:      {report.anemic_count} ({report.anemic_percentage}%)")
    print(f"Average AMS:         {report.average_ams:.4f}")
    print()

    if not report.files:
        print("No Python files found.")
        return

    path_width = min(max(len(f.file_path) for f in report.files), 60)
    header = (
        f"{'File':<{path_width}}  "
        f"{'Classes':>7}  "
        f"{'Anemic':>6}  "
        f"{'Worst AMS':>9}"
    )
    print(header)
    print("-" * len(header))

    for f in report.files[:20]:
        path = f.file_path
        if len(path) > path_width:
            path = "..." + path[-(path_width - 3):]
        print(
            f"{path:<{path_width}}  "
            f"{f.class_count:>7}  "
            f"{f.anemic_class_count:>6}  "
            f"{f.worst_ams:>9.4f}"
        )

    if len(report.files) > 20:
        print(f"  ... and {len(report.files) - 20} more files")


def _print_complexity(report) -> None:
    print(
        f"--- Complexity Analysis "
        f"({report.total_functions} functions in {report.total_files} files) ---\n"
    )

    print(f"Total functions:     {report.total_functions}")
    print(
        f"High complexity:     {report.high_complexity_count} "
        f"(above threshold {report.complexity_threshold})"
    )
    print(f"Avg complexity:      {report.avg_complexity:.2f}")
    print(f"Max complexity:      {report.max_complexity}")
    print()

    if not report.files:
        print("No Python files with functions found.")
        return

    path_width = min(max(len(f.file_path) for f in report.files), 60)
    header = (
        f"{'File':<{path_width}}  "
        f"{'Functions':>9}  "
        f"{'Max CC':>6}  "
        f"{'Avg CC':>6}  "
        f"{'Max Depth':>9}  "
        f"{'Max Len':>7}"
    )
    print(header)
    print("-" * len(header))

    for f in report.files[:20]:
        path = f.file_path
        if len(path) > path_width:
            path = "..." + path[-(path_width - 3):]
        print(
            f"{path:<{path_width}}  "
            f"{f.function_count:>9}  "
            f"{f.max_complexity:>6}  "
            f"{f.avg_complexity:>6.2f}  "
            f"{f.max_nesting:>9}  "
            f"{f.max_length:>7}"
        )

    if len(report.files) > 20:
        print(f"  ... and {len(report.files) - 20} more files")


def _print_clustering(report) -> None:
    print(
        f"--- Clustering Analysis "
        f"(last {report.window_days} days, {report.total_commits} commits) ---\n"
    )

    if not report.clusters:
        print("No commits found in this window.")
        return

    print(f"Clusters:        {report.k}")
    print(f"Silhouette:      {report.silhouette_score:.4f}")
    print()

    header = (
        f"{'Label':<15}  "
        f"{'Size':>4}  "
        f"{'Avg Files':>9}  "
        f"{'Avg Churn':>9}  "
        f"{'Add Ratio':>9}"
    )
    print(header)
    print("-" * len(header))

    for c in report.clusters:
        print(
            f"{c.label:<15}  "
            f"{c.size:>4}  "
            f"{c.centroid_file_count:>9.1f}  "
            f"{c.centroid_total_churn:>9.1f}  "
            f"{c.centroid_add_ratio:>9.2f}"
        )

    if report.drift:
        print()
        print("--- Cluster Drift (first half vs second half) ---\n")

        drift_header = (
            f"{'Label':<15}  "
            f"{'1st Half':>8}  "
            f"{'2nd Half':>8}  "
            f"{'Drift':>6}  "
            f"{'Trend':<10}"
        )
        print(drift_header)
        print("-" * len(drift_header))

        for d in report.drift:
            drift_str = f"{d.drift:+.1f}" if d.drift != 0 else "  0.0"
            print(
                f"{d.cluster_label:<15}  "
                f"{d.first_half_pct:>7.1f}%  "
                f"{d.second_half_pct:>7.1f}%  "
                f"{drift_str:>6}  "
                f"{d.trend:<10}"
            )


def _print_effort(report) -> None:
    print(
        f"--- Effort Analysis "
        f"(last {report.window_days} days, {report.total_files} files) ---\n"
    )

    print(f"Model R²:        {report.model_r_squared:.4f}")
    print(f"Alpha:           {report.alpha}")
    print()

    if not report.files:
        print("No files found in this window.")
        return

    path_width = min(max(len(f.file_path) for f in report.files), 60)
    header = (
        f"{'File':<{path_width}}  "
        f"{'REI':>6}  "
        f"{'Proxy':>6}  "
        f"{'Top Factor':<25}"
    )
    print(header)
    print("-" * len(header))

    for f in report.files[:20]:
        path = f.file_path
        if len(path) > path_width:
            path = "..." + path[-(path_width - 3):]
        top_factor = f.attributions[0].feature_name if f.attributions else ""
        print(
            f"{path:<{path_width}}  "
            f"{f.rei_score:>6.4f}  "
            f"{f.proxy_label:>6.4f}  "
            f"{top_factor:<25}"
        )

    if len(report.files) > 20:
        print(f"  ... and {len(report.files) - 20} more files")


def _print_dx(report) -> None:
    print(
        f"--- Developer Experience Analysis "
        f"(last {report.window_days} days, "
        f"{report.total_commits} commits, "
        f"{report.total_files} files) ---\n"
    )

    print(f"DX Score:            {report.dx_score:.4f}")
    print()
    print("Core Metrics:")
    print(f"  Throughput:        {report.metrics.throughput:.4f}  (weighted commit rate)")
    print(f"  Feedback Delay:    {report.metrics.feedback_delay:.4f}  (iteration speed)")
    print(f"  Focus Ratio:       {report.metrics.focus_ratio:.4f}  (feature vs toil)")
    print(f"  Cognitive Load:    {report.metrics.cognitive_load:.4f}  (lower is better)")

    if not report.cognitive_load_files:
        return

    print()
    print("Top Cognitive Load Files:")

    files = report.cognitive_load_files[:10]
    path_width = min(max(len(f.file_path) for f in files), 40)
    header = (
        f"{'File':<{path_width}}  "
        f"{'Complexity':>10}  "
        f"{'Coordination':>12}  "
        f"{'Knowledge':>9}  "
        f"{'ChangeRate':>10}  "
        f"{'Load':>6}"
    )
    print(header)
    print("-" * len(header))

    for f in files:
        path = f.file_path
        if len(path) > path_width:
            path = "..." + path[-(path_width - 3):]
        print(
            f"{path:<{path_width}}  "
            f"{f.complexity_score:>10.4f}  "
            f"{f.coordination_score:>12.4f}  "
            f"{f.knowledge_score:>9.4f}  "
            f"{f.change_rate_score:>10.4f}  "
            f"{f.composite_load:>6.4f}"
        )

    if len(report.cognitive_load_files) > 10:
        print(f"  ... and {len(report.cognitive_load_files) - 10} more files")
