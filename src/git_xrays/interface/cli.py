import argparse
import re
import sys

from git_xrays.application.use_cases import (
    _resolve_ref_to_datetime,
    analyze_anemic,
    analyze_change_clusters,
    analyze_complexity,
    analyze_coupling,
    analyze_dx,
    analyze_effort,
    analyze_god_classes,
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


def _error_exit(msg: str) -> None:
    """Print error message to stderr and exit with code 1."""
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(1)


def _header_fmt(fmt_spec: str) -> str:
    """Extract header-safe format from a value format spec.

    E.g. ">7.4f" → ">7", "<25" → "<25", ">6.2f" → ">6".
    Strips trailing type char and precision so it works for str headers.
    """
    # Remove trailing type characters (d, f, %)
    stripped = fmt_spec.rstrip("df%")
    # Remove precision (.N)
    dot = stripped.find(".")
    if dot != -1:
        stripped = stripped[:dot]
    return stripped


def _print_table(rows, columns, limit=20, path_attr="file_path", max_path=60, suffix="files") -> None:
    """Generic table printer.

    Args:
        rows: list of objects to print.
        columns: list of (header, format_spec, value_fn) tuples.
            - header: column header string
            - format_spec: value format string like ">6" or ">7.4f".
              Use None for the path column (auto-computed).
            - value_fn: callable(row) -> value to format, or None for path column.
        limit: max rows to print.
        path_attr: attribute name for the file path on each row.
        max_path: max path column width.
        suffix: word used in "... and N more {suffix}" message.
    """
    if not rows:
        return

    # Compute path width from all rows (not just displayed ones)
    path_width = min(max(len(getattr(r, path_attr)) for r in rows), max_path)

    # Build header
    parts = []
    for header, fmt_spec, _value_fn in columns:
        if fmt_spec is None:
            parts.append(f"{header:<{path_width}}")
        else:
            parts.append(f"{header:{_header_fmt(fmt_spec)}}")
    header_line = "  ".join(parts)
    print(header_line)
    print("-" * len(header_line))

    # Print rows
    for r in rows[:limit]:
        parts = []
        for _header, fmt_spec, value_fn in columns:
            if fmt_spec is None:
                path = getattr(r, path_attr)
                if len(path) > path_width:
                    path = "..." + path[-(path_width - 3):]
                parts.append(f"{path:<{path_width}}")
            else:
                parts.append(f"{value_fn(r):{fmt_spec}}")
        print("  ".join(parts))

    if len(rows) > limit:
        print(f"  ... and {len(rows) - limit} more {suffix}")


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
        "--anemic",
        action="store_true",
        help="Show anemic domain model analysis",
    )
    parser.add_argument(
        "--complexity",
        action="store_true",
        help="Show function-level complexity analysis",
    )
    parser.add_argument(
        "--god-class",
        dest="god_class",
        action="store_true",
        help="Show god class detection analysis",
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
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Launch web dashboard (requires pip install git-xrays[web])",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        metavar="PORT",
        help="API port for --serve (Streamlit uses PORT+1, default: 8000)",
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

    # Handle --serve early (no repo_path required)
    if args.serve:
        try:
            from git_xrays.web.server import launch
        except ImportError:
            _error_exit(
                "web dependencies not installed. "
                "Run: pip install git-xrays[web]"
            )
        launch(db_path=args.db, api_port=args.port)
        return

    # repo_path is required for all other paths
    if args.repo_path is None:
        _error_exit("repo_path is required")

    # Validate mutual exclusivity
    if getattr(args, "at", None) and (args.from_ref or args.to_ref):
        _error_exit("--at cannot be combined with --from/--to")
    if args.from_ref and not args.to_ref:
        _error_exit("--from requires --to")
    if args.to_ref and not args.from_ref:
        _error_exit("--to requires --from")
    if args.run_all and (args.from_ref or args.to_ref):
        _error_exit("--all cannot be combined with --from/--to")

    try:
        git_reader = GitCliReader(args.repo_path)
    except ValueError as e:
        _error_exit(str(e))

    # Handle --from/--to comparison mode
    if args.from_ref and args.to_ref:
        try:
            report = compare_hotspots(
                git_reader, args.repo_path, args.window,
                args.from_ref, args.to_ref,
            )
        except (ValueError, RuntimeError) as e:
            _error_exit(str(e))
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
            _error_exit(str(e))

    try:
        summary = get_repo_summary(git_reader, args.repo_path)
    except Exception as e:
        _error_exit(f"reading repository: {e}")

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

    if args.anemic:
        source_reader = GitSourceReader(args.repo_path)
        ref = args.at if args.at else None
        anemic_report = analyze_anemic(source_reader, args.repo_path, ref=ref)
        print()
        _print_anemic(anemic_report)

    if args.complexity:
        source_reader = GitSourceReader(args.repo_path)
        ref = args.at if args.at else None
        complexity_report = analyze_complexity(source_reader, args.repo_path, ref=ref)
        print()
        _print_complexity(complexity_report)

    if args.god_class:
        source_reader = GitSourceReader(args.repo_path)
        ref = args.at if args.at else None
        god_class_report = analyze_god_classes(source_reader, args.repo_path, ref=ref)
        print()
        _print_god_classes(god_class_report)

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

    # 4. Anemic
    anemic_report = analyze_anemic(source_reader, args.repo_path, ref=ref)
    print()
    _print_anemic(anemic_report)

    # 5. Complexity
    complexity_report = analyze_complexity(source_reader, args.repo_path, ref=ref)
    print()
    _print_complexity(complexity_report)

    # 5b. God Classes
    god_class_report = analyze_god_classes(source_reader, args.repo_path, ref=ref)
    print()
    _print_god_classes(god_class_report)

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
        anemic_report, complexity_report, god_class_report,
        clustering_report, effort_report, dx_report,
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
    _print_table(
        report.files,
        [
            ("File", None, None),
            ("Freq", ">4", lambda f: f.change_frequency),
            ("Churn", ">6", lambda f: f.code_churn),
            ("Hotspot", ">7.4f", lambda f: f.hotspot_score),
            ("Rework", ">6.2f", lambda f: f.rework_ratio),
        ],
    )


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

    print(f"Developer Risk Index: {report.developer_risk_index:.4f}")
    print(f"Knowledge Islands:    {report.knowledge_island_count}")
    print()

    if not report.files:
        print("No file changes found in this window.")
        return

    _print_table(
        report.files,
        [
            ("File", None, None),
            ("Concentration", ">13.4f", lambda f: f.knowledge_concentration),
            ("Primary Author", "<20", lambda f: (f.primary_author[:17] + "...") if len(f.primary_author) > 20 else f.primary_author),
            ("Pct", ">5.0%", lambda f: f.primary_author_pct),
            ("Authors", ">7", lambda f: f.author_count),
            ("Island", ">6", lambda f: "Yes" if f.is_knowledge_island else "No"),
        ],
    )


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
        f"{'Support':>7}  "
        f"{'Lift':>6}"
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
            f"{p.support:>7.4f}  "
            f"{p.lift:>6.2f}"
        )

    if len(report.coupling_pairs) > 20:
        print(f"  ... and {len(report.coupling_pairs) - 20} more pairs")


def _print_pain(report) -> None:
    print("--- PAIN Scores (Size x Distance x Volatility) ---\n")

    if not report.file_pain:
        print("No files found in this window.")
        return

    _print_table(
        report.file_pain,
        [
            ("File", None, None),
            ("Size", ">6.4f", lambda fp: fp.size_normalized),
            ("Volatility", ">10.4f", lambda fp: fp.volatility_normalized),
            ("Distance", ">8.4f", lambda fp: fp.distance_normalized),
            ("PAIN", ">6.4f", lambda fp: fp.pain_score),
        ],
    )


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

    _print_table(
        report.files,
        [
            ("File", None, None),
            ("From", ">6.4f", lambda f: f.from_score),
            ("To", ">6.4f", lambda f: f.to_score),
            ("Delta", ">7", lambda f: f"{f.score_delta:+.4f}" if f.score_delta != 0 else " 0.0000"),
            ("Status", "<10", lambda f: f.status),
        ],
    )


def _print_anemic(report) -> None:
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

    _print_table(
        report.files,
        [
            ("File", None, None),
            ("Classes", ">7", lambda f: f.class_count),
            ("Anemic", ">6", lambda f: f.anemic_class_count),
            ("Worst AMS", ">9.4f", lambda f: f.worst_ams),
        ],
    )


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

    _print_table(
        report.files,
        [
            ("File", None, None),
            ("Functions", ">9", lambda f: f.function_count),
            ("Max CC", ">6", lambda f: f.max_complexity),
            ("Avg CC", ">6.2f", lambda f: f.avg_complexity),
            ("Max CogC", ">8", lambda f: f.max_cognitive),
            ("Avg CogC", ">8.2f", lambda f: f.avg_cognitive),
            ("Max Depth", ">9", lambda f: f.max_nesting),
            ("Max Len", ">7", lambda f: f.max_length),
        ],
    )


def _print_god_classes(report) -> None:
    print(
        f"--- God Class Analysis "
        f"({report.total_classes} classes in {report.total_files} files) ---\n"
    )

    print(f"Total classes:       {report.total_classes}")
    print(
        f"God classes:         {report.god_class_count} "
        f"({report.god_class_percentage}%)"
    )
    print(f"Average GCS:         {report.average_gcs:.4f}")
    print()

    if not report.files:
        print("No source files with classes found.")
        return

    _print_table(
        report.files,
        [
            ("File", None, None),
            ("Classes", ">7", lambda f: f.class_count),
            ("God", ">3", lambda f: f.god_class_count),
            ("Worst GCS", ">9.4f", lambda f: f.worst_gcs),
        ],
    )

    # List individual god classes
    god_classes = [
        c for f in report.files for c in f.classes
        if c.god_class_score > report.gcs_threshold
    ]
    god_classes.sort(key=lambda c: c.god_class_score, reverse=True)

    if god_classes:
        print()
        print(f"God Classes (GCS > {report.gcs_threshold}):\n")

        path_w = min(max(len(c.file_path) for c in god_classes), 40)
        name_w = min(max(len(c.class_name) for c in god_classes), 25)
        header = (
            f"{'File':<{path_w}}  "
            f"{'Class':<{name_w}}  "
            f"{'Methods':>7}  "
            f"{'Fields':>6}  "
            f"{'WMC':>5}  "
            f"{'Cohesion':>8}  "
            f"{'GCS':>6}"
        )
        print(header)
        print("-" * len(header))

        for c in god_classes[:20]:
            fp = c.file_path
            if len(fp) > path_w:
                fp = "..." + fp[-(path_w - 3):]
            cn = c.class_name
            if len(cn) > name_w:
                cn = cn[:name_w - 3] + "..."
            print(
                f"{fp:<{path_w}}  "
                f"{cn:<{name_w}}  "
                f"{c.method_count:>7}  "
                f"{c.field_count:>6}  "
                f"{c.total_complexity:>5}  "
                f"{c.cohesion:>8.4f}  "
                f"{c.god_class_score:>6.4f}"
            )

        if len(god_classes) > 20:
            print(f"  ... and {len(god_classes) - 20} more god classes")


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
    print(f"Alpha:           {report.alpha} (auto-tuned)" if isinstance(report.alpha, float) else f"Alpha:           {report.alpha}")
    print()

    if not report.files:
        print("No files found in this window.")
        return

    _print_table(
        report.files,
        [
            ("File", None, None),
            ("REI", ">6.4f", lambda f: f.rei_score),
            ("Proxy", ">6.4f", lambda f: f.proxy_label),
            ("Top Factor", "<25", lambda f: f.attributions[0].feature_name if f.attributions else ""),
        ],
    )


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

    _print_table(
        report.cognitive_load_files,
        [
            ("File", None, None),
            ("Complexity", ">10.4f", lambda f: f.complexity_score),
            ("Coordination", ">12.4f", lambda f: f.coordination_score),
            ("Knowledge", ">9.4f", lambda f: f.knowledge_score),
            ("ChangeRate", ">10.4f", lambda f: f.change_rate_score),
            ("Load", ">6.4f", lambda f: f.composite_load),
        ],
        limit=10,
        max_path=40,
    )
