import argparse
import re
import sys

from git_xrays.application.use_cases import analyze_coupling, analyze_hotspots, analyze_knowledge, get_repo_summary
from git_xrays.infrastructure.git_cli_reader import GitCliReader


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
    parser.add_argument("repo_path", help="Path to a local git repository")
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
    args = parser.parse_args()

    try:
        git_reader = GitCliReader(args.repo_path)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        summary = get_repo_summary(git_reader, args.repo_path)
    except Exception as e:
        print(f"Error reading repository: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Repository:   {summary.repo_path}")
    print(f"Commits:      {summary.commit_count}")
    if summary.first_commit_date:
        print(f"First commit: {summary.first_commit_date.strftime('%Y-%m-%d %H:%M:%S %z')}")
        print(f"Last commit:  {summary.last_commit_date.strftime('%Y-%m-%d %H:%M:%S %z')}")
    else:
        print("No commits found.")
        return

    # Hotspot analysis
    report = analyze_hotspots(git_reader, args.repo_path, args.window)

    print(f"\n--- Hotspot Analysis (last {report.window_days} days, {report.total_commits} commits) ---\n")

    if not report.files:
        print("No file changes found in this window.")
    else:
        _print_hotspots(report)
        print()
        _print_effort_distribution(report)

    if args.knowledge:
        knowledge_report = analyze_knowledge(git_reader, args.repo_path, args.window)
        print()
        _print_knowledge(knowledge_report)

    if args.coupling:
        coupling_report = analyze_coupling(git_reader, args.repo_path, args.window)
        print()
        _print_coupling(coupling_report)
        print()
        _print_pain(coupling_report)


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
