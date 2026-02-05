import argparse
import re
import sys

from git_xrays.application.use_cases import analyze_hotspots, get_repo_summary
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
        return

    _print_hotspots(report)
    print()
    _print_effort_distribution(report)


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
