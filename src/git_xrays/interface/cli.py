import argparse
import sys

from git_xrays.application.use_cases import get_repo_summary
from git_xrays.infrastructure.git_cli_reader import GitCliReader


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="analyze-repo",
        description="Behavioral & Architectural Code Intelligence",
    )
    parser.add_argument("repo_path", help="Path to a local git repository")
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
