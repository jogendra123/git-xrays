import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


@pytest.fixture
def tmp_git_repo(tmp_path: Path) -> Path:
    """Create a temporary git repository."""
    subprocess.run(
        ["git", "init", str(tmp_path)],
        capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "Test User"],
        capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "test@example.com"],
        capture_output=True, check=True,
    )
    return tmp_path


def commit_file(
    repo: Path,
    file_path: str,
    content: str,
    message: str,
    days_ago: int = 0,
) -> None:
    """Create a commit at a known relative date."""
    full_path = repo / file_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(content)

    date = datetime.now(timezone.utc) - timedelta(days=days_ago)
    date_str = date.strftime("%Y-%m-%dT%H:%M:%S %z")

    subprocess.run(
        ["git", "-C", str(repo), "add", file_path],
        capture_output=True, check=True,
    )
    env = {
        **os.environ,
        "GIT_AUTHOR_DATE": date_str,
        "GIT_COMMITTER_DATE": date_str,
        "GIT_AUTHOR_NAME": "Test User",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "Test User",
        "GIT_COMMITTER_EMAIL": "test@example.com",
    }
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-m", message],
        capture_output=True, check=True,
        env=env,
    )


@pytest.fixture
def git_repo_with_history(tmp_git_repo: Path) -> Path:
    """Create a repo with 5 commits across 3 files over 60 days."""
    commit_file(tmp_git_repo, "README.md", "# Project\n", "Initial commit", days_ago=60)
    commit_file(tmp_git_repo, "src/main.py", "print('hello')\n", "Add main", days_ago=45)
    commit_file(tmp_git_repo, "src/utils.py", "def helper(): pass\n", "Add utils", days_ago=30)
    commit_file(tmp_git_repo, "src/main.py", "print('hello world')\n", "Update main", days_ago=15)
    commit_file(tmp_git_repo, "README.md", "# Project\nUpdated.\n", "Update README", days_ago=5)
    return tmp_git_repo
