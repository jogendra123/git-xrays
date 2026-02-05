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
    author_name: str = "Test User",
    author_email: str = "test@example.com",
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
        "GIT_AUTHOR_NAME": author_name,
        "GIT_AUTHOR_EMAIL": author_email,
        "GIT_COMMITTER_NAME": author_name,
        "GIT_COMMITTER_EMAIL": author_email,
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


@pytest.fixture
def multi_author_repo(tmp_git_repo: Path) -> Path:
    """Create a repo with 3 authors, 3 files, 9 commits.

    Alice: dominates main.py (3 commits), 1 utils.py commit
    Bob:   1 main.py commit, dominates config.py (2 commits)
    Carol: 1 utils.py commit, 1 config.py commit
    """
    # Alice creates main.py (commit 1)
    commit_file(tmp_git_repo, "main.py", "print('v1')\n", "Alice: create main",
                days_ago=30, author_name="Alice", author_email="alice@example.com")
    # Alice updates main.py (commit 2)
    commit_file(tmp_git_repo, "main.py", "print('v2')\n", "Alice: update main",
                days_ago=25, author_name="Alice", author_email="alice@example.com")
    # Alice updates main.py again (commit 3)
    commit_file(tmp_git_repo, "main.py", "print('v3')\n", "Alice: update main again",
                days_ago=20, author_name="Alice", author_email="alice@example.com")
    # Bob touches main.py (commit 4)
    commit_file(tmp_git_repo, "main.py", "print('v4')\n", "Bob: touch main",
                days_ago=18, author_name="Bob", author_email="bob@example.com")
    # Alice adds utils.py (commit 5)
    commit_file(tmp_git_repo, "utils.py", "def util(): pass\n", "Alice: add utils",
                days_ago=15, author_name="Alice", author_email="alice@example.com")
    # Carol adds to utils.py (commit 6)
    commit_file(tmp_git_repo, "utils.py", "def util(): pass\ndef other(): pass\n",
                "Carol: add to utils", days_ago=12,
                author_name="Carol", author_email="carol@example.com")
    # Bob creates config.py (commit 7)
    commit_file(tmp_git_repo, "config.py", "DEBUG=True\n", "Bob: create config",
                days_ago=10, author_name="Bob", author_email="bob@example.com")
    # Bob updates config.py (commit 8)
    commit_file(tmp_git_repo, "config.py", "DEBUG=False\n", "Bob: update config",
                days_ago=5, author_name="Bob", author_email="bob@example.com")
    # Carol touches config.py (commit 9)
    commit_file(tmp_git_repo, "config.py", "DEBUG=False\nVERBOSE=True\n",
                "Carol: touch config", days_ago=2,
                author_name="Carol", author_email="carol@example.com")
    return tmp_git_repo
