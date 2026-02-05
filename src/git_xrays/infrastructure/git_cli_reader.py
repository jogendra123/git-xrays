import subprocess
from datetime import datetime
from pathlib import Path


class GitCliReader:
    def __init__(self, repo_path: str) -> None:
        path = Path(repo_path).resolve()
        if not (path / ".git").is_dir():
            raise ValueError(f"Not a git repository: {path}")
        self._path = str(path)

    def _run(self, *args: str) -> str:
        result = subprocess.run(
            ["git", "-C", self._path, *args],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip())
        return result.stdout.strip()

    def commit_count(self) -> int:
        try:
            output = self._run("rev-list", "--count", "HEAD")
        except RuntimeError:
            return 0
        return int(output)

    def first_commit_date(self) -> datetime | None:
        try:
            output = self._run(
                "log", "--reverse", "--format=%aI", "--max-count=1"
            )
        except RuntimeError:
            return None
        if not output:
            return None
        return datetime.fromisoformat(output)

    def last_commit_date(self) -> datetime | None:
        try:
            output = self._run("log", "--format=%aI", "--max-count=1")
        except RuntimeError:
            return None
        if not output:
            return None
        return datetime.fromisoformat(output)
