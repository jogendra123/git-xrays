import subprocess
from datetime import datetime
from pathlib import Path

from git_xrays.domain.models import FileChange


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
            output = self._run("log", "--format=%aI")
        except RuntimeError:
            return None
        if not output:
            return None
        # Last line is the earliest commit (git log outputs newest first)
        return datetime.fromisoformat(output.splitlines()[-1])

    def last_commit_date(self) -> datetime | None:
        try:
            output = self._run("log", "--format=%aI", "--max-count=1")
        except RuntimeError:
            return None
        if not output:
            return None
        return datetime.fromisoformat(output)

    def file_changes(
        self, since: datetime | None = None, until: datetime | None = None
    ) -> list[FileChange]:
        args = ["log", "--numstat", "--format=COMMIT:%H %aI"]
        if since:
            args.append(f"--since={since.isoformat()}")
        if until:
            args.append(f"--until={until.isoformat()}")
        try:
            output = self._run(*args)
        except RuntimeError:
            return []
        if not output:
            return []
        return _parse_numstat(output)


def _parse_numstat(output: str) -> list[FileChange]:
    changes: list[FileChange] = []
    current_hash = ""
    current_date = datetime.min

    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("COMMIT:"):
            parts = line[7:].split(" ", 1)
            current_hash = parts[0]
            current_date = datetime.fromisoformat(parts[1])
            continue
        # numstat line: <added>\t<deleted>\t<file>
        parts = line.split("\t", 2)
        if len(parts) != 3:
            continue
        added_str, deleted_str, file_path = parts
        # Binary files show "-" for added/deleted
        if added_str == "-" or deleted_str == "-":
            continue
        changes.append(
            FileChange(
                commit_hash=current_hash,
                date=current_date,
                file_path=file_path,
                lines_added=int(added_str),
                lines_deleted=int(deleted_str),
            )
        )
    return changes
