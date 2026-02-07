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

    def resolve_ref(self, ref: str) -> datetime:
        """Resolve a git ref (commit, tag, branch) to its author date."""
        try:
            output = self._run("log", "-1", "--format=%aI", ref)
        except RuntimeError:
            raise ValueError(f"Cannot resolve ref: {ref}")
        if not output:
            raise ValueError(f"Cannot resolve ref: {ref}")
        return datetime.fromisoformat(output.strip())

    def file_sizes(self, ref: str | None = None) -> dict[str, int]:
        """Return mapping of file path → size in bytes using git ls-tree."""
        tree_ref = ref or "HEAD"
        try:
            output = self._run("ls-tree", "-r", "-l", tree_ref)
        except RuntimeError:
            return {}
        result: dict[str, int] = {}
        for line in output.splitlines():
            # Format: <mode> <type> <hash> <size>\t<path>
            parts = line.split(None, 4)
            if len(parts) < 5:
                continue
            size_str = parts[3]
            file_path = parts[4]
            if size_str == "-":
                continue
            result[file_path] = int(size_str)
        return result

    def file_changes(
        self, since: datetime | None = None, until: datetime | None = None
    ) -> list[FileChange]:
        args = ["log", "--numstat", "--format=COMMIT:%H %aI %aN %aE"]
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
    current_author_name = ""
    current_author_email = ""

    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("COMMIT:"):
            tokens = line[7:].split(" ")
            current_hash = tokens[0]
            current_date = datetime.fromisoformat(tokens[1])
            current_author_email = tokens[-1]
            current_author_name = " ".join(tokens[2:-1])
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
                author_name=current_author_name,
                author_email=current_author_email,
            )
        )
    return changes
