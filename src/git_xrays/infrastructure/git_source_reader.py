"""Git-backed SourceCodeReader using git ls-tree and git show."""

from __future__ import annotations

import subprocess
from pathlib import Path


class GitSourceReader:
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
        return result.stdout

    def list_python_files(self, ref: str | None = None) -> list[str]:
        tree_ref = ref or "HEAD"
        output = self._run("ls-tree", "-r", "--name-only", tree_ref)
        return sorted(
            line for line in output.splitlines()
            if line.endswith(".py")
        )

    def read_file(self, file_path: str, ref: str | None = None) -> str:
        tree_ref = ref or "HEAD"
        return self._run("show", f"{tree_ref}:{file_path}")
