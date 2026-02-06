from datetime import datetime

from git_xrays.domain.models import FileChange


class FakeGitRepository:
    """Plain stub implementing the GitRepository protocol."""

    def __init__(
        self,
        commit_count_val: int = 0,
        first_commit_date_val: datetime | None = None,
        last_commit_date_val: datetime | None = None,
        file_changes_val: list[FileChange] | None = None,
        ref_dates: dict[str, datetime] | None = None,
    ) -> None:
        self._commit_count = commit_count_val
        self._first_commit_date = first_commit_date_val
        self._last_commit_date = last_commit_date_val
        self._file_changes = file_changes_val or []
        self._ref_dates = ref_dates or {}

    def commit_count(self) -> int:
        return self._commit_count

    def first_commit_date(self) -> datetime | None:
        return self._first_commit_date

    def last_commit_date(self) -> datetime | None:
        return self._last_commit_date

    def file_changes(
        self, since: datetime | None = None, until: datetime | None = None
    ) -> list[FileChange]:
        result = self._file_changes
        if since:
            result = [c for c in result if c.date >= since]
        if until:
            result = [c for c in result if c.date <= until]
        return result

    def resolve_ref(self, ref: str) -> datetime:
        if ref not in self._ref_dates:
            raise ValueError(f"Unknown ref: {ref}")
        return self._ref_dates[ref]


class FakeSourceCodeReader:
    """Plain stub implementing the SourceCodeReader protocol."""

    def __init__(self, files: dict[str, str] | None = None) -> None:
        self._files = files or {}

    def list_python_files(self, ref: str | None = None) -> list[str]:
        return sorted(self._files.keys())

    def read_file(self, file_path: str, ref: str | None = None) -> str:
        if file_path not in self._files:
            raise FileNotFoundError(f"File not found: {file_path}")
        return self._files[file_path]
