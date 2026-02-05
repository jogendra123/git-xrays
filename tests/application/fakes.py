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
    ) -> None:
        self._commit_count = commit_count_val
        self._first_commit_date = first_commit_date_val
        self._last_commit_date = last_commit_date_val
        self._file_changes = file_changes_val or []

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
