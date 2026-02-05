from git_xrays.domain.models import RepoSummary
from git_xrays.domain.ports import GitRepository


def get_repo_summary(repo: GitRepository, repo_path: str) -> RepoSummary:
    return RepoSummary(
        repo_path=repo_path,
        commit_count=repo.commit_count(),
        first_commit_date=repo.first_commit_date(),
        last_commit_date=repo.last_commit_date(),
    )
