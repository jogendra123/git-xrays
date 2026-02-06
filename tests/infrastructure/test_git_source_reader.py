from pathlib import Path

import pytest

from git_xrays.infrastructure.git_source_reader import GitSourceReader


class TestGitSourceReader:
    def test_list_python_files_finds_py_files(self, anemic_repo: Path):
        reader = GitSourceReader(str(anemic_repo))
        files = reader.list_python_files()
        assert any(f.endswith(".py") for f in files)

    def test_list_python_files_excludes_non_py(self, anemic_repo: Path):
        reader = GitSourceReader(str(anemic_repo))
        files = reader.list_python_files()
        assert all(f.endswith(".py") for f in files)

    def test_read_file_returns_content(self, anemic_repo: Path):
        reader = GitSourceReader(str(anemic_repo))
        files = reader.list_python_files()
        content = reader.read_file(files[0])
        assert len(content) > 0

    def test_read_file_at_ref(self, anemic_repo: Path):
        reader = GitSourceReader(str(anemic_repo))
        files = reader.list_python_files(ref="HEAD")
        assert len(files) > 0
        content = reader.read_file(files[0], ref="HEAD")
        assert len(content) > 0
