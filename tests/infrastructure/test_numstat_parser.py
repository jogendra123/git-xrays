from datetime import datetime, timezone

from git_xrays.infrastructure.git_cli_reader import _parse_numstat


class TestParseNumstat:
    def test_single_commit_single_file(self):
        output = "COMMIT:abc123 2024-06-01T12:00:00+00:00 Alice alice@example.com\n10\t5\tsrc/main.py"
        result = _parse_numstat(output)
        assert len(result) == 1
        assert result[0].commit_hash == "abc123"
        assert result[0].file_path == "src/main.py"
        assert result[0].lines_added == 10
        assert result[0].lines_deleted == 5
        assert result[0].date == datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        assert result[0].author_name == "Alice"
        assert result[0].author_email == "alice@example.com"

    def test_multiple_commits_multiple_files(self):
        output = (
            "COMMIT:aaa 2024-06-01T12:00:00+00:00 Alice alice@example.com\n"
            "10\t2\ta.py\n"
            "5\t1\tb.py\n"
            "\n"
            "COMMIT:bbb 2024-05-01T10:00:00+00:00 Bob bob@example.com\n"
            "3\t0\tc.py"
        )
        result = _parse_numstat(output)
        assert len(result) == 3
        assert result[0].commit_hash == "aaa"
        assert result[0].file_path == "a.py"
        assert result[0].author_name == "Alice"
        assert result[0].author_email == "alice@example.com"
        assert result[1].commit_hash == "aaa"
        assert result[1].file_path == "b.py"
        assert result[1].author_name == "Alice"
        assert result[2].commit_hash == "bbb"
        assert result[2].file_path == "c.py"
        assert result[2].author_name == "Bob"
        assert result[2].author_email == "bob@example.com"

    def test_binary_files_skipped(self):
        output = (
            "COMMIT:aaa 2024-06-01T12:00:00+00:00 Alice alice@example.com\n"
            "-\t-\timage.png\n"
            "10\t5\tcode.py"
        )
        result = _parse_numstat(output)
        assert len(result) == 1
        assert result[0].file_path == "code.py"

    def test_empty_output(self):
        result = _parse_numstat("")
        assert result == []

    def test_blank_lines_skipped(self):
        output = (
            "\n"
            "COMMIT:aaa 2024-06-01T12:00:00+00:00 Alice alice@example.com\n"
            "\n"
            "10\t5\ta.py\n"
            "\n"
        )
        result = _parse_numstat(output)
        assert len(result) == 1
        assert result[0].file_path == "a.py"

    def test_malformed_lines_skipped(self):
        output = (
            "COMMIT:aaa 2024-06-01T12:00:00+00:00 Alice alice@example.com\n"
            "not-a-valid-line\n"
            "10\t5\ta.py"
        )
        result = _parse_numstat(output)
        assert len(result) == 1
        assert result[0].file_path == "a.py"

    def test_commit_with_no_files(self):
        output = (
            "COMMIT:aaa 2024-06-01T12:00:00+00:00 Alice alice@example.com\n"
            "\n"
            "COMMIT:bbb 2024-05-01T10:00:00+00:00 Bob bob@example.com\n"
            "5\t3\tb.py"
        )
        result = _parse_numstat(output)
        assert len(result) == 1
        assert result[0].commit_hash == "bbb"

    def test_author_name_with_spaces(self):
        output = "COMMIT:abc123 2024-06-01T12:00:00+00:00 Alice Van Der Berg alice@example.com\n10\t5\ta.py"
        result = _parse_numstat(output)
        assert result[0].author_name == "Alice Van Der Berg"
        assert result[0].author_email == "alice@example.com"
