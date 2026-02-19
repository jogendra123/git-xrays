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


def commit_files(
    repo: Path,
    files: dict[str, str],
    message: str,
    days_ago: int = 0,
    author_name: str = "Test User",
    author_email: str = "test@example.com",
) -> None:
    """Create a single commit touching multiple files at a known relative date."""
    for file_path, content in files.items():
        full_path = repo / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)

    date = datetime.now(timezone.utc) - timedelta(days=days_ago)
    date_str = date.strftime("%Y-%m-%dT%H:%M:%S %z")

    for file_path in files:
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


def create_tag(repo: Path, tag_name: str) -> None:
    """Create a lightweight tag at the current HEAD."""
    subprocess.run(
        ["git", "-C", str(repo), "tag", tag_name],
        capture_output=True, check=True,
    )


@pytest.fixture
def tagged_repo(tmp_git_repo: Path) -> Path:
    """Create a repo with tags for time-travel testing.

    Timeline:
    - 60 days ago: README.md
    - 50 days ago: src/main.py → tag v1.0
    - 30 days ago: update main.py
    - 20 days ago: add utils.py
    - 10 days ago: update main.py → tag v2.0
    """
    commit_file(tmp_git_repo, "README.md", "# Project\n", "Init README", days_ago=60)
    commit_file(tmp_git_repo, "src/main.py", "print('v1')\n", "Add main", days_ago=50)
    create_tag(tmp_git_repo, "v1.0")
    commit_file(tmp_git_repo, "src/main.py", "print('v2')\n", "Update main", days_ago=30)
    commit_file(tmp_git_repo, "src/utils.py", "def helper(): pass\n", "Add utils", days_ago=20)
    commit_file(tmp_git_repo, "src/main.py", "print('v3')\n", "Update main again", days_ago=10)
    create_tag(tmp_git_repo, "v2.0")
    return tmp_git_repo


@pytest.fixture
def anemic_repo(tmp_git_repo: Path) -> Path:
    """Create a repo with a mix of anemic and healthy Python classes.

    - models.py: UserDTO (anemic - data only + getters)
    - services.py: UserService (healthy - logic methods)
    - README.md: non-Python file (should be excluded)
    """
    commit_file(
        tmp_git_repo, "models.py",
        (
            "class UserDTO:\n"
            "    name = ''\n"
            "    email = ''\n"
            "    age = 0\n"
            "    def get_name(self):\n"
            "        return self.name\n"
            "    def get_email(self):\n"
            "        return self.email\n"
        ),
        "Add anemic models",
        days_ago=10,
    )
    commit_file(
        tmp_git_repo, "services.py",
        (
            "import models\n"
            "\n"
            "class UserService:\n"
            "    def validate(self, user):\n"
            "        if not user:\n"
            "            raise ValueError('invalid')\n"
            "    def process(self, items):\n"
            "        for item in items:\n"
            "            print(item)\n"
        ),
        "Add healthy services",
        days_ago=5,
    )
    commit_file(
        tmp_git_repo, "README.md",
        "# Project\n",
        "Add readme",
        days_ago=3,
    )
    return tmp_git_repo


@pytest.fixture
def complexity_repo(tmp_git_repo: Path) -> Path:
    """Create a repo with Python files of varying complexity.

    - simple.py: one simple function (CC=1)
    - complex.py: one complex function with multiple branches (CC>1)
    - README.md: non-Python file (should be excluded)
    """
    commit_file(
        tmp_git_repo, "simple.py",
        (
            "def greet(name):\n"
            "    return f'Hello {name}'\n"
        ),
        "Add simple module",
        days_ago=10,
    )
    commit_file(
        tmp_git_repo, "complex.py",
        (
            "def process(data, mode):\n"
            "    if not data:\n"
            "        return None\n"
            "    for item in data:\n"
            "        if mode == 'strict':\n"
            "            if item.valid:\n"
            "                yield item\n"
            "        elif mode == 'lenient':\n"
            "            yield item\n"
        ),
        "Add complex module",
        days_ago=5,
    )
    commit_file(
        tmp_git_repo, "README.md",
        "# Project\n",
        "Add readme",
        days_ago=3,
    )
    return tmp_git_repo


@pytest.fixture
def clustering_repo(tmp_git_repo: Path) -> Path:
    """Create a repo with diverse commit patterns for clustering analysis.

    - 3 feature-like commits: many files, high additions
    - 3 bugfix-like commits: 1 file, low churn
    - 2 refactoring-like commits: many files, balanced add/delete
    """
    # Feature-like: many files, high additions
    commit_files(tmp_git_repo, {
        "src/feature1.py": "class Feature1:\n    pass\n",
        "src/feature2.py": "class Feature2:\n    pass\n",
        "src/feature3.py": "class Feature3:\n    pass\n",
        "src/feature4.py": "class Feature4:\n    pass\n",
    }, "feat: add feature module", days_ago=40)
    commit_files(tmp_git_repo, {
        "src/feature1.py": "class Feature1:\n    def run(self):\n        pass\n",
        "src/feature2.py": "class Feature2:\n    def run(self):\n        pass\n",
        "src/feature3.py": "class Feature3:\n    def run(self):\n        pass\n",
    }, "feat: add run methods", days_ago=35)
    commit_files(tmp_git_repo, {
        "src/feature1.py": "class Feature1:\n    def run(self):\n        return 42\n",
        "src/feature2.py": "class Feature2:\n    def run(self):\n        return 43\n",
        "src/feature3.py": "class Feature3:\n    def run(self):\n        return 44\n",
        "src/feature4.py": "class Feature4:\n    def run(self):\n        return 45\n",
    }, "feat: implement run", days_ago=30)

    # Bugfix-like: single file, low churn
    commit_file(tmp_git_repo, "src/feature1.py",
                "class Feature1:\n    def run(self):\n        return 42  # fix\n",
                "fix: correct return value", days_ago=25)
    commit_file(tmp_git_repo, "src/feature2.py",
                "class Feature2:\n    def run(self):\n        return 43  # fix\n",
                "fix: correct feature2", days_ago=20)
    commit_file(tmp_git_repo, "src/feature3.py",
                "class Feature3:\n    def run(self):\n        return 44  # fix\n",
                "fix: correct feature3", days_ago=15)

    # Refactoring-like: many files, balanced add/delete
    commit_files(tmp_git_repo, {
        "src/feature1.py": "class Feature1Refactored:\n    def execute(self):\n        return 42\n",
        "src/feature2.py": "class Feature2Refactored:\n    def execute(self):\n        return 43\n",
        "src/feature3.py": "class Feature3Refactored:\n    def execute(self):\n        return 44\n",
    }, "refactor: rename classes and methods", days_ago=10)
    commit_files(tmp_git_repo, {
        "src/feature1.py": "class Feature1Final:\n    def execute(self):\n        result = 42\n        return result\n",
        "src/feature4.py": "class Feature4Final:\n    def execute(self):\n        result = 45\n        return result\n",
    }, "refactor: finalize implementations", days_ago=5)

    return tmp_git_repo


@pytest.fixture
def effort_repo(tmp_git_repo: Path) -> Path:
    """Create a repo with 5 files, ~8 commits, diverse effort patterns.

    - hot.py: high churn, high frequency (3 commits, 2 authors)
    - mod.py: moderate churn (2 commits)
    - cold.py: low churn (1 commit)
    - coupled_a.py + coupled_b.py: always committed together (2 commits)
    """
    commit_file(tmp_git_repo, "hot.py", "print('v1')\n" * 10,
                "Create hot module", days_ago=30,
                author_name="Alice", author_email="alice@example.com")
    commit_file(tmp_git_repo, "hot.py", "print('v2')\n" * 10,
                "Update hot module", days_ago=20,
                author_name="Bob", author_email="bob@example.com")
    commit_file(tmp_git_repo, "hot.py", "print('v3')\n" * 10,
                "Fix hot module", days_ago=10,
                author_name="Alice", author_email="alice@example.com")
    commit_file(tmp_git_repo, "mod.py", "def moderate(): pass\n",
                "Add moderate module", days_ago=25)
    commit_file(tmp_git_repo, "mod.py", "def moderate():\n    return 42\n",
                "Update moderate module", days_ago=15)
    commit_file(tmp_git_repo, "cold.py", "CONST = 1\n",
                "Add cold module", days_ago=40)
    commit_files(tmp_git_repo, {
        "coupled_a.py": "import coupled_b\n",
        "coupled_b.py": "data = []\n",
    }, "Add coupled modules", days_ago=20)
    commit_files(tmp_git_repo, {
        "coupled_a.py": "import coupled_b\nresult = coupled_b.data\n",
        "coupled_b.py": "data = [1, 2, 3]\n",
    }, "Update coupled modules", days_ago=5)
    return tmp_git_repo


@pytest.fixture
def dx_repo(tmp_git_repo: Path) -> Path:
    """Create a repo with diverse patterns for DX analysis.

    - 5 Python files, ~8 commits, diverse patterns
    - Mix of feature/bugfix/refactoring patterns
    - Multiple authors for knowledge signals
    - Python files for complexity analysis
    """
    # Feature-like: many files, high additions
    commit_files(tmp_git_repo, {
        "src/core.py": (
            "def process(data, mode):\n"
            "    if not data:\n"
            "        return None\n"
            "    for item in data:\n"
            "        if mode == 'strict':\n"
            "            if item:\n"
            "                yield item\n"
        ),
        "src/utils.py": "def helper(x):\n    return x + 1\n",
        "src/config.py": "DEBUG = True\n",
    }, "feat: initial implementation", days_ago=40,
        author_name="Alice", author_email="alice@example.com")

    # Feature: add more code
    commit_files(tmp_git_repo, {
        "src/core.py": (
            "def process(data, mode):\n"
            "    if not data:\n"
            "        return None\n"
            "    for item in data:\n"
            "        if mode == 'strict':\n"
            "            if item:\n"
            "                yield item\n"
            "        elif mode == 'lenient':\n"
            "            yield item\n"
        ),
        "src/models.py": "class User:\n    name = ''\n    email = ''\n",
    }, "feat: add lenient mode and models", days_ago=30,
        author_name="Alice", author_email="alice@example.com")

    # Bugfix-like: single file, small change
    commit_file(tmp_git_repo, "src/core.py",
        (
            "def process(data, mode):\n"
            "    if not data:\n"
            "        return []\n"
            "    for item in data:\n"
            "        if mode == 'strict':\n"
            "            if item:\n"
            "                yield item\n"
            "        elif mode == 'lenient':\n"
            "            yield item\n"
        ),
        "fix: return empty list instead of None", days_ago=20,
        author_name="Bob", author_email="bob@example.com")

    # Another feature
    commit_file(tmp_git_repo, "src/api.py",
        "def handle_request(req):\n    return {'status': 'ok'}\n",
        "feat: add API handler", days_ago=15,
        author_name="Alice", author_email="alice@example.com")

    # Refactoring: touch multiple files
    commit_files(tmp_git_repo, {
        "src/core.py": (
            "def process(data, mode='strict'):\n"
            "    if not data:\n"
            "        return []\n"
            "    result = []\n"
            "    for item in data:\n"
            "        if mode == 'strict' and item:\n"
            "            result.append(item)\n"
            "        elif mode == 'lenient':\n"
            "            result.append(item)\n"
            "    return result\n"
        ),
        "src/utils.py": "def helper(x):\n    return x + 1\n\ndef validate(x):\n    if x is None:\n        raise ValueError('invalid')\n",
    }, "refactor: simplify core and add validation", days_ago=10,
        author_name="Bob", author_email="bob@example.com")

    # Config update
    commit_file(tmp_git_repo, "src/config.py",
        "DEBUG = False\nVERBOSE = True\n",
        "config: disable debug", days_ago=5,
        author_name="Alice", author_email="alice@example.com")

    # Another bugfix
    commit_file(tmp_git_repo, "src/api.py",
        "def handle_request(req):\n    if not req:\n        return {'status': 'error'}\n    return {'status': 'ok'}\n",
        "fix: handle empty request", days_ago=3,
        author_name="Bob", author_email="bob@example.com")

    # Late feature
    commit_file(tmp_git_repo, "src/models.py",
        "class User:\n    name = ''\n    email = ''\n    def __init__(self, name, email):\n        self.name = name\n        self.email = email\n",
        "feat: add User constructor", days_ago=1,
        author_name="Alice", author_email="alice@example.com")

    return tmp_git_repo


@pytest.fixture
def coupled_repo(tmp_git_repo: Path) -> Path:
    """Create a repo for coupling analysis.

    a+b always together (3 commits), a+c sometimes (2 commits),
    c alone once, d alone once. Total: 5 unique commits.
    """
    # Commit 1: a+b+c together
    commit_files(tmp_git_repo, {
        "a.py": "a_v1\n",
        "b.py": "b_v1\n",
        "c.py": "c_v1\n",
    }, "commit 1: a+b+c", days_ago=25)

    # Commit 2: a+b+c together
    commit_files(tmp_git_repo, {
        "a.py": "a_v2\n",
        "b.py": "b_v2\n",
        "c.py": "c_v2\n",
    }, "commit 2: a+b+c", days_ago=20)

    # Commit 3: a+b together
    commit_files(tmp_git_repo, {
        "a.py": "a_v3\n",
        "b.py": "b_v3\n",
    }, "commit 3: a+b", days_ago=15)

    # Commit 4: c alone
    commit_files(tmp_git_repo, {
        "c.py": "c_v3\n",
    }, "commit 4: c alone", days_ago=10)

    # Commit 5: d alone
    commit_files(tmp_git_repo, {
        "d.py": "d_v1\n",
    }, "commit 5: d alone", days_ago=5)

    return tmp_git_repo


@pytest.fixture
def java_repo(tmp_git_repo: Path) -> Path:
    """Create a repo with Java files for complexity and anemia testing.

    - UserDTO.java: anemic DTO (fields + getters, no logic)
    - UserService.java: healthy service (logic methods importing UserDTO)
    """
    commit_file(
        tmp_git_repo, "UserDTO.java",
        (
            "public class UserDTO {\n"
            "    private String name;\n"
            "    private String email;\n"
            "    private int age;\n"
            "    public String getName() { return this.name; }\n"
            "    public String getEmail() { return this.email; }\n"
            "    public int getAge() { return this.age; }\n"
            "    public void setName(String n) { this.name = n; }\n"
            "}\n"
        ),
        "Add anemic Java DTO",
        days_ago=10,
    )
    commit_file(
        tmp_git_repo, "UserService.java",
        (
            "import com.example.UserDTO;\n"
            "\n"
            "public class UserService {\n"
            "    public void validate(Object user) {\n"
            "        if (user == null) {\n"
            "            throw new IllegalArgumentException();\n"
            "        }\n"
            "    }\n"
            "    public void processAll(int[] items) {\n"
            "        for (int item : items) {\n"
            "            if (item > 0) {\n"
            "                System.out.println(item);\n"
            "            }\n"
            "        }\n"
            "    }\n"
            "}\n"
        ),
        "Add healthy Java service",
        days_ago=5,
    )
    return tmp_git_repo


@pytest.fixture
def mixed_lang_repo(tmp_git_repo: Path) -> Path:
    """Create a repo with both Python and Java files.

    - models.py: Python anemic DTO
    - services.py: Python service
    - UserDTO.java: Java anemic DTO
    - UserService.java: Java service
    """
    commit_file(
        tmp_git_repo, "models.py",
        (
            "class UserDTO:\n"
            "    name = ''\n"
            "    email = ''\n"
            "    def get_name(self):\n"
            "        return self.name\n"
        ),
        "Add Python DTO",
        days_ago=15,
    )
    commit_file(
        tmp_git_repo, "services.py",
        (
            "class UserService:\n"
            "    def validate(self, user):\n"
            "        if not user:\n"
            "            raise ValueError('invalid')\n"
        ),
        "Add Python service",
        days_ago=12,
    )
    commit_file(
        tmp_git_repo, "UserDTO.java",
        (
            "public class UserDTO {\n"
            "    private String name;\n"
            "    private int age;\n"
            "    public String getName() { return this.name; }\n"
            "    public int getAge() { return this.age; }\n"
            "}\n"
        ),
        "Add Java DTO",
        days_ago=10,
    )
    commit_file(
        tmp_git_repo, "UserService.java",
        (
            "public class UserService {\n"
            "    public void validate(Object user) {\n"
            "        if (user == null) {\n"
            "            throw new IllegalArgumentException();\n"
            "        }\n"
            "    }\n"
            "}\n"
        ),
        "Add Java service",
        days_ago=5,
    )
    return tmp_git_repo


@pytest.fixture
def god_class_repo(tmp_git_repo: Path) -> Path:
    """Create a repo with a small clean class + a large god class."""
    commit_file(
        tmp_git_repo, "small.py",
        (
            "class SmallHelper:\n"
            "    def __init__(self):\n"
            "        self.x = 1\n"
            "    def get_x(self):\n"
            "        return self.x\n"
        ),
        "Add small helper",
        days_ago=10,
    )
    commit_file(
        tmp_git_repo, "god.py",
        (
            "class GodClass:\n"
            "    def __init__(self):\n"
            "        self.a = 1\n"
            "        self.b = 2\n"
            "        self.c = 3\n"
            "        self.d = 4\n"
            "        self.e = 5\n"
            "    def m1(self):\n"
            "        if self.a > 0:\n"
            "            for i in range(self.b):\n"
            "                if i > self.c:\n"
            "                    pass\n"
            "    def m2(self):\n"
            "        while self.d:\n"
            "            pass\n"
            "    def m3(self):\n"
            "        return self.e\n"
            "    def m4(self):\n"
            "        if self.a and self.b:\n"
            "            return True\n"
            "        return False\n"
            "    def m5(self):\n"
            "        try:\n"
            "            pass\n"
            "        except Exception:\n"
            "            pass\n"
            "    def m6(self):\n"
            "        for x in range(self.c):\n"
            "            if x > self.d:\n"
            "                while self.e:\n"
            "                    break\n"
        ),
        "Add god class",
        days_ago=5,
    )
    return tmp_git_repo
