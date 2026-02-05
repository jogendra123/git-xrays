# git-xrays

Behavioral & Architectural Code Intelligence Platform.

Analyzes Git repositories to measure behavioral, architectural, and socio-technical signals.

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- Git

## Setup

```bash
uv sync
```

## Usage

```bash
uv run analyze-repo /path/to/git/repo
```

Example output:

```
Repository:   /path/to/git/repo
Commits:      142
First commit: 2023-01-15 10:30:00 +0530
Last commit:  2025-12-01 14:22:00 +0530
```

## Project Structure

```
src/git_xrays/
├── domain/          # Models, ports (no external dependencies)
├── application/     # Use cases
├── infrastructure/  # Git CLI reader, storage (later)
└── interface/       # CLI entry point
```
