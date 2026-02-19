"""Python AST-based god class detection."""

from __future__ import annotations

import ast
from itertools import combinations

from git_xrays.domain.models import FileGodClass, GodClassMetrics

_LOGIC_NODES = (ast.If, ast.For, ast.While, ast.Try, ast.With)


def _count_fields(node: ast.ClassDef) -> int:
    """Count class-level attributes + self.x assignments in __init__ only."""
    field_names: set[str] = set()

    for child in node.body:
        if isinstance(child, ast.Assign):
            for target in child.targets:
                if isinstance(target, ast.Name):
                    field_names.add(target.id)
        elif isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
            field_names.add(child.target.id)
        elif isinstance(child, ast.FunctionDef) and child.name == "__init__":
            for stmt in ast.walk(child):
                if isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        if (
                            isinstance(target, ast.Attribute)
                            and isinstance(target.value, ast.Name)
                            and target.value.id == "self"
                        ):
                            field_names.add(target.attr)

    return len(field_names)


def _is_dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


def _is_property(func: ast.FunctionDef) -> bool:
    return any(
        isinstance(d, ast.Name) and d.id == "property"
        for d in func.decorator_list
    )


def _get_candidate_methods(node: ast.ClassDef) -> list[ast.FunctionDef]:
    """Return non-dunder, non-property methods."""
    return [
        m for m in node.body
        if isinstance(m, ast.FunctionDef)
        and not _is_dunder(m.name)
        and not _is_property(m)
    ]


def _compute_method_complexity(func: ast.FunctionDef) -> int:
    """Compute cyclomatic complexity for a single method."""
    complexity = 1
    for child in ast.walk(func):
        if isinstance(child, (ast.If, ast.IfExp)):
            complexity += 1
        elif isinstance(child, ast.For):
            complexity += 1
        elif isinstance(child, ast.While):
            complexity += 1
        elif isinstance(child, ast.ExceptHandler):
            complexity += 1
        elif isinstance(child, ast.Assert):
            complexity += 1
        elif isinstance(child, ast.BoolOp):
            complexity += len(child.values) - 1
    return complexity


def _compute_wmc(node: ast.ClassDef) -> int:
    """Weighted Methods per Class: sum of cyclomatic complexity of all methods."""
    methods = [m for m in node.body if isinstance(m, ast.FunctionDef)]
    return sum(_compute_method_complexity(m) for m in methods)


def _get_field_accesses(func: ast.FunctionDef) -> set[str]:
    """Collect set of self.x field accesses in a method."""
    fields: set[str] = set()
    for child in ast.walk(func):
        if (
            isinstance(child, ast.Attribute)
            and isinstance(child.value, ast.Name)
            and child.value.id == "self"
        ):
            fields.add(child.attr)
    return fields


def _compute_tcc(node: ast.ClassDef) -> float:
    """Tight Class Cohesion: fraction of method pairs sharing field access.

    TCC = connected_pairs / total_possible_pairs.
    Returns 1.0 if <= 1 candidate method (maximally cohesive by convention).
    """
    candidates = _get_candidate_methods(node)
    if len(candidates) <= 1:
        return 1.0

    method_fields = [_get_field_accesses(m) for m in candidates]
    total_pairs = len(candidates) * (len(candidates) - 1) // 2
    connected = sum(
        1 for a, b in combinations(range(len(candidates)), 2)
        if method_fields[a] & method_fields[b]
    )
    return round(connected / total_pairs, 4) if total_pairs > 0 else 1.0


def _analyze_python_class(node: ast.ClassDef, file_path: str) -> GodClassMetrics:
    """Analyze a single Python class for god class indicators.

    Returns raw metrics with god_class_score=0.0 (normalization happens in use case).
    """
    field_count = _count_fields(node)
    candidate_methods = _get_candidate_methods(node)
    method_count = len(candidate_methods)
    total_complexity = _compute_wmc(node)
    cohesion = _compute_tcc(node)

    return GodClassMetrics(
        class_name=node.name,
        file_path=file_path,
        method_count=method_count,
        field_count=field_count,
        total_complexity=total_complexity,
        cohesion=cohesion,
        god_class_score=0.0,
    )


def analyze_python_god_classes(source: str, file_path: str) -> FileGodClass:
    """Analyze all top-level classes in a Python source file for god class patterns."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return FileGodClass(
            file_path=file_path, class_count=0,
            god_class_count=0, worst_gcs=0.0, classes=[],
        )

    classes: list[GodClassMetrics] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            classes.append(_analyze_python_class(node, file_path))

    return FileGodClass(
        file_path=file_path,
        class_count=len(classes),
        god_class_count=0,
        worst_gcs=0.0,
        classes=classes,
    )
