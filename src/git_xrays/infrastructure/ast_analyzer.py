"""Pure AST analysis functions for detecting anemic domain models."""

from __future__ import annotations

import ast
import os

from git_xrays.domain.models import ClassMetrics, FileAnemia

_LOGIC_NODES = (ast.If, ast.For, ast.While, ast.Try, ast.With)


def _count_fields(node: ast.ClassDef) -> int:
    """Count class-level attributes + self.x assignments in __init__ only."""
    field_names: set[str] = set()

    for child in node.body:
        # Class-level assignments: x = 1
        if isinstance(child, ast.Assign):
            for target in child.targets:
                if isinstance(target, ast.Name):
                    field_names.add(target.id)
        # Annotated assignments: x: int or x: int = 1
        elif isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
            field_names.add(child.target.id)
        # __init__ self.x assignments
        elif (
            isinstance(child, ast.FunctionDef)
            and child.name == "__init__"
        ):
            for stmt in ast.walk(child):
                if (
                    isinstance(stmt, ast.Assign)
                ):
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


def _has_logic(func: ast.FunctionDef) -> bool:
    """Check if function body contains logic statements (if/for/while/try/with)."""
    for child in ast.walk(func):
        if isinstance(child, _LOGIC_NODES):
            return True
    return False


def analyze_class_source(node: ast.ClassDef, file_path: str) -> ClassMetrics:
    """Analyze a single class AST node and return metrics."""
    field_count = _count_fields(node)

    methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
    method_count = len(methods)

    dunder_count = sum(1 for m in methods if _is_dunder(m.name))
    property_count = sum(1 for m in methods if _is_property(m))

    # Behavior: non-dunder, non-property methods with logic
    candidate_methods = [
        m for m in methods
        if not _is_dunder(m.name) and not _is_property(m)
    ]
    behavior_count = sum(1 for m in candidate_methods if _has_logic(m))

    # DBSI
    denom = field_count + behavior_count
    dbsi = round(field_count / denom, 4) if denom > 0 else 0.0

    # Logic density
    n_candidates = len(candidate_methods)
    if n_candidates > 0:
        logic_density = round(behavior_count / n_candidates, 4)
    else:
        logic_density = 0.0

    orchestration_pressure = round(1 - logic_density, 4)
    ams = round(dbsi * orchestration_pressure, 4)

    return ClassMetrics(
        class_name=node.name,
        file_path=file_path,
        field_count=field_count,
        method_count=method_count,
        behavior_method_count=behavior_count,
        dunder_method_count=dunder_count,
        property_count=property_count,
        dbsi=dbsi,
        logic_density=logic_density,
        orchestration_pressure=orchestration_pressure,
        ams=ams,
    )


def analyze_file(
    source: str, file_path: str, ams_threshold: float = 0.5,
) -> FileAnemia:
    """Analyze all top-level classes in a Python source string."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return FileAnemia(
            file_path=file_path,
            class_count=0,
            anemic_class_count=0,
            worst_ams=0.0,
            classes=[],
            touch_count=0,
        )

    classes: list[ClassMetrics] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            classes.append(analyze_class_source(node, file_path))

    classes.sort(key=lambda c: c.ams, reverse=True)

    anemic_count = sum(1 for c in classes if c.ams > ams_threshold)
    worst_ams = classes[0].ams if classes else 0.0

    return FileAnemia(
        file_path=file_path,
        class_count=len(classes),
        anemic_class_count=anemic_count,
        worst_ams=worst_ams,
        classes=classes,
        touch_count=0,
    )


def compute_touch_counts(file_sources: dict[str, str]) -> dict[str, int]:
    """Count how many other files import from each file.

    Uses heuristic module name matching: strips .py extension and matches
    against import/from-import module names.
    """
    # Build module name → file path mapping
    module_to_file: dict[str, str] = {}
    for fp in file_sources:
        module_name = os.path.splitext(os.path.basename(fp))[0]
        module_to_file[module_name] = fp

    counts: dict[str, int] = {fp: 0 for fp in file_sources}

    for importer_path, source in file_sources.items():
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue

        imported_modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_modules.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module.split(".")[0])

        for mod_name in imported_modules:
            target = module_to_file.get(mod_name)
            if target and target != importer_path:
                counts[target] += 1

    return counts
