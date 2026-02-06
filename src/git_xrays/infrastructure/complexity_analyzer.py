"""Function-level cyclomatic complexity analysis."""

from __future__ import annotations

import ast

from git_xrays.domain.models import FileComplexity, FunctionComplexity


def _compute_cyclomatic(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """Compute cyclomatic complexity: 1 + decision points."""
    complexity = 1
    for child in ast.walk(node):
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
            # `a and b and c` = BoolOp(values=[a,b,c]) → adds len(values)-1
            complexity += len(child.values) - 1
    return complexity


def _compute_max_nesting(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """Compute maximum nesting depth of control flow structures."""
    nesting_types = (ast.If, ast.For, ast.While, ast.With, ast.Try)

    def _walk_depth(body: list[ast.stmt], depth: int) -> int:
        max_depth = depth
        for stmt in body:
            if isinstance(stmt, nesting_types):
                child_depth = depth + 1
                if child_depth > max_depth:
                    max_depth = child_depth
                # Recurse into sub-bodies
                for attr in ("body", "orelse", "handlers", "finalbody"):
                    sub = getattr(stmt, attr, None)
                    if sub:
                        if attr == "handlers":
                            for handler in sub:
                                d = _walk_depth(handler.body, child_depth)
                                if d > max_depth:
                                    max_depth = d
                        else:
                            d = _walk_depth(sub, child_depth)
                            if d > max_depth:
                                max_depth = d
        return max_depth

    return _walk_depth(node.body, 0)


def _count_branches(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """Count number of ast.If nodes (includes elif)."""
    count = 0
    for child in ast.walk(node):
        if isinstance(child, ast.If):
            count += 1
    return count


def _count_exception_paths(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """Count number of ast.ExceptHandler nodes."""
    count = 0
    for child in ast.walk(node):
        if isinstance(child, ast.ExceptHandler):
            count += 1
    return count


def _analyze_function_node(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    file_path: str,
    class_name: str | None = None,
) -> FunctionComplexity:
    """Analyze a single function/method AST node."""
    length = (node.end_lineno or node.lineno) - node.lineno + 1
    return FunctionComplexity(
        function_name=node.name,
        file_path=file_path,
        class_name=class_name,
        line_number=node.lineno,
        length=length,
        cyclomatic_complexity=_compute_cyclomatic(node),
        max_nesting_depth=_compute_max_nesting(node),
        branch_count=_count_branches(node),
        exception_paths=_count_exception_paths(node),
    )


def analyze_file_complexity(source: str, file_path: str) -> FileComplexity:
    """Analyze all functions in a Python source file."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return FileComplexity(
            file_path=file_path, function_count=0, total_complexity=0,
            avg_complexity=0.0, max_complexity=0, worst_function="",
            avg_length=0.0, max_length=0, avg_nesting=0.0, max_nesting=0,
            functions=[],
        )

    functions: list[FunctionComplexity] = []

    # Top-level functions
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(_analyze_function_node(node, file_path))
        elif isinstance(node, ast.ClassDef):
            # Class methods (direct children only, skip nested)
            for item in ast.iter_child_nodes(node):
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    functions.append(
                        _analyze_function_node(item, file_path, class_name=node.name)
                    )

    functions.sort(key=lambda f: f.cyclomatic_complexity, reverse=True)

    if not functions:
        return FileComplexity(
            file_path=file_path, function_count=0, total_complexity=0,
            avg_complexity=0.0, max_complexity=0, worst_function="",
            avg_length=0.0, max_length=0, avg_nesting=0.0, max_nesting=0,
            functions=[],
        )

    total_cc = sum(f.cyclomatic_complexity for f in functions)
    max_cc = max(f.cyclomatic_complexity for f in functions)
    avg_cc = round(total_cc / len(functions), 2)
    avg_len = round(sum(f.length for f in functions) / len(functions), 2)
    max_len = max(f.length for f in functions)
    avg_nest = round(sum(f.max_nesting_depth for f in functions) / len(functions), 2)
    max_nest = max(f.max_nesting_depth for f in functions)
    worst = functions[0].function_name  # already sorted desc

    return FileComplexity(
        file_path=file_path,
        function_count=len(functions),
        total_complexity=total_cc,
        avg_complexity=avg_cc,
        max_complexity=max_cc,
        worst_function=worst,
        avg_length=avg_len,
        max_length=max_len,
        avg_nesting=avg_nest,
        max_nesting=max_nest,
        functions=functions,
    )
