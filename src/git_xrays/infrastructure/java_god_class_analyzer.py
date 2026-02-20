"""Java god class detection using tree-sitter."""

from __future__ import annotations

import re
from itertools import combinations

import tree_sitter_java as tsjava
from tree_sitter import Language, Node, Parser

from git_xrays.domain.models import FileGodClass, GodClassMetrics

JAVA_LANGUAGE = Language(tsjava.language())

_GETTER_SETTER_RE = re.compile(r"^(get|set|is)[A-Z]")

_DECISION_TYPES = frozenset({
    "if_statement", "for_statement", "enhanced_for_statement",
    "while_statement", "do_statement", "catch_clause",
    "ternary_expression",
})

_CLASS_TYPES = frozenset({"class_declaration", "record_declaration"})


def _count_fields(class_node: Node) -> int:
    """Count field declarations in a class/record body."""
    count = 0
    if class_node.type == "record_declaration":
        params = class_node.child_by_field_name("parameters")
        if params:
            count += sum(1 for c in params.named_children
                         if c.type == "formal_parameter")

    body = class_node.child_by_field_name("body")
    if body is None:
        return count

    for child in body.named_children:
        if child.type == "field_declaration":
            for sub in child.named_children:
                if sub.type == "variable_declarator":
                    count += 1
    return count


def _is_getter_or_setter(method_node: Node) -> bool:
    """Heuristic: method name matches get/set/is pattern and has no logic."""
    name_node = method_node.child_by_field_name("name")
    if not name_node:
        return False
    name = name_node.text.decode("utf-8")
    if not _GETTER_SETTER_RE.match(name):
        return False
    body = method_node.child_by_field_name("body")
    if body and _has_logic(body):
        return False
    return True


def _has_logic(node: Node) -> bool:
    """Check if a node contains logic statements (iterative)."""
    logic_types = frozenset({
        "if_statement", "for_statement", "while_statement",
        "do_statement", "try_statement", "switch_expression",
        "enhanced_for_statement",
    })
    stack = [node]
    while stack:
        n = stack.pop()
        if n.type in logic_types:
            return True
        stack.extend(n.children)
    return False


def _get_candidate_methods(class_node: Node) -> list[Node]:
    """Return non-getter/setter, non-constructor method declarations."""
    body = class_node.child_by_field_name("body")
    if body is None:
        return []
    return [
        m for m in body.named_children
        if m.type == "method_declaration"
        and not _is_getter_or_setter(m)
    ]


def _compute_method_complexity(method_node: Node) -> int:
    """Compute cyclomatic complexity for a single Java method (iterative)."""
    complexity = 1
    body = method_node.child_by_field_name("body")
    if not body:
        return complexity
    stack = [body]
    while stack:
        n = stack.pop()
        if n.type in _DECISION_TYPES:
            complexity += 1
        elif n.type == "switch_expression":
            for child in n.named_children:
                if child.type == "switch_block_statement_group":
                    for sub in child.named_children:
                        if sub.type == "switch_label" and sub.text != b"default:":
                            complexity += 1
        elif n.type == "binary_expression":
            op = n.child_by_field_name("operator")
            if op and op.text in (b"&&", b"||"):
                complexity += 1
        stack.extend(n.children)
    return complexity


def _compute_wmc(class_node: Node) -> int:
    """Weighted Methods per Class: sum of CC for all methods."""
    body = class_node.child_by_field_name("body")
    if body is None:
        return 0
    methods = [m for m in body.named_children
               if m.type in ("method_declaration", "constructor_declaration")]
    return sum(_compute_method_complexity(m) for m in methods)


def _get_field_accesses(method_node: Node) -> set[str]:
    """Collect field names accessed via this.field or direct field references (iterative)."""
    fields: set[str] = set()
    body = method_node.child_by_field_name("body")
    if not body:
        return fields
    stack = [body]
    while stack:
        n = stack.pop()
        if n.type == "field_access":
            obj = n.child_by_field_name("object")
            field = n.child_by_field_name("field")
            if obj and field and obj.text == b"this":
                fields.add(field.text.decode("utf-8"))
        stack.extend(n.children)
    return fields


def _compute_tcc(class_node: Node) -> float:
    """Tight Class Cohesion: fraction of method pairs sharing field access."""
    candidates = _get_candidate_methods(class_node)
    if len(candidates) <= 1:
        return 1.0

    method_fields = [_get_field_accesses(m) for m in candidates]
    total_pairs = len(candidates) * (len(candidates) - 1) // 2
    connected = sum(
        1 for a, b in combinations(range(len(candidates)), 2)
        if method_fields[a] & method_fields[b]
    )
    return round(connected / total_pairs, 4) if total_pairs > 0 else 1.0


def _analyze_java_class(class_node: Node, file_path: str) -> GodClassMetrics:
    """Analyze a single Java class for god class indicators.

    Returns raw metrics with god_class_score=0.0 (normalization happens in use case).
    """
    name_node = class_node.child_by_field_name("name")
    class_name = name_node.text.decode("utf-8") if name_node else "<unknown>"

    field_count = _count_fields(class_node)
    candidate_methods = _get_candidate_methods(class_node)
    method_count = len(candidate_methods)
    total_complexity = _compute_wmc(class_node)
    cohesion = _compute_tcc(class_node)

    return GodClassMetrics(
        class_name=class_name,
        file_path=file_path,
        method_count=method_count,
        field_count=field_count,
        total_complexity=total_complexity,
        cohesion=cohesion,
        god_class_score=0.0,
    )


def analyze_java_god_classes(source: str, file_path: str) -> FileGodClass:
    """Analyze all top-level classes in a Java source file for god class patterns."""
    parser = Parser(JAVA_LANGUAGE)
    tree = parser.parse(source.encode("utf-8"))
    root = tree.root_node

    if root.has_error:
        return FileGodClass(
            file_path=file_path, class_count=0,
            god_class_count=0, worst_gcs=0.0, classes=[],
        )

    classes: list[GodClassMetrics] = []
    for top_node in root.named_children:
        if top_node.type in _CLASS_TYPES:
            classes.append(_analyze_java_class(top_node, file_path))

    return FileGodClass(
        file_path=file_path,
        class_count=len(classes),
        god_class_count=0,
        worst_gcs=0.0,
        classes=classes,
    )
