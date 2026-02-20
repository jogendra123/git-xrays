"""Java anemic domain model detection using tree-sitter."""

from __future__ import annotations

import os
import re

import tree_sitter_java as tsjava
from tree_sitter import Language, Parser, Node

from git_xrays.domain.models import ClassMetrics, FileAnemic

JAVA_LANGUAGE = Language(tsjava.language())

_LOGIC_TYPES = frozenset({
    "if_statement", "for_statement", "while_statement",
    "do_statement", "try_statement", "switch_expression",
    "enhanced_for_statement",
})

_GETTER_SETTER_RE = re.compile(r"^(get|set|is)[A-Z]")


def _count_fields(class_node: Node) -> int:
    """Count field declarations in a class/record body."""
    count = 0

    # Record parameters count as fields
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
            # Count declarators (e.g., int a, b; → 2 fields)
            for sub in child.named_children:
                if sub.type == "variable_declarator":
                    count += 1
    return count


def _has_logic(method_body: Node) -> bool:
    """Check if a method body contains logic statements (iterative)."""
    stack = [method_body]
    while stack:
        n = stack.pop()
        if n.type in _LOGIC_TYPES:
            return True
        stack.extend(n.children)
    return False


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


def _extract_class_name(node: Node) -> str:
    """Get the class name from a class/record declaration."""
    name_node = node.child_by_field_name("name")
    if name_node:
        return name_node.text.decode("utf-8")
    return "<unknown>"


def _analyze_java_class(class_node: Node, file_path: str,
                        ams_threshold: float = 0.5) -> ClassMetrics:
    """Analyze a single Java class/record node."""
    class_name = _extract_class_name(class_node)
    field_count = _count_fields(class_node)

    body = class_node.child_by_field_name("body")
    methods: list[Node] = []
    if body:
        methods = [c for c in body.named_children
                   if c.type in ("method_declaration", "constructor_declaration")]

    method_count = len(methods)

    # For Java, dunder_method_count is 0 (Java doesn't have dunders)
    dunder_method_count = 0

    # Count getters/setters (mapped to property_count)
    getter_setter_count = sum(1 for m in methods if _is_getter_or_setter(m))

    # Candidate methods: all non-getter/setter, non-constructor
    candidate_methods = [
        m for m in methods
        if m.type == "method_declaration" and not _is_getter_or_setter(m)
    ]

    # Behavior: candidates with logic
    behavior_count = 0
    for m in candidate_methods:
        body_node = m.child_by_field_name("body")
        if body_node and _has_logic(body_node):
            behavior_count += 1

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
        class_name=class_name,
        file_path=file_path,
        field_count=field_count,
        method_count=method_count,
        behavior_method_count=behavior_count,
        dunder_method_count=dunder_method_count,
        property_count=getter_setter_count,
        dbsi=dbsi,
        logic_density=logic_density,
        orchestration_pressure=orchestration_pressure,
        ams=ams,
    )


def analyze_java_file_anemic(
    source: str, file_path: str, ams_threshold: float = 0.5,
) -> FileAnemic:
    """Analyze all top-level classes in a Java source file."""
    parser = Parser(JAVA_LANGUAGE)
    tree = parser.parse(source.encode("utf-8"))
    root = tree.root_node

    if root.has_error:
        return FileAnemic(
            file_path=file_path, class_count=0, anemic_class_count=0,
            worst_ams=0.0, classes=[], touch_count=0,
        )

    _CLASS_TYPES = ("class_declaration", "record_declaration")
    classes: list[ClassMetrics] = []

    for top_node in root.named_children:
        if top_node.type in _CLASS_TYPES:
            classes.append(
                _analyze_java_class(top_node, file_path, ams_threshold)
            )

    classes.sort(key=lambda c: c.ams, reverse=True)

    anemic_count = sum(1 for c in classes if c.ams > ams_threshold)
    worst_ams = classes[0].ams if classes else 0.0

    return FileAnemic(
        file_path=file_path,
        class_count=len(classes),
        anemic_class_count=anemic_count,
        worst_ams=worst_ams,
        classes=classes,
        touch_count=0,
    )


def compute_java_touch_counts(file_sources: dict[str, str]) -> dict[str, int]:
    """Count how many other Java files import from each file.

    Uses heuristic: extracts class name from filename, matches against
    import statements in other files.
    """
    # Build simple class name → file path mapping
    class_to_file: dict[str, str] = {}
    for fp in file_sources:
        class_name = os.path.splitext(os.path.basename(fp))[0]
        class_to_file[class_name] = fp

    counts: dict[str, int] = {fp: 0 for fp in file_sources}

    parser = Parser(JAVA_LANGUAGE)

    for importer_path, source in file_sources.items():
        tree = parser.parse(source.encode("utf-8"))
        root = tree.root_node

        imported_classes: set[str] = set()
        for node in root.named_children:
            if node.type == "import_declaration":
                # Extract the imported class name (last segment)
                text = node.text.decode("utf-8")
                # Remove "import " prefix and ";" suffix
                import_path = text.replace("import ", "").replace(";", "").strip()
                # Remove "static " if present
                import_path = import_path.replace("static ", "")
                # Get last segment (class name)
                parts = import_path.split(".")
                if parts:
                    imported_classes.add(parts[-1])

        for cls_name in imported_classes:
            target = class_to_file.get(cls_name)
            if target and target != importer_path:
                counts[target] += 1

    return counts
