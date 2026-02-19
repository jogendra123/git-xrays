"""Java function-level complexity analysis using tree-sitter."""

from __future__ import annotations

import tree_sitter_java as tsjava
from tree_sitter import Language, Parser, Node

from git_xrays.domain.models import FileComplexity, FunctionComplexity

JAVA_LANGUAGE = Language(tsjava.language())

_NESTING_TYPES = frozenset({
    "if_statement", "for_statement", "while_statement",
    "do_statement", "try_statement", "switch_expression",
    "enhanced_for_statement",
})

_DECISION_TYPES = frozenset({
    "if_statement", "for_statement", "while_statement",
    "do_statement", "catch_clause", "ternary_expression",
    "enhanced_for_statement",
})


def _compute_cyclomatic(node: Node) -> int:
    """Compute cyclomatic complexity: 1 + decision points."""
    complexity = 1

    def _walk(n: Node) -> None:
        nonlocal complexity
        if n.type in _DECISION_TYPES:
            complexity += 1
        elif n.type == "switch_expression":
            # Count case labels (each non-default case is a decision)
            for child in n.named_children:
                if child.type == "switch_block_statement_group":
                    for label in child.named_children:
                        if label.type == "switch_label" and label.text != b"default":
                            complexity += 1
                elif child.type == "switch_rule":
                    for label in child.named_children:
                        if label.type == "switch_label" and label.text != b"default":
                            complexity += 1
        elif n.type == "binary_expression":
            op = n.child_by_field_name("operator")
            if op and op.type in ("&&", "||"):
                complexity += 1
        for child in n.children:
            _walk(child)

    for child in node.children:
        _walk(child)
    return complexity


def _compute_max_nesting(node: Node) -> int:
    """Compute maximum nesting depth of control flow structures."""

    def _walk(n: Node, depth: int) -> int:
        max_depth = depth
        for child in n.named_children:
            if child.type in _NESTING_TYPES:
                child_depth = _walk(child, depth + 1)
                if child_depth > max_depth:
                    max_depth = child_depth
            else:
                child_depth = _walk(child, depth)
                if child_depth > max_depth:
                    max_depth = child_depth
        return max_depth

    return _walk(node, 0)


def _count_branches(node: Node) -> int:
    """Count if_statement nodes."""
    count = 0

    def _walk(n: Node) -> None:
        nonlocal count
        if n.type == "if_statement":
            count += 1
        for child in n.children:
            _walk(child)

    for child in node.children:
        _walk(child)
    return count


def _count_exception_paths(node: Node) -> int:
    """Count catch_clause nodes."""
    count = 0

    def _walk(n: Node) -> None:
        nonlocal count
        if n.type == "catch_clause":
            count += 1
        for child in n.children:
            _walk(child)

    for child in node.children:
        _walk(child)
    return count


def _compute_cognitive_complexity(node: Node) -> int:
    """Compute cognitive complexity per the SonarSource algorithm adapted for Java."""
    total = 0

    def _walk(n: Node, depth: int) -> None:
        nonlocal total
        for child in n.named_children:
            if child.type == "if_statement":
                total += 1 + depth
                _count_bool_ops_in_condition(child)
                # Walk the consequence (body)
                body = child.child_by_field_name("consequence")
                if body:
                    _walk(body, depth + 1)
                # Handle else/else-if
                alt = child.child_by_field_name("alternative")
                if alt:
                    if alt.type == "if_statement":
                        # else if: adds 1 (no nesting increment)
                        total += 1
                        _count_bool_ops_in_condition(alt)
                        alt_body = alt.child_by_field_name("consequence")
                        if alt_body:
                            _walk(alt_body, depth + 1)
                        alt_alt = alt.child_by_field_name("alternative")
                        if alt_alt:
                            _handle_else_chain(alt_alt, depth)
                    else:
                        # else block
                        total += 1
                        _walk(alt, depth + 1)
            elif child.type in ("for_statement", "while_statement",
                                "do_statement", "enhanced_for_statement"):
                total += 1 + depth
                if child.type == "while_statement":
                    _count_bool_ops_in_condition(child)
                _walk(child, depth + 1)
            elif child.type == "try_statement":
                # try body at same depth
                for tc in child.named_children:
                    if tc.type == "block":
                        _walk(tc, depth)
                    elif tc.type == "catch_clause":
                        total += 1 + depth
                        _walk(tc, depth + 1)
                    elif tc.type == "finally_clause":
                        _walk(tc, depth)
            elif child.type == "switch_expression":
                total += 1 + depth
                _walk(child, depth + 1)
            else:
                _walk(child, depth)

    def _handle_else_chain(alt: Node, depth: int) -> None:
        nonlocal total
        if alt.type == "if_statement":
            total += 1  # else if
            _count_bool_ops_in_condition(alt)
            body = alt.child_by_field_name("consequence")
            if body:
                _walk(body, depth + 1)
            next_alt = alt.child_by_field_name("alternative")
            if next_alt:
                _handle_else_chain(next_alt, depth)
        else:
            total += 1  # else
            _walk(alt, depth + 1)

    def _count_bool_ops_in_condition(stmt: Node) -> None:
        """Count boolean operator sequences in condition."""
        nonlocal total
        cond = stmt.child_by_field_name("condition")
        if cond:
            total += _count_bool_sequences(cond)

    _walk(node, 0)
    return total


def _count_bool_sequences(node: Node) -> int:
    """Count boolean operator sequences. Each chain of && or || adds 1."""
    count = 0

    def _walk(n: Node) -> None:
        nonlocal count
        if n.type == "binary_expression":
            op = n.child_by_field_name("operator")
            if op and op.type in ("&&", "||"):
                count += 1
        for child in n.children:
            _walk(child)

    _walk(node)
    return count


def _get_method_line(node: Node) -> int:
    """Get 1-based line number of a method node."""
    return node.start_point[0] + 1


def _get_method_length(node: Node) -> int:
    """Compute length in lines of a method node."""
    return node.end_point[0] - node.start_point[0] + 1


def _extract_class_name(node: Node) -> str | None:
    """Get the class name from a class/record/enum/interface declaration."""
    name_node = node.child_by_field_name("name")
    if name_node:
        return name_node.text.decode("utf-8")
    return None


def analyze_java_file_complexity(source: str, file_path: str) -> FileComplexity:
    """Analyze all methods in a Java source file."""
    parser = Parser(JAVA_LANGUAGE)
    tree = parser.parse(source.encode("utf-8"))
    root = tree.root_node

    if root.has_error:
        return FileComplexity(
            file_path=file_path, function_count=0, total_complexity=0,
            avg_complexity=0.0, max_complexity=0, worst_function="",
            avg_length=0.0, max_length=0, avg_nesting=0.0, max_nesting=0,
            avg_cognitive=0.0, max_cognitive=0,
            functions=[],
        )

    functions: list[FunctionComplexity] = []
    _CLASS_TYPES = ("class_declaration", "record_declaration",
                    "enum_declaration", "interface_declaration")

    for top_node in root.named_children:
        if top_node.type in _CLASS_TYPES:
            class_name = _extract_class_name(top_node)
            body = top_node.child_by_field_name("body")
            if body:
                for member in body.named_children:
                    if member.type in ("method_declaration", "constructor_declaration"):
                        name_node = member.child_by_field_name("name")
                        method_name = name_node.text.decode("utf-8") if name_node else "<constructor>"
                        if member.type == "constructor_declaration" and not name_node:
                            method_name = class_name or "<constructor>"

                        method_body = member.child_by_field_name("body")
                        if method_body is None:
                            # Abstract method, no body
                            functions.append(FunctionComplexity(
                                function_name=method_name,
                                file_path=file_path,
                                class_name=class_name,
                                line_number=_get_method_line(member),
                                length=_get_method_length(member),
                                cyclomatic_complexity=1,
                                cognitive_complexity=0,
                                max_nesting_depth=0,
                                branch_count=0,
                                exception_paths=0,
                            ))
                            continue

                        functions.append(FunctionComplexity(
                            function_name=method_name,
                            file_path=file_path,
                            class_name=class_name,
                            line_number=_get_method_line(member),
                            length=_get_method_length(member),
                            cyclomatic_complexity=_compute_cyclomatic(method_body),
                            cognitive_complexity=_compute_cognitive_complexity(method_body),
                            max_nesting_depth=_compute_max_nesting(method_body),
                            branch_count=_count_branches(method_body),
                            exception_paths=_count_exception_paths(method_body),
                        ))

    functions.sort(key=lambda f: f.cyclomatic_complexity, reverse=True)

    if not functions:
        return FileComplexity(
            file_path=file_path, function_count=0, total_complexity=0,
            avg_complexity=0.0, max_complexity=0, worst_function="",
            avg_length=0.0, max_length=0, avg_nesting=0.0, max_nesting=0,
            avg_cognitive=0.0, max_cognitive=0,
            functions=[],
        )

    total_cc = sum(f.cyclomatic_complexity for f in functions)
    max_cc = max(f.cyclomatic_complexity for f in functions)
    avg_cc = round(total_cc / len(functions), 2)
    avg_len = round(sum(f.length for f in functions) / len(functions), 2)
    max_len = max(f.length for f in functions)
    avg_nest = round(sum(f.max_nesting_depth for f in functions) / len(functions), 2)
    max_nest = max(f.max_nesting_depth for f in functions)
    all_cog = [f.cognitive_complexity for f in functions]
    avg_cog = round(sum(all_cog) / len(all_cog), 2)
    max_cog = max(all_cog)
    worst = functions[0].function_name

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
        avg_cognitive=avg_cog,
        max_cognitive=max_cog,
        functions=functions,
    )
