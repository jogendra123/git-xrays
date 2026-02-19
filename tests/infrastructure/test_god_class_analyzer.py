"""Tests for Python AST-based god class detection."""

import pytest

from git_xrays.infrastructure.god_class_analyzer import (
    _compute_method_complexity,
    _compute_tcc,
    _compute_wmc,
    _count_fields,
    _get_candidate_methods,
    analyze_python_god_classes,
)
import ast


def _parse_class(source: str) -> ast.ClassDef:
    """Helper to parse a source string and return the first class."""
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            return node
    raise ValueError("No class found")


# ── WMC Tests ─────────────────────────────────────────────────────────


class TestWMC:
    def test_empty_class(self):
        src = "class Empty:\n    pass\n"
        node = _parse_class(src)
        assert _compute_wmc(node) == 0

    def test_single_simple_method(self):
        src = """
class Foo:
    def bar(self):
        return 1
"""
        node = _parse_class(src)
        assert _compute_wmc(node) == 1  # base CC of 1

    def test_method_with_branches(self):
        src = """
class Foo:
    def bar(self, x):
        if x > 0:
            return x
        elif x < 0:
            return -x
        return 0
"""
        node = _parse_class(src)
        # CC = 1 + if + elif(nested If) = 3
        assert _compute_wmc(node) == 3

    def test_multiple_methods(self):
        src = """
class Foo:
    def a(self):
        return 1
    def b(self, x):
        if x:
            return x
        return 0
"""
        node = _parse_class(src)
        # a: CC=1, b: CC=2 → WMC=3
        assert _compute_wmc(node) == 3

    def test_includes_dunder_methods(self):
        src = """
class Foo:
    def __init__(self):
        if True:
            pass
    def bar(self):
        return 1
"""
        node = _parse_class(src)
        # __init__: CC=2, bar: CC=1 → WMC=3
        assert _compute_wmc(node) == 3


# ── TCC Tests ─────────────────────────────────────────────────────────


class TestTCC:
    def test_single_method_returns_one(self):
        src = """
class Foo:
    def bar(self):
        return self.x
"""
        node = _parse_class(src)
        assert _compute_tcc(node) == 1.0

    def test_no_candidate_methods_returns_one(self):
        src = """
class Foo:
    def __init__(self):
        self.x = 1
"""
        node = _parse_class(src)
        assert _compute_tcc(node) == 1.0

    def test_disjoint_methods_return_zero(self):
        src = """
class Foo:
    def a(self):
        return self.x
    def b(self):
        return self.y
"""
        node = _parse_class(src)
        assert _compute_tcc(node) == 0.0

    def test_all_methods_share_field(self):
        src = """
class Foo:
    def a(self):
        return self.x
    def b(self):
        return self.x + 1
    def c(self):
        return self.x * 2
"""
        node = _parse_class(src)
        # 3 methods, all share self.x → 3/3 = 1.0
        assert _compute_tcc(node) == 1.0

    def test_partial_cohesion(self):
        src = """
class Foo:
    def a(self):
        return self.x
    def b(self):
        return self.x + self.y
    def c(self):
        return self.z
"""
        node = _parse_class(src)
        # Pairs: (a,b)=share x ✓, (a,c)=no ✗, (b,c)=no ✗
        # TCC = 1/3 ≈ 0.3333
        assert _compute_tcc(node) == pytest.approx(0.3333, abs=0.001)

    def test_excludes_dunders_and_properties(self):
        src = """
class Foo:
    def __init__(self):
        self.x = 1
        self.y = 2

    @property
    def val(self):
        return self.x

    def a(self):
        return self.x

    def b(self):
        return self.y
"""
        node = _parse_class(src)
        # Only a and b are candidates; they don't share fields
        assert _compute_tcc(node) == 0.0


# ── Method Complexity Tests ───────────────────────────────────────────


class TestMethodComplexity:
    def test_simple_return(self):
        src = """
class Foo:
    def bar(self):
        return 1
"""
        tree = ast.parse(src)
        func = tree.body[0].body[0]
        assert _compute_method_complexity(func) == 1

    def test_with_if_and_for(self):
        src = """
class Foo:
    def bar(self, items):
        for item in items:
            if item > 0:
                pass
        return 0
"""
        tree = ast.parse(src)
        func = tree.body[0].body[0]
        # CC = 1 + for + if = 3
        assert _compute_method_complexity(func) == 3

    def test_boolean_ops(self):
        src = """
class Foo:
    def bar(self, a, b, c):
        if a and b or c:
            pass
"""
        tree = ast.parse(src)
        func = tree.body[0].body[0]
        # CC = 1 + if + BoolOp(and: 1) + BoolOp(or: 1)  — depends on ast structure
        # Actually: `a and b or c` parses as BoolOp(Or, [BoolOp(And, [a,b]), c])
        # Inner BoolOp(And, [a,b]): +1
        # Outer BoolOp(Or, [inner, c]): +1
        # Total: 1 + 1(if) + 1(and) + 1(or) = 4
        assert _compute_method_complexity(func) == 4


# ── Field Count Tests ─────────────────────────────────────────────────


class TestFieldCount:
    def test_class_level_attrs(self):
        src = """
class Foo:
    x = 1
    y: int = 2
"""
        node = _parse_class(src)
        assert _count_fields(node) == 2

    def test_init_self_attrs(self):
        src = """
class Foo:
    def __init__(self):
        self.a = 1
        self.b = 2
        self.c = 3
"""
        node = _parse_class(src)
        assert _count_fields(node) == 3

    def test_combined(self):
        src = """
class Foo:
    x: int
    def __init__(self):
        self.y = 1
"""
        node = _parse_class(src)
        assert _count_fields(node) == 2


# ── Candidate Methods Tests ──────────────────────────────────────────


class TestCandidateMethods:
    def test_excludes_dunders(self):
        src = """
class Foo:
    def __init__(self):
        pass
    def __repr__(self):
        return "Foo"
    def bar(self):
        return 1
"""
        node = _parse_class(src)
        candidates = _get_candidate_methods(node)
        assert len(candidates) == 1
        assert candidates[0].name == "bar"

    def test_excludes_properties(self):
        src = """
class Foo:
    @property
    def val(self):
        return self._val
    def bar(self):
        return 1
"""
        node = _parse_class(src)
        candidates = _get_candidate_methods(node)
        assert len(candidates) == 1
        assert candidates[0].name == "bar"


# ── Full Analyzer Tests ──────────────────────────────────────────────


class TestAnalyzePythonGodClasses:
    def test_empty_file(self):
        result = analyze_python_god_classes("", "empty.py")
        assert result.class_count == 0
        assert result.classes == []

    def test_syntax_error(self):
        result = analyze_python_god_classes("def class (broken", "bad.py")
        assert result.class_count == 0
        assert result.classes == []

    def test_single_small_class(self):
        src = """
class Small:
    def __init__(self):
        self.x = 1

    def get_x(self):
        return self.x
"""
        result = analyze_python_god_classes(src, "small.py")
        assert result.class_count == 1
        assert len(result.classes) == 1
        c = result.classes[0]
        assert c.class_name == "Small"
        assert c.method_count == 1  # get_x (excl __init__ dunder)
        assert c.field_count == 1
        assert c.total_complexity >= 1  # at least base CC
        assert c.cohesion == 1.0  # single candidate method
        assert c.god_class_score == 0.0  # raw score before normalization

    def test_multiple_classes(self):
        src = """
class A:
    def foo(self):
        return 1

class B:
    def bar(self):
        return 2
    def baz(self):
        return 3
"""
        result = analyze_python_god_classes(src, "multi.py")
        assert result.class_count == 2
        assert len(result.classes) == 2

    def test_god_class_raw_metrics(self):
        """A large class should have high raw metrics."""
        src = """
class GodClass:
    def __init__(self):
        self.a = 1
        self.b = 2
        self.c = 3
        self.d = 4
        self.e = 5

    def m1(self):
        if self.a > 0:
            for i in range(self.b):
                if i > self.c:
                    pass

    def m2(self):
        while self.d:
            pass

    def m3(self):
        return self.e

    def m4(self):
        if self.a and self.b:
            return True
        return False

    def m5(self):
        try:
            pass
        except Exception:
            pass
"""
        result = analyze_python_god_classes(src, "god.py")
        assert result.class_count == 1
        c = result.classes[0]
        assert c.class_name == "GodClass"
        assert c.field_count == 5
        assert c.method_count == 5  # m1-m5 (excl __init__)
        assert c.total_complexity > 5  # significant WMC
        assert 0.0 <= c.cohesion <= 1.0

    def test_file_path_propagated(self):
        src = "class X:\n    pass\n"
        result = analyze_python_god_classes(src, "path/to/file.py")
        assert result.file_path == "path/to/file.py"
        assert result.classes[0].file_path == "path/to/file.py"
