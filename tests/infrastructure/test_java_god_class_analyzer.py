"""Tests for Java god class detection using tree-sitter."""

import pytest

from git_xrays.infrastructure.java_god_class_analyzer import (
    _compute_method_complexity,
    _compute_tcc,
    _compute_wmc,
    _count_fields,
    _get_candidate_methods,
    analyze_java_god_classes,
)

import tree_sitter_java as tsjava
from tree_sitter import Language, Parser

JAVA_LANGUAGE = Language(tsjava.language())


def _parse_class(source: str):
    """Helper: parse Java source and return the first class/record node."""
    parser = Parser(JAVA_LANGUAGE)
    tree = parser.parse(source.encode("utf-8"))
    for node in tree.root_node.named_children:
        if node.type in ("class_declaration", "record_declaration"):
            return node
    raise ValueError("No class found")


def _parse_method(source: str):
    """Helper: parse Java source and return the first method node."""
    parser = Parser(JAVA_LANGUAGE)
    tree = parser.parse(source.encode("utf-8"))
    cls = None
    for node in tree.root_node.named_children:
        if node.type == "class_declaration":
            cls = node
            break
    if cls is None:
        raise ValueError("No class found")
    body = cls.child_by_field_name("body")
    for child in body.named_children:
        if child.type == "method_declaration":
            return child
    raise ValueError("No method found")


# ── WMC Tests ─────────────────────────────────────────────────────────


class TestJavaWMC:
    def test_empty_class(self):
        src = "public class Empty {}"
        node = _parse_class(src)
        assert _compute_wmc(node) == 0

    def test_single_simple_method(self):
        src = """
public class Foo {
    public void bar() {
        return;
    }
}
"""
        node = _parse_class(src)
        assert _compute_wmc(node) == 1

    def test_method_with_branches(self):
        src = """
public class Foo {
    public int bar(int x) {
        if (x > 0) {
            return x;
        } else if (x < 0) {
            return -x;
        }
        return 0;
    }
}
"""
        node = _parse_class(src)
        # CC = 1 + if + else-if = 3
        assert _compute_wmc(node) == 3

    def test_multiple_methods(self):
        src = """
public class Foo {
    public int a() { return 1; }
    public int b(int x) {
        if (x > 0) return x;
        return 0;
    }
}
"""
        node = _parse_class(src)
        # a: CC=1, b: CC=2 → WMC=3
        assert _compute_wmc(node) == 3

    def test_includes_constructor(self):
        src = """
public class Foo {
    public Foo() {
        if (true) {}
    }
    public void bar() { return; }
}
"""
        node = _parse_class(src)
        # constructor: CC=2, bar: CC=1 → WMC=3
        assert _compute_wmc(node) == 3


# ── TCC Tests ─────────────────────────────────────────────────────────


class TestJavaTCC:
    def test_single_method_returns_one(self):
        src = """
public class Foo {
    private int x;
    public int getVal() { return this.x; }
}
"""
        node = _parse_class(src)
        assert _compute_tcc(node) == 1.0

    def test_no_candidate_methods_returns_one(self):
        src = "public class Foo {}"
        node = _parse_class(src)
        assert _compute_tcc(node) == 1.0

    def test_disjoint_methods_return_zero(self):
        src = """
public class Foo {
    private int x;
    private int y;
    public int a() { return this.x; }
    public int b() { return this.y; }
}
"""
        node = _parse_class(src)
        assert _compute_tcc(node) == 0.0

    def test_all_methods_share_field(self):
        src = """
public class Foo {
    private int x;
    public int a() { return this.x; }
    public int b() { return this.x + 1; }
    public int c() { return this.x * 2; }
}
"""
        node = _parse_class(src)
        # 3 non-getter methods all share this.x → 3/3 = 1.0
        assert _compute_tcc(node) == 1.0

    def test_partial_cohesion(self):
        src = """
public class Foo {
    private int x;
    private int y;
    private int z;
    public int a() { return this.x; }
    public int b() { return this.x + this.y; }
    public int c() { return this.z; }
}
"""
        node = _parse_class(src)
        # Pairs: (a,b)=share x ✓, (a,c)=no ✗, (b,c)=no ✗
        # TCC = 1/3 ≈ 0.3333
        assert _compute_tcc(node) == pytest.approx(0.3333, abs=0.001)

    def test_excludes_getters_setters(self):
        src = """
public class Foo {
    private int x;
    private int y;
    public int getX() { return this.x; }
    public void setX(int x) { this.x = x; }
    public int compute() { return this.x + this.y; }
}
"""
        node = _parse_class(src)
        # Only compute is a candidate → single method → 1.0
        assert _compute_tcc(node) == 1.0


# ── Method Complexity Tests ───────────────────────────────────────────


class TestJavaMethodComplexity:
    def test_simple_return(self):
        src = """
public class Foo {
    public int bar() { return 1; }
}
"""
        method = _parse_method(src)
        assert _compute_method_complexity(method) == 1

    def test_with_if_and_for(self):
        src = """
public class Foo {
    public void bar(int[] items) {
        for (int item : items) {
            if (item > 0) {
                System.out.println(item);
            }
        }
    }
}
"""
        method = _parse_method(src)
        # CC = 1 + enhanced_for + if = 3
        assert _compute_method_complexity(method) == 3

    def test_boolean_ops(self):
        src = """
public class Foo {
    public void bar(boolean a, boolean b) {
        if (a && b) {
            return;
        }
    }
}
"""
        method = _parse_method(src)
        # CC = 1 + if + && = 3
        assert _compute_method_complexity(method) == 3

    def test_try_catch(self):
        src = """
public class Foo {
    public void bar() {
        try {
            int x = 1;
        } catch (Exception e) {
            System.out.println(e);
        }
    }
}
"""
        method = _parse_method(src)
        # CC = 1 + catch = 2
        assert _compute_method_complexity(method) == 2


# ── Field Count Tests ─────────────────────────────────────────────────


class TestJavaFieldCount:
    def test_instance_fields(self):
        src = """
public class Foo {
    private int x;
    private String y;
}
"""
        node = _parse_class(src)
        assert _count_fields(node) == 2

    def test_record_fields(self):
        src = "public record Point(int x, int y) {}"
        node = _parse_class(src)
        assert _count_fields(node) == 2

    def test_multiple_declarators(self):
        src = """
public class Foo {
    private int x, y, z;
}
"""
        node = _parse_class(src)
        assert _count_fields(node) == 3


# ── Candidate Methods Tests ──────────────────────────────────────────


class TestJavaCandidateMethods:
    def test_excludes_getters(self):
        src = """
public class Foo {
    private int x;
    public int getX() { return this.x; }
    public void process() { return; }
}
"""
        node = _parse_class(src)
        candidates = _get_candidate_methods(node)
        assert len(candidates) == 1

    def test_excludes_constructors(self):
        src = """
public class Foo {
    public Foo() {}
    public void bar() { return; }
}
"""
        node = _parse_class(src)
        candidates = _get_candidate_methods(node)
        assert len(candidates) == 1


# ── Full Analyzer Tests ──────────────────────────────────────────────


class TestAnalyzeJavaGodClasses:
    def test_empty_class(self):
        src = "public class Empty {}"
        result = analyze_java_god_classes(src, "Empty.java")
        assert result.class_count == 1
        assert result.classes[0].method_count == 0
        assert result.classes[0].field_count == 0

    def test_parse_error(self):
        result = analyze_java_god_classes("public broken {{{", "bad.java")
        assert result.class_count == 0
        assert result.classes == []

    def test_small_pojo(self):
        src = """
public class User {
    private String name;
    private int age;
    public String getName() { return this.name; }
    public void setName(String n) { this.name = n; }
    public int getAge() { return this.age; }
}
"""
        result = analyze_java_god_classes(src, "User.java")
        assert result.class_count == 1
        c = result.classes[0]
        assert c.class_name == "User"
        assert c.field_count == 2
        assert c.method_count == 0  # all are getters/setters
        assert c.god_class_score == 0.0  # raw

    def test_large_service_raw_metrics(self):
        src = """
public class OrderService {
    private int count;
    private int total;
    private String status;
    private boolean active;
    private int limit;

    public void processA() {
        if (this.active) {
            for (int i = 0; i < this.count; i++) {
                this.total += i;
            }
        }
    }
    public void processB() {
        while (this.count > 0) {
            this.count--;
        }
    }
    public void processC() {
        if (this.total > this.limit) {
            this.status = "over";
        }
    }
    public void processD() {
        try {
            this.count = 0;
        } catch (Exception e) {
            this.active = false;
        }
    }
    public void processE() {
        if (this.active && this.count > 0) {
            return;
        }
    }
}
"""
        result = analyze_java_god_classes(src, "OrderService.java")
        assert result.class_count == 1
        c = result.classes[0]
        assert c.field_count == 5
        assert c.method_count == 5
        assert c.total_complexity > 5

    def test_record_class(self):
        src = "public record Point(int x, int y) {}"
        result = analyze_java_god_classes(src, "Point.java")
        assert result.class_count == 1
        c = result.classes[0]
        assert c.class_name == "Point"
        assert c.field_count == 2
        assert c.method_count == 0

    def test_multiple_classes(self):
        src = """
public class A {
    public void foo() { return; }
}
class B {
    public void bar() { return; }
}
"""
        result = analyze_java_god_classes(src, "Multi.java")
        assert result.class_count == 2

    def test_file_path_propagated(self):
        src = "public class X {}"
        result = analyze_java_god_classes(src, "com/example/X.java")
        assert result.file_path == "com/example/X.java"
        assert result.classes[0].file_path == "com/example/X.java"
