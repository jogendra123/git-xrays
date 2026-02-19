"""Tests for Java complexity analysis using tree-sitter."""

import pytest

from git_xrays.infrastructure.java_complexity_analyzer import (
    analyze_java_file_complexity,
)


class TestJavaCyclomaticComplexity:
    def test_empty_method_has_cc_1(self):
        source = """
public class Foo {
    public void doNothing() {
    }
}
"""
        fc = analyze_java_file_complexity(source, "Foo.java")
        assert fc.function_count == 1
        assert fc.functions[0].cyclomatic_complexity == 1

    def test_single_if(self):
        source = """
public class Foo {
    public int check(int x) {
        if (x > 0) {
            return x;
        }
        return 0;
    }
}
"""
        fc = analyze_java_file_complexity(source, "Foo.java")
        assert fc.functions[0].cyclomatic_complexity == 2

    def test_if_else_if(self):
        source = """
public class Foo {
    public String classify(int x) {
        if (x > 0) {
            return "positive";
        } else if (x < 0) {
            return "negative";
        } else {
            return "zero";
        }
    }
}
"""
        fc = analyze_java_file_complexity(source, "Foo.java")
        # 1 + if + else-if = 3
        assert fc.functions[0].cyclomatic_complexity == 3

    def test_for_loop(self):
        source = """
public class Foo {
    public void loop(int[] arr) {
        for (int i = 0; i < arr.length; i++) {
            System.out.println(arr[i]);
        }
    }
}
"""
        fc = analyze_java_file_complexity(source, "Foo.java")
        assert fc.functions[0].cyclomatic_complexity == 2

    def test_enhanced_for_loop(self):
        source = """
public class Foo {
    public void loop(int[] arr) {
        for (int x : arr) {
            System.out.println(x);
        }
    }
}
"""
        fc = analyze_java_file_complexity(source, "Foo.java")
        assert fc.functions[0].cyclomatic_complexity == 2

    def test_while_loop(self):
        source = """
public class Foo {
    public void loop(int n) {
        while (n > 0) {
            n--;
        }
    }
}
"""
        fc = analyze_java_file_complexity(source, "Foo.java")
        assert fc.functions[0].cyclomatic_complexity == 2

    def test_do_while_loop(self):
        source = """
public class Foo {
    public void loop(int n) {
        do {
            n--;
        } while (n > 0);
    }
}
"""
        fc = analyze_java_file_complexity(source, "Foo.java")
        assert fc.functions[0].cyclomatic_complexity == 2

    def test_try_catch(self):
        source = """
public class Foo {
    public void risky() {
        try {
            doSomething();
        } catch (Exception e) {
            handleError(e);
        }
    }
}
"""
        fc = analyze_java_file_complexity(source, "Foo.java")
        # 1 + catch = 2
        assert fc.functions[0].cyclomatic_complexity == 2

    def test_multiple_catch(self):
        source = """
public class Foo {
    public void risky() {
        try {
            doSomething();
        } catch (IOException e) {
            handleIO(e);
        } catch (Exception e) {
            handleGeneral(e);
        }
    }
}
"""
        fc = analyze_java_file_complexity(source, "Foo.java")
        # 1 + 2 catches = 3
        assert fc.functions[0].cyclomatic_complexity == 3

    def test_ternary_expression(self):
        source = """
public class Foo {
    public int pick(boolean flag) {
        return flag ? 1 : 0;
    }
}
"""
        fc = analyze_java_file_complexity(source, "Foo.java")
        # 1 + ternary = 2
        assert fc.functions[0].cyclomatic_complexity == 2

    def test_logical_and(self):
        source = """
public class Foo {
    public boolean check(int a, int b) {
        return a > 0 && b > 0;
    }
}
"""
        fc = analyze_java_file_complexity(source, "Foo.java")
        # 1 + && = 2
        assert fc.functions[0].cyclomatic_complexity == 2

    def test_logical_or(self):
        source = """
public class Foo {
    public boolean check(int a, int b) {
        return a > 0 || b > 0;
    }
}
"""
        fc = analyze_java_file_complexity(source, "Foo.java")
        # 1 + || = 2
        assert fc.functions[0].cyclomatic_complexity == 2

    def test_mixed_boolean_ops(self):
        source = """
public class Foo {
    public boolean check(int a, int b, int c) {
        if (a > 0 && b > 0 || c > 0) {
            return true;
        }
        return false;
    }
}
"""
        fc = analyze_java_file_complexity(source, "Foo.java")
        # 1 + if + && + || = 4
        assert fc.functions[0].cyclomatic_complexity == 4

    def test_complex_method(self):
        source = """
public class Foo {
    public void process(int[] data, String mode) {
        if (data == null) {
            return;
        }
        for (int item : data) {
            if (mode.equals("strict")) {
                if (item > 0) {
                    System.out.println(item);
                }
            } else if (mode.equals("lenient")) {
                System.out.println(item);
            }
        }
    }
}
"""
        fc = analyze_java_file_complexity(source, "Foo.java")
        # 1 + if(null) + for + if(strict) + if(item>0) + else-if(lenient) = 6
        assert fc.functions[0].cyclomatic_complexity == 6


class TestJavaNestingDepth:
    def test_flat_method(self):
        source = """
public class Foo {
    public void flat() {
        System.out.println("hello");
    }
}
"""
        fc = analyze_java_file_complexity(source, "Foo.java")
        assert fc.functions[0].max_nesting_depth == 0

    def test_single_nesting(self):
        source = """
public class Foo {
    public void nested(int x) {
        if (x > 0) {
            System.out.println(x);
        }
    }
}
"""
        fc = analyze_java_file_complexity(source, "Foo.java")
        assert fc.functions[0].max_nesting_depth == 1

    def test_deeply_nested(self):
        source = """
public class Foo {
    public void deep(int[] data) {
        for (int item : data) {
            if (item > 0) {
                try {
                    process(item);
                } catch (Exception e) {
                    handleError(e);
                }
            }
        }
    }
}
"""
        fc = analyze_java_file_complexity(source, "Foo.java")
        assert fc.functions[0].max_nesting_depth == 3


class TestJavaBranchCount:
    def test_no_branches(self):
        source = """
public class Foo {
    public void noBranch() {
        System.out.println("hello");
    }
}
"""
        fc = analyze_java_file_complexity(source, "Foo.java")
        assert fc.functions[0].branch_count == 0

    def test_multiple_if_branches(self):
        source = """
public class Foo {
    public void branches(int x, int y) {
        if (x > 0) { foo(); }
        if (y > 0) { bar(); }
    }
}
"""
        fc = analyze_java_file_complexity(source, "Foo.java")
        assert fc.functions[0].branch_count == 2


class TestJavaExceptionPaths:
    def test_no_exceptions(self):
        source = """
public class Foo {
    public void safe() {
        System.out.println("safe");
    }
}
"""
        fc = analyze_java_file_complexity(source, "Foo.java")
        assert fc.functions[0].exception_paths == 0

    def test_multiple_catches(self):
        source = """
public class Foo {
    public void risky() {
        try {
            step1();
        } catch (IOException e) {
            handle1(e);
        } catch (Exception e) {
            handle2(e);
        }
    }
}
"""
        fc = analyze_java_file_complexity(source, "Foo.java")
        assert fc.functions[0].exception_paths == 2


class TestJavaMethodMetadata:
    def test_class_name_captured(self):
        source = """
public class MyService {
    public void serve() { }
}
"""
        fc = analyze_java_file_complexity(source, "Service.java")
        assert fc.functions[0].class_name == "MyService"

    def test_line_number_captured(self):
        source = """public class Foo {
    public void first() { }
    public void second() { }
}
"""
        fc = analyze_java_file_complexity(source, "Foo.java")
        names = {f.function_name: f for f in fc.functions}
        assert names["first"].line_number == 2
        assert names["second"].line_number == 3


class TestAnalyzeJavaFileComplexity:
    def test_empty_class(self):
        source = """
public class Empty {
}
"""
        fc = analyze_java_file_complexity(source, "Empty.java")
        assert fc.function_count == 0
        assert fc.functions == []

    def test_single_method(self):
        source = """
public class Single {
    public int getValue() {
        return 42;
    }
}
"""
        fc = analyze_java_file_complexity(source, "Single.java")
        assert fc.function_count == 1
        assert fc.max_complexity == 1
        assert fc.worst_function == "getValue"

    def test_multiple_classes(self):
        source = """
public class A {
    public void doA() {
        if (true) { }
    }
}

class B {
    public void doB() {
        for (int i = 0; i < 10; i++) {
            if (i > 5) { }
        }
    }
}
"""
        fc = analyze_java_file_complexity(source, "Multi.java")
        assert fc.function_count == 2
        # B.doB has CC=3 (1+for+if), A.doA has CC=2 (1+if)
        assert fc.files if hasattr(fc, "files") else True
        assert fc.max_complexity == 3
        assert fc.worst_function == "doB"

    def test_record_class(self):
        source = """
public record Point(int x, int y) {
    public double distance() {
        return Math.sqrt(x * x + y * y);
    }
}
"""
        fc = analyze_java_file_complexity(source, "Point.java")
        assert fc.function_count == 1
        assert fc.functions[0].class_name == "Point"

    def test_constructor(self):
        source = """
public class Foo {
    public Foo(int x) {
        if (x < 0) {
            throw new IllegalArgumentException();
        }
    }
}
"""
        fc = analyze_java_file_complexity(source, "Foo.java")
        assert fc.function_count == 1
        assert fc.functions[0].cyclomatic_complexity == 2

    def test_aggregation_correct(self):
        source = """
public class Svc {
    public void simple() { }
    public void complex(int x) {
        if (x > 0) {
            for (int i = 0; i < x; i++) {
                System.out.println(i);
            }
        }
    }
}
"""
        fc = analyze_java_file_complexity(source, "Svc.java")
        assert fc.function_count == 2
        assert fc.total_complexity == 1 + 3  # simple=1, complex=3
        assert fc.avg_complexity == 2.0
        assert fc.max_complexity == 3


class TestJavaCognitiveComplexity:
    def test_flat_method_zero(self):
        source = """
public class Foo {
    public void flat() {
        System.out.println("hello");
    }
}
"""
        fc = analyze_java_file_complexity(source, "Foo.java")
        assert fc.functions[0].cognitive_complexity == 0

    def test_single_if(self):
        source = """
public class Foo {
    public void check(int x) {
        if (x > 0) {
            System.out.println(x);
        }
    }
}
"""
        fc = analyze_java_file_complexity(source, "Foo.java")
        # if at depth 0 → +1
        assert fc.functions[0].cognitive_complexity == 1

    def test_nested_if(self):
        source = """
public class Foo {
    public void check(int x, int y) {
        if (x > 0) {
            if (y > 0) {
                System.out.println("both");
            }
        }
    }
}
"""
        fc = analyze_java_file_complexity(source, "Foo.java")
        # outer if: 1+0=1, inner if: 1+1=2, total=3
        assert fc.functions[0].cognitive_complexity == 3

    def test_if_else(self):
        source = """
public class Foo {
    public String check(int x) {
        if (x > 0) {
            return "positive";
        } else {
            return "non-positive";
        }
    }
}
"""
        fc = analyze_java_file_complexity(source, "Foo.java")
        # if: 1+0=1, else: +1, total=2
        assert fc.functions[0].cognitive_complexity == 2

    def test_if_else_if_else(self):
        source = """
public class Foo {
    public String classify(int x) {
        if (x > 0) {
            return "positive";
        } else if (x < 0) {
            return "negative";
        } else {
            return "zero";
        }
    }
}
"""
        fc = analyze_java_file_complexity(source, "Foo.java")
        # if: 1, else if: 1, else: 1, total=3
        assert fc.functions[0].cognitive_complexity == 3

    def test_for_at_depth(self):
        source = """
public class Foo {
    public void process(int[] data) {
        if (data != null) {
            for (int x : data) {
                System.out.println(x);
            }
        }
    }
}
"""
        fc = analyze_java_file_complexity(source, "Foo.java")
        # if: 1+0=1, for: 1+1=2, total=3
        assert fc.functions[0].cognitive_complexity == 3

    def test_boolean_ops_in_condition(self):
        source = """
public class Foo {
    public void check(int a, int b) {
        if (a > 0 && b > 0) {
            System.out.println("both");
        }
    }
}
"""
        fc = analyze_java_file_complexity(source, "Foo.java")
        # if: 1+0=1, && in condition: +1, total=2
        assert fc.functions[0].cognitive_complexity == 2

    def test_try_catch(self):
        source = """
public class Foo {
    public void risky() {
        try {
            doSomething();
        } catch (Exception e) {
            handleError(e);
        }
    }
}
"""
        fc = analyze_java_file_complexity(source, "Foo.java")
        # catch: 1+0=1, total=1
        assert fc.functions[0].cognitive_complexity == 1
