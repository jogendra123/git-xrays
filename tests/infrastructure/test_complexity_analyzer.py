import textwrap

from git_xrays.infrastructure.complexity_analyzer import analyze_file_complexity


def _analyze_func(source: str, func_name: str = "f"):
    """Helper: parse source, return the FunctionComplexity for func_name."""
    result = analyze_file_complexity(textwrap.dedent(source), "test.py")
    funcs = {f.function_name: f for f in result.functions}
    assert func_name in funcs, f"{func_name} not found in {list(funcs)}"
    return funcs[func_name]


class TestCyclomaticComplexity:
    def test_empty_function_complexity_one(self):
        fc = _analyze_func("""\
            def f():
                pass
        """)
        assert fc.cyclomatic_complexity == 1

    def test_single_if_adds_one(self):
        fc = _analyze_func("""\
            def f(x):
                if x:
                    return 1
                return 0
        """)
        assert fc.cyclomatic_complexity == 2

    def test_if_elif_else(self):
        fc = _analyze_func("""\
            def f(x):
                if x > 0:
                    return 1
                elif x < 0:
                    return -1
                else:
                    return 0
        """)
        assert fc.cyclomatic_complexity == 3

    def test_for_loop_adds_one(self):
        fc = _analyze_func("""\
            def f(items):
                for item in items:
                    print(item)
        """)
        assert fc.cyclomatic_complexity == 2

    def test_while_loop_adds_one(self):
        fc = _analyze_func("""\
            def f():
                while True:
                    break
        """)
        assert fc.cyclomatic_complexity == 2

    def test_except_handler_adds_one(self):
        fc = _analyze_func("""\
            def f():
                try:
                    pass
                except Exception:
                    pass
        """)
        assert fc.cyclomatic_complexity == 2

    def test_boolean_and_adds_one(self):
        fc = _analyze_func("""\
            def f(a, b):
                if a and b:
                    return True
                return False
        """)
        assert fc.cyclomatic_complexity == 3

    def test_boolean_or_adds_one(self):
        fc = _analyze_func("""\
            def f(a, b):
                if a or b:
                    return True
                return False
        """)
        assert fc.cyclomatic_complexity == 3

    def test_assert_adds_one(self):
        fc = _analyze_func("""\
            def f(x):
                assert x > 0
        """)
        assert fc.cyclomatic_complexity == 2

    def test_ternary_ifexp_adds_one(self):
        fc = _analyze_func("""\
            def f(cond):
                x = 1 if cond else 0
                return x
        """)
        assert fc.cyclomatic_complexity == 2


class TestNestingDepth:
    def test_flat_function_depth_zero(self):
        fc = _analyze_func("""\
            def f():
                x = 1
                return x
        """)
        assert fc.max_nesting_depth == 0

    def test_single_if_depth_one(self):
        fc = _analyze_func("""\
            def f(x):
                if x:
                    return 1
                return 0
        """)
        assert fc.max_nesting_depth == 1

    def test_nested_if_depth_two(self):
        fc = _analyze_func("""\
            def f(x, y):
                if x:
                    if y:
                        return 1
                return 0
        """)
        assert fc.max_nesting_depth == 2

    def test_for_with_if_depth_two(self):
        fc = _analyze_func("""\
            def f(items):
                for item in items:
                    if item:
                        print(item)
        """)
        assert fc.max_nesting_depth == 2

    def test_try_with_if_depth_two(self):
        fc = _analyze_func("""\
            def f():
                try:
                    if True:
                        pass
                except Exception:
                    pass
        """)
        assert fc.max_nesting_depth == 2


class TestBranchCount:
    def test_no_branches(self):
        fc = _analyze_func("""\
            def f():
                pass
        """)
        assert fc.branch_count == 0

    def test_single_if(self):
        fc = _analyze_func("""\
            def f(x):
                if x:
                    return 1
                return 0
        """)
        assert fc.branch_count == 1

    def test_if_elif_elif(self):
        fc = _analyze_func("""\
            def f(x):
                if x > 0:
                    return 1
                elif x < 0:
                    return -1
                elif x == 0:
                    return 0
        """)
        assert fc.branch_count == 3


class TestExceptionPaths:
    def test_no_except_handlers(self):
        fc = _analyze_func("""\
            def f():
                pass
        """)
        assert fc.exception_paths == 0

    def test_single_except(self):
        fc = _analyze_func("""\
            def f():
                try:
                    pass
                except Exception:
                    pass
        """)
        assert fc.exception_paths == 1

    def test_multiple_except_handlers(self):
        fc = _analyze_func("""\
            def f():
                try:
                    pass
                except ValueError:
                    pass
                except TypeError:
                    pass
                except Exception:
                    pass
        """)
        assert fc.exception_paths == 3


class TestFunctionLength:
    def test_single_line_function(self):
        fc = _analyze_func("""\
            def f(): pass
        """)
        assert fc.length == 1

    def test_multiline_function(self):
        fc = _analyze_func("""\
            def f():
                x = 1
                y = 2
                return x + y
        """)
        assert fc.length == 4

    def test_function_with_blank_lines(self):
        fc = _analyze_func("""\
            def f():
                x = 1

                y = 2

                return x + y
        """)
        assert fc.length == 6


class TestFunctionMetadata:
    def test_top_level_function_class_name_none(self):
        fc = _analyze_func("""\
            def f():
                pass
        """)
        assert fc.class_name is None

    def test_class_method_has_class_name(self):
        result = analyze_file_complexity(textwrap.dedent("""\
            class MyClass:
                def method(self):
                    pass
        """), "test.py")
        funcs = {f.function_name: f for f in result.functions}
        assert "method" in funcs
        assert funcs["method"].class_name == "MyClass"

    def test_line_number_captured(self):
        result = analyze_file_complexity(textwrap.dedent("""\
            x = 1
            y = 2

            def f():
                pass
        """), "test.py")
        funcs = {f.function_name: f for f in result.functions}
        assert funcs["f"].line_number == 4

    def test_file_path_passed_through(self):
        fc = _analyze_func("""\
            def f():
                pass
        """)
        assert fc.file_path == "test.py"


class TestAnalyzeFileComplexity:
    def test_empty_file_returns_empty(self):
        result = analyze_file_complexity("", "empty.py")
        assert result.function_count == 0
        assert result.functions == []
        assert result.worst_function == ""

    def test_single_function(self):
        result = analyze_file_complexity(textwrap.dedent("""\
            def process(x):
                if x > 0:
                    return x
                return 0
        """), "svc.py")
        assert result.function_count == 1
        assert result.max_complexity == 2
        assert result.worst_function == "process"

    def test_multiple_functions_sorted_by_complexity_desc(self):
        result = analyze_file_complexity(textwrap.dedent("""\
            def simple():
                pass

            def complex_func(x, y, z):
                if x:
                    if y:
                        if z:
                            return 1
                return 0
        """), "test.py")
        assert result.function_count == 2
        assert result.functions[0].function_name == "complex_func"
        assert result.functions[1].function_name == "simple"

    def test_class_methods_included(self):
        result = analyze_file_complexity(textwrap.dedent("""\
            class Foo:
                def bar(self):
                    if True:
                        pass
                def baz(self):
                    pass
        """), "test.py")
        assert result.function_count == 2
        names = {f.function_name for f in result.functions}
        assert names == {"bar", "baz"}

    def test_mixed_top_level_and_methods(self):
        result = analyze_file_complexity(textwrap.dedent("""\
            def top_level():
                pass

            class MyClass:
                def method(self):
                    pass
        """), "test.py")
        assert result.function_count == 2
        names = {f.function_name for f in result.functions}
        assert names == {"top_level", "method"}

    def test_nested_functions_skipped(self):
        result = analyze_file_complexity(textwrap.dedent("""\
            def outer():
                def inner():
                    pass
                return inner()
        """), "test.py")
        # Only outer should be counted, inner is nested
        assert result.function_count == 1
        assert result.functions[0].function_name == "outer"

    def test_avg_and_max_length(self):
        result = analyze_file_complexity(textwrap.dedent("""\
            def short():
                pass

            def longer():
                x = 1
                y = 2
                z = 3
                return x + y + z
        """), "test.py")
        assert result.max_length == 5  # longer: 5 lines
        assert result.avg_length == 3.5  # (2 + 5) / 2 = 3.5

    def test_syntax_error_returns_empty(self):
        result = analyze_file_complexity("def f(:\n", "bad.py")
        assert result.function_count == 0
        assert result.functions == []


class TestComplexCombinations:
    def test_high_complexity_function(self):
        """Realistic function with multiple decision points."""
        fc = _analyze_func("""\
            def f(data, config):
                try:
                    for item in data:
                        if item.active:
                            if item.value > config.threshold:
                                if item.priority == 'high':
                                    process(item)
                                elif item.priority == 'low':
                                    defer(item)
                            else:
                                skip(item)
                except ValueError:
                    handle_error()
        """)
        # 1 + for + if + if + if + elif(nested If) + except = 7
        assert fc.cyclomatic_complexity == 7
        # try(1) > for(2) > if(3) > if(4) > if(5), elif is If in orelse at depth(6)
        assert fc.max_nesting_depth == 6
        assert fc.branch_count == 4
        assert fc.exception_paths == 1

    def test_boolean_chain_complexity(self):
        """a and b and c or d → BoolOp(And, [a,b,c]) + If + BoolOp(Or, [And, d])."""
        fc = _analyze_func("""\
            def f(a, b, c, d):
                if a and b and c or d:
                    return True
                return False
        """)
        # if=1, or has 2 values → +1, and has 3 values → +2, total = 1+1+1+2 = 5
        assert fc.cyclomatic_complexity == 5

    def test_multiple_except_handlers_counted(self):
        fc = _analyze_func("""\
            def f():
                try:
                    pass
                except ValueError:
                    pass
                except TypeError:
                    pass
                except RuntimeError:
                    pass
        """)
        assert fc.exception_paths == 3
        assert fc.cyclomatic_complexity == 4  # 1 + 3 handlers

    def test_deeply_nested_with_for_while(self):
        """Depth=4 with for/while/if/with."""
        fc = _analyze_func("""\
            def f(items):
                for item in items:
                    while item.has_next():
                        if item.valid():
                            with item.lock():
                                process(item)
        """)
        assert fc.max_nesting_depth == 4
