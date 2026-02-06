import pytest

from git_xrays.infrastructure.ast_analyzer import (
    analyze_class_source,
    analyze_file,
    compute_touch_counts,
)


# --- Step 3: Field counting ---

class TestFieldCounting:
    def test_class_level_attributes_counted(self):
        source = "class Foo:\n    x = 1\n    y = 2\n"
        result = analyze_file(source, "foo.py")
        assert result.classes[0].field_count == 2

    def test_init_self_assignments_counted(self):
        source = (
            "class Foo:\n"
            "    def __init__(self):\n"
            "        self.a = 1\n"
            "        self.b = 2\n"
        )
        result = analyze_file(source, "foo.py")
        assert result.classes[0].field_count == 2

    def test_combined_class_and_init_fields(self):
        source = (
            "class Foo:\n"
            "    x = 1\n"
            "    def __init__(self):\n"
            "        self.a = 1\n"
        )
        result = analyze_file(source, "foo.py")
        assert result.classes[0].field_count == 2

    def test_no_fields_zero_count(self):
        source = "class Empty:\n    pass\n"
        result = analyze_file(source, "empty.py")
        assert result.classes[0].field_count == 0

    def test_self_in_non_init_not_counted(self):
        source = (
            "class Foo:\n"
            "    def do_stuff(self):\n"
            "        self.x = 42\n"
        )
        result = analyze_file(source, "foo.py")
        assert result.classes[0].field_count == 0

    def test_annotated_assignments_counted(self):
        source = "class Foo:\n    x: int\n    y: str\n"
        result = analyze_file(source, "foo.py")
        assert result.classes[0].field_count == 2


# --- Step 4: Method counting & classification ---

class TestMethodCounting:
    def test_method_count_all_methods(self):
        source = (
            "class Foo:\n"
            "    def __init__(self): pass\n"
            "    def do_stuff(self): pass\n"
            "    def __str__(self): return ''\n"
        )
        result = analyze_file(source, "foo.py")
        assert result.classes[0].method_count == 3

    def test_dunder_methods_identified(self):
        source = (
            "class Foo:\n"
            "    def __init__(self): pass\n"
            "    def __str__(self): return ''\n"
            "    def normal(self): pass\n"
        )
        result = analyze_file(source, "foo.py")
        assert result.classes[0].dunder_method_count == 2

    def test_property_methods_identified(self):
        source = (
            "class Foo:\n"
            "    @property\n"
            "    def name(self): return self._name\n"
            "    def other(self): pass\n"
        )
        result = analyze_file(source, "foo.py")
        assert result.classes[0].property_count == 1

    def test_behavior_method_with_if(self):
        source = (
            "class Foo:\n"
            "    def check(self):\n"
            "        if self.x > 0:\n"
            "            return True\n"
            "        return False\n"
        )
        result = analyze_file(source, "foo.py")
        assert result.classes[0].behavior_method_count == 1

    def test_behavior_method_with_for(self):
        source = (
            "class Foo:\n"
            "    def process(self):\n"
            "        for item in self.items:\n"
            "            print(item)\n"
        )
        result = analyze_file(source, "foo.py")
        assert result.classes[0].behavior_method_count == 1

    def test_behavior_method_with_while_try_with(self):
        source = (
            "class Foo:\n"
            "    def a(self):\n"
            "        while True: break\n"
            "    def b(self):\n"
            "        try:\n"
            "            pass\n"
            "        except: pass\n"
            "    def c(self):\n"
            "        with open('f') as f: pass\n"
        )
        result = analyze_file(source, "foo.py")
        assert result.classes[0].behavior_method_count == 3

    def test_pass_through_not_behavior(self):
        source = (
            "class Foo:\n"
            "    def get_x(self):\n"
            "        return self.x\n"
            "    def noop(self):\n"
            "        pass\n"
        )
        result = analyze_file(source, "foo.py")
        assert result.classes[0].behavior_method_count == 0


# --- Step 5: Derived metrics ---

class TestDerivedMetrics:
    def test_pure_data_class_dbsi_one(self):
        source = "class Data:\n    x = 1\n    y = 2\n    z = 3\n"
        result = analyze_file(source, "data.py")
        assert result.classes[0].dbsi == 1.0

    def test_pure_behavior_class_dbsi_zero(self):
        source = (
            "class Service:\n"
            "    def process(self):\n"
            "        if True:\n"
            "            return 42\n"
        )
        result = analyze_file(source, "svc.py")
        assert result.classes[0].dbsi == 0.0

    def test_mixed_class_dbsi_fractional(self):
        source = (
            "class Mixed:\n"
            "    x = 1\n"
            "    y = 2\n"
            "    z = 3\n"
            "    def do_a(self):\n"
            "        if True: return 1\n"
            "    def do_b(self):\n"
            "        for i in []: pass\n"
        )
        result = analyze_file(source, "mixed.py")
        # 3 fields / (3 fields + 2 behavior) = 0.6
        assert result.classes[0].dbsi == 0.6

    def test_empty_class_dbsi_zero(self):
        source = "class Empty:\n    pass\n"
        result = analyze_file(source, "empty.py")
        assert result.classes[0].dbsi == 0.0

    def test_logic_density_all_logic(self):
        source = (
            "class Svc:\n"
            "    def a(self):\n"
            "        if True: pass\n"
            "    def b(self):\n"
            "        for x in []: pass\n"
        )
        result = analyze_file(source, "svc.py")
        assert result.classes[0].logic_density == 1.0

    def test_logic_density_no_logic(self):
        source = (
            "class DTO:\n"
            "    def get_x(self):\n"
            "        return self.x\n"
            "    def get_y(self):\n"
            "        return self.y\n"
        )
        result = analyze_file(source, "dto.py")
        assert result.classes[0].logic_density == 0.0

    def test_orchestration_pressure_inverse(self):
        source = (
            "class Svc:\n"
            "    def a(self):\n"
            "        if True: pass\n"
            "    def b(self):\n"
            "        return self.x\n"
        )
        result = analyze_file(source, "svc.py")
        # logic_density = 1/2 = 0.5, orchestration = 1 - 0.5 = 0.5
        assert result.classes[0].logic_density == 0.5
        assert result.classes[0].orchestration_pressure == 0.5

    def test_ams_product(self):
        source = (
            "class Anemic:\n"
            "    x = 1\n"
            "    y = 2\n"
            "    z = 3\n"
            "    def get_x(self):\n"
            "        return self.x\n"
            "    def get_y(self):\n"
            "        return self.y\n"
        )
        result = analyze_file(source, "anemic.py")
        cm = result.classes[0]
        # dbsi = 3 / (3 + 0) = 1.0
        # logic_density = 0/2 = 0.0, orchestration = 1.0
        # ams = 1.0 * 1.0 = 1.0
        assert cm.ams == cm.dbsi * cm.orchestration_pressure


# --- Step 6: File-level analysis ---

class TestAnalyzeFile:
    def test_single_class_file(self):
        source = "class Foo:\n    x = 1\n"
        result = analyze_file(source, "foo.py")
        assert result.class_count == 1

    def test_multiple_classes_sorted_by_ams_desc(self):
        source = (
            "class Anemic:\n"
            "    x = 1\n"
            "    y = 2\n"
            "    def get_x(self):\n"
            "        return self.x\n"
            "\n"
            "class Healthy:\n"
            "    def process(self):\n"
            "        if True:\n"
            "            return 42\n"
        )
        result = analyze_file(source, "multi.py")
        assert result.class_count == 2
        assert result.classes[0].ams >= result.classes[1].ams

    def test_anemic_class_above_threshold(self):
        source = (
            "class Anemic:\n"
            "    x = 1\n"
            "    y = 2\n"
            "    z = 3\n"
            "    def get_x(self):\n"
            "        return self.x\n"
        )
        result = analyze_file(source, "anemic.py", ams_threshold=0.5)
        assert result.anemic_class_count == 1

    def test_healthy_class_below_threshold(self):
        source = (
            "class Healthy:\n"
            "    def process(self):\n"
            "        if True:\n"
            "            return 42\n"
        )
        result = analyze_file(source, "healthy.py", ams_threshold=0.5)
        assert result.anemic_class_count == 0

    def test_empty_file_no_classes(self):
        result = analyze_file("", "empty.py")
        assert result.class_count == 0
        assert result.worst_ams == 0.0

    def test_custom_ams_threshold(self):
        source = (
            "class Semi:\n"
            "    x = 1\n"
            "    def a(self):\n"
            "        return self.x\n"
            "    def b(self):\n"
            "        if True: pass\n"
        )
        # dbsi = 1/(1+1)=0.5, logic_density=0.5, orch=0.5, ams=0.25
        result_low = analyze_file(source, "semi.py", ams_threshold=0.2)
        result_high = analyze_file(source, "semi.py", ams_threshold=0.3)
        assert result_low.anemic_class_count == 1
        assert result_high.anemic_class_count == 0


# --- Step 7: Touch count (import analysis) ---

class TestTouchCount:
    def test_single_import_counted(self):
        files = {
            "foo.py": "x = 1",
            "bar.py": "import foo",
        }
        touches = compute_touch_counts(files)
        assert touches["foo.py"] == 1

    def test_from_import_counted(self):
        files = {
            "models.py": "class Foo: pass",
            "app.py": "from models import Foo",
        }
        touches = compute_touch_counts(files)
        assert touches["models.py"] == 1

    def test_multiple_importers_accumulated(self):
        files = {
            "models.py": "class Foo: pass",
            "a.py": "import models",
            "b.py": "from models import Foo",
            "c.py": "import models",
        }
        touches = compute_touch_counts(files)
        assert touches["models.py"] == 3

    def test_self_import_not_counted(self):
        files = {
            "models.py": "import models",
        }
        touches = compute_touch_counts(files)
        assert touches["models.py"] == 0

    def test_no_imports_zero_touch(self):
        files = {
            "standalone.py": "x = 1",
        }
        touches = compute_touch_counts(files)
        assert touches["standalone.py"] == 0
