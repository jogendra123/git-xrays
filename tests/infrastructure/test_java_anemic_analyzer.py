"""Tests for Java anemic domain model detection using tree-sitter."""

import pytest

from git_xrays.infrastructure.java_anemic_analyzer import (
    analyze_java_file_anemic,
    compute_java_touch_counts,
)


class TestJavaFieldCounting:
    def test_instance_fields(self):
        source = """
public class User {
    private String name;
    private String email;
    private int age;
}
"""
        fa = analyze_java_file_anemic(source, "User.java")
        assert fa.classes[0].field_count == 3

    def test_static_fields_counted(self):
        source = """
public class Config {
    private static final String DEFAULT = "hello";
    private int value;
}
"""
        fa = analyze_java_file_anemic(source, "Config.java")
        assert fa.classes[0].field_count == 2

    def test_multiple_declarators(self):
        source = """
public class Foo {
    private int x, y;
}
"""
        fa = analyze_java_file_anemic(source, "Foo.java")
        assert fa.classes[0].field_count == 2

    def test_record_fields(self):
        source = """
public record Point(int x, int y) {
}
"""
        fa = analyze_java_file_anemic(source, "Point.java")
        assert fa.classes[0].field_count == 2


class TestJavaMethodCounting:
    def test_all_methods_counted(self):
        source = """
public class Svc {
    public void doA() { }
    public void doB() { }
    public void doC() { }
}
"""
        fa = analyze_java_file_anemic(source, "Svc.java")
        assert fa.classes[0].method_count == 3

    def test_getter_setter_detected(self):
        source = """
public class User {
    private String name;
    public String getName() {
        return this.name;
    }
    public void setName(String name) {
        this.name = name;
    }
    public boolean isActive() {
        return true;
    }
}
"""
        fa = analyze_java_file_anemic(source, "User.java")
        assert fa.classes[0].property_count == 3  # getName, setName, isActive

    def test_getter_with_logic_not_counted_as_getter(self):
        source = """
public class User {
    private String name;
    public String getName() {
        if (name == null) {
            return "unknown";
        }
        return name;
    }
}
"""
        fa = analyze_java_file_anemic(source, "User.java")
        # getName has logic, so not a simple getter
        assert fa.classes[0].property_count == 0

    def test_behavior_methods(self):
        source = """
public class Svc {
    public void process(int x) {
        if (x > 0) {
            System.out.println(x);
        }
    }
    public void iterate(int[] arr) {
        for (int item : arr) {
            System.out.println(item);
        }
    }
    public void simple() {
        System.out.println("hello");
    }
}
"""
        fa = analyze_java_file_anemic(source, "Svc.java")
        # process and iterate have logic → 2 behavior methods
        assert fa.classes[0].behavior_method_count == 2

    def test_constructor_excluded_from_candidates(self):
        source = """
public class Foo {
    private int x;
    public Foo(int x) {
        this.x = x;
    }
    public int getX() {
        return this.x;
    }
}
"""
        fa = analyze_java_file_anemic(source, "Foo.java")
        # Constructor is excluded from candidate methods
        # getX is a getter → property_count
        assert fa.classes[0].method_count == 2  # constructor + getX
        assert fa.classes[0].property_count == 1  # getX


class TestJavaDerivedMetrics:
    def test_dbsi_data_heavy(self):
        """A DTO with many fields and no behavior has high DBSI."""
        source = """
public class UserDTO {
    private String name;
    private String email;
    private int age;
    public String getName() { return this.name; }
    public String getEmail() { return this.email; }
    public int getAge() { return this.age; }
}
"""
        fa = analyze_java_file_anemic(source, "UserDTO.java")
        cls = fa.classes[0]
        # fields=3, behavior=0, dbsi=3/(3+0)=1.0
        assert cls.dbsi == 1.0

    def test_dbsi_behavior_heavy(self):
        """A service with logic and no fields has low DBSI."""
        source = """
public class UserService {
    public void validate(Object user) {
        if (user == null) {
            throw new IllegalArgumentException();
        }
    }
    public void process(int[] items) {
        for (int item : items) {
            System.out.println(item);
        }
    }
}
"""
        fa = analyze_java_file_anemic(source, "UserService.java")
        cls = fa.classes[0]
        # fields=0, behavior=2, dbsi=0/(0+2)=0.0
        assert cls.dbsi == 0.0

    def test_orchestration_pressure(self):
        source = """
public class Svc {
    public void withLogic() {
        if (true) { }
    }
    public void noLogic() {
        System.out.println("hi");
    }
}
"""
        fa = analyze_java_file_anemic(source, "Svc.java")
        cls = fa.classes[0]
        # 2 candidate methods, 1 with logic → logic_density=0.5
        assert cls.logic_density == 0.5
        # orchestration_pressure = 1 - 0.5 = 0.5
        assert cls.orchestration_pressure == 0.5

    def test_ams_anemic_dto(self):
        """Anemic DTO should have AMS > 0.5."""
        source = """
public class UserDTO {
    private String name;
    private String email;
    private int age;
    public String getName() { return this.name; }
    public String getEmail() { return this.email; }
    public int getAge() { return this.age; }
    public void setName(String name) { this.name = name; }
}
"""
        fa = analyze_java_file_anemic(source, "UserDTO.java")
        cls = fa.classes[0]
        # DBSI=1.0, all methods are getters/setters → no candidates → orchestration=1.0
        # AMS = 1.0 * 1.0 = 1.0
        assert cls.ams > 0.5

    def test_ams_healthy_service(self):
        """Healthy service should have AMS <= 0.5."""
        source = """
public class UserService {
    public void validate(Object user) {
        if (user == null) {
            throw new IllegalArgumentException();
        }
    }
    public void process(int[] items) {
        for (int item : items) {
            System.out.println(item);
        }
    }
}
"""
        fa = analyze_java_file_anemic(source, "UserService.java")
        cls = fa.classes[0]
        assert cls.ams <= 0.5


class TestAnalyzeJavaFileAnemic:
    def test_single_class(self):
        source = """
public class Foo {
    private int x;
    public int getX() { return this.x; }
}
"""
        fa = analyze_java_file_anemic(source, "Foo.java")
        assert fa.class_count == 1
        assert fa.file_path == "Foo.java"

    def test_multiple_classes(self):
        source = """
public class DTO {
    private String name;
    public String getName() { return this.name; }
}

class Service {
    public void process(int[] data) {
        for (int x : data) {
            if (x > 0) { System.out.println(x); }
        }
    }
}
"""
        fa = analyze_java_file_anemic(source, "Multi.java")
        assert fa.class_count == 2

    def test_anemic_dto_detected(self):
        source = """
public class UserDTO {
    private String name;
    private String email;
    public String getName() { return this.name; }
    public void setName(String n) { this.name = n; }
    public String getEmail() { return this.email; }
    public void setEmail(String e) { this.email = e; }
}
"""
        fa = analyze_java_file_anemic(source, "UserDTO.java")
        assert fa.anemic_class_count == 1

    def test_healthy_service_not_flagged(self):
        source = """
public class OrderService {
    public void validateOrder(Object order) {
        if (order == null) {
            throw new IllegalArgumentException("null order");
        }
    }
    public void processOrder(Object order) {
        try {
            doProcess(order);
        } catch (Exception e) {
            handleError(e);
        }
    }
}
"""
        fa = analyze_java_file_anemic(source, "OrderService.java")
        assert fa.anemic_class_count == 0

    def test_worst_ams_tracked(self):
        source = """
public class DTO {
    private String name;
    public String getName() { return this.name; }
}

class Svc {
    public void run(int[] data) {
        for (int x : data) { System.out.println(x); }
    }
}
"""
        fa = analyze_java_file_anemic(source, "Mix.java")
        # DTO has higher AMS than Svc
        assert fa.worst_ams == fa.classes[0].ams

    def test_dunder_method_count_always_zero(self):
        source = """
public class Foo {
    public void doSomething() { }
}
"""
        fa = analyze_java_file_anemic(source, "Foo.java")
        assert fa.classes[0].dunder_method_count == 0


class TestJavaTouchCounts:
    def test_import_resolution(self):
        user_source = """
public class User {
    private String name;
}
"""
        service_source = """
import com.example.User;

public class UserService {
    public void process(User user) {
        if (user != null) {
            System.out.println(user);
        }
    }
}
"""
        sources = {
            "User.java": user_source,
            "UserService.java": service_source,
        }
        counts = compute_java_touch_counts(sources)
        assert counts["User.java"] == 1
        assert counts["UserService.java"] == 0

    def test_no_self_import(self):
        source = """
import com.example.Foo;

public class Foo {
    private int x;
}
"""
        counts = compute_java_touch_counts({"Foo.java": source})
        assert counts["Foo.java"] == 0

    def test_multiple_importers(self):
        model_source = "public class Model { private int id; }"
        svc_source = "import com.example.Model;\npublic class Service { }"
        ctrl_source = "import com.example.Model;\npublic class Controller { }"
        sources = {
            "Model.java": model_source,
            "Service.java": svc_source,
            "Controller.java": ctrl_source,
        }
        counts = compute_java_touch_counts(sources)
        assert counts["Model.java"] == 2
