import random

from sftlint.schema import _type_name, validate

WEATHER = {
    "type": "object",
    "properties": {
        "city": {"type": "string", "minLength": 1},
        "unit": {"type": "string", "enum": ["c", "f"]},
        "days": {"type": "integer", "minimum": 1, "maximum": 14},
        "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 2},
    },
    "required": ["city"],
    "additionalProperties": False,
}


def test_valid_arguments():
    assert validate({"city": "Pune", "unit": "c", "days": 3, "tags": ["a"]}, WEATHER) == []
    assert validate({"city": "Pune", "days": 3.0}, WEATHER) == []


def test_each_keyword_reports_with_path():
    errs = validate(
        {"unit": "k", "days": 20, "tags": ["a", 1, "b"], "extra": True, "city": ""}, WEATHER
    )
    joined = "\n".join(errs)
    assert "$.unit: 'k' is not one of" in joined
    assert "$.days: 20 is above maximum 14" in joined
    assert "$.tags[1]: expected string, got integer" in joined
    assert "$.tags: 3 items, more than maxItems 2" in joined
    assert "$: unexpected property 'extra'" in joined
    assert "$.city: shorter than minLength 1" in joined
    assert validate({}, WEATHER) == ["$: missing required property 'city'"]


def test_type_rules():
    assert validate(True, {"type": "integer"}) == ["$: expected integer, got boolean"]
    assert validate(True, {"type": "number"})
    assert validate(2.5, {"type": "integer"})
    assert validate(2, {"type": "number"}) == []
    assert validate(None, {"type": ["string", "null"]}) == []
    assert validate(None, {"type": "string", "nullable": True}) == []
    assert validate("x", {"type": "mystery"}) == []


def test_combinators_and_bounds():
    s = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
    assert validate("a", s) == [] and validate(1, s) == []
    assert validate(1.5, s) == ["$: matches none of anyOf"]
    one = {"oneOf": [{"type": "number"}, {"type": "integer"}]}
    assert validate(3, one) == ["$: matches 2 of oneOf, expected exactly 1"]
    assert validate(5, {"allOf": [{"minimum": 6}, {"exclusiveMaximum": 5}]}) == [
        "$: 5 is below minimum 6",
        "$: 5 is not below exclusiveMaximum 5",
    ]
    assert validate(1, {"exclusiveMinimum": 1})
    assert validate([], {"minItems": 1})
    assert validate("abcd", {"maxLength": 3})
    assert validate("ab", {"pattern": "^a"}) == []
    assert validate("ba", {"pattern": "^a"})
    assert validate("ba", {"pattern": "("}) == []
    assert validate(3, {"const": 4}) == ["$: expected 4"]
    extra = {"type": "object", "additionalProperties": {"type": "integer"}}
    assert validate({"a": "x"}, extra) == ["$.a: expected integer, got string"]


def test_non_schema_accepts_everything():
    for schema in (True, None, {}, "x", {"$ref": "#/defs/a"}):
        assert validate({"a": [1, {"b": None}]}, schema) == []


def _random_value(rng, depth=0):
    kind = rng.choice(
        ["null", "bool", "int", "float", "str", "list", "dict"][: 7 if depth < 3 else 5]
    )
    if kind == "null":
        return None
    if kind == "bool":
        return rng.random() < 0.5
    if kind == "int":
        return rng.randint(-100, 100)
    if kind == "float":
        return rng.uniform(-10, 10)
    if kind == "str":
        return "".join(rng.choice("ab1 ") for _ in range(rng.randint(0, 5)))
    if kind == "list":
        return [_random_value(rng, depth + 1) for _ in range(rng.randint(0, 3))]
    return {f"k{i}": _random_value(rng, depth + 1) for i in range(rng.randint(0, 3))}


def test_property_own_type_and_enum_always_validate():
    rng = random.Random(0)
    for _ in range(2000):
        v = _random_value(rng)
        assert validate(v, {"type": _type_name(v)}) == []
        assert validate(v, {"enum": [0, v]}) == []
        assert validate(v, {"anyOf": [{"type": _type_name(v)}, {"type": "null"}]}) == []


def test_enum_and_const_compare_json_types():
    assert validate(True, {"enum": [1]}) != []
    assert validate(1, {"const": True}) != []
    assert validate(False, {"enum": [0, None]}) != []
    assert validate(1, {"enum": [1.0]}) == [] and validate(1.0, {"const": 1}) == []
    assert validate(True, {"enum": [True]}) == []
    assert validate([1, True], {"const": [1.0, True]}) == []
    assert validate([True], {"const": [1]}) != []
    assert validate({"a": 0}, {"enum": [{"a": False}]}) != []
    assert validate({"a": 0}, {"enum": [{"a": 0.0}]}) == []
    assert validate(None, {"enum": [None]}) == [] and validate("x", {"const": "x"}) == []


def test_null_fails_a_typed_schema_unless_nullable():
    assert validate(None, {"type": "string"}) == ["$: expected string, got null"]
    assert validate(None, {"type": "string", "nullable": True}) == []
    assert validate(5, {"type": "string", "nullable": True}) == ["$: expected string, got integer"]
    assert validate(None, {"type": "string", "nullable": False}) != []


def test_bounds_are_inclusive():
    cases = [
        ({"type": "array", "minItems": 2}, [1, 2], [1]),
        ({"type": "array", "maxItems": 2}, [1, 2], [1, 2, 3]),
        ({"type": "string", "minLength": 2}, "ab", "a"),
        ({"type": "string", "maxLength": 2}, "ab", "abc"),
        ({"type": "number", "minimum": 2}, 2, 1.5),
        ({"type": "number", "maximum": 2}, 2, 2.5),
    ]
    for schema, at_bound, past_bound in cases:
        assert validate(at_bound, schema) == [], schema
        assert len(validate(past_bound, schema)) == 1, schema


def test_enum_of_arrays_against_a_scalar():
    assert validate(1, {"enum": [[1, 2]]}) == ["$: 1 is not one of [[1, 2]]"]
    assert validate([1, 2], {"enum": [[1, 2]]}) == []
    assert validate("ab", {"enum": [["a", "b"]]}) == ["$: 'ab' is not one of [['a', 'b']]"]
    assert validate(["a", "b"], {"enum": ["ab"]}) != []
