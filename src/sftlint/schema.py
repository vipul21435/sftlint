"""A small JSON Schema validator for tool-call arguments.

Function declarations in the OpenAI and Hugging Face tool formats describe their
parameters with JSON Schema. Tool-call data mostly uses a narrow part of it, so this
module implements that part and no more, with no dependency:

``type`` (a name or a list of names), ``enum``, ``const``, ``properties``,
``required``, ``additionalProperties`` (``false`` or a schema), ``items`` (one
schema), ``minItems``, ``maxItems``, ``minLength``, ``maxLength``, ``pattern``,
``minimum``, ``maximum``, ``exclusiveMinimum``, ``exclusiveMaximum`` (numbers),
``anyOf``, ``oneOf``, ``allOf`` and ``nullable``.

Every other keyword is ignored, which makes the validator lenient rather than strict:
``$ref``, ``format``, ``patternProperties``, ``dependentRequired`` and conditionals
never cause an error. Following the JSON Schema validation vocabulary, ``integer``
accepts whole-valued numbers, and booleans are not numbers.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

_TYPES = {
    "string": lambda v: isinstance(v, str),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
    "object": lambda v: isinstance(v, Mapping),
    "array": lambda v: isinstance(v, list),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "integer": lambda v: (
        (isinstance(v, int) and not isinstance(v, bool))
        or (isinstance(v, float) and math.isfinite(v) and v.is_integer())
    ),
}


def _json_equal(a: Any, b: Any) -> bool:
    """Equality as JSON Schema defines it for ``enum`` and ``const``: values of different
    JSON types are never equal (``true`` is not ``1``), while ``1`` equals ``1.0``.
    Arrays and objects compare element by element."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if _TYPES["number"](a) or _TYPES["number"](b):
        return _TYPES["number"](a) and _TYPES["number"](b) and a == b
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_json_equal(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return a.keys() == b.keys() and all(_json_equal(a[k], b[k]) for k in a)
    return type(a) is type(b) and a == b


def _type_name(value: Any) -> str:
    for name in ("null", "boolean", "integer", "number", "string", "array", "object"):
        if _TYPES[name](value):
            return name
    return type(value).__name__


def _short(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= 40 else text[:37] + "..."


def validate(value: Any, schema: Any, path: str = "$", depth: int = 0) -> list[str]:
    """Return a list of error messages, each starting with a ``$.a.b[0]`` path.

    An empty list means ``value`` is valid against the supported keywords. A schema
    that is not an object (``true``, ``{}``, or garbage) accepts everything.
    """
    if not isinstance(schema, Mapping) or depth > 32:
        return []
    if value is None and schema.get("nullable") is True:
        return []
    errors: list[str] = []

    declared = schema.get("type")
    if declared is not None:
        names = declared if isinstance(declared, list) else [declared]
        known = [n for n in names if n in _TYPES]
        if known and not any(_TYPES[n](value) for n in known):
            want = " or ".join(known)
            return [f"{path}: expected {want}, got {_type_name(value)}"]

    enum = schema.get("enum")
    if isinstance(enum, list) and not any(_json_equal(value, e) for e in enum):
        errors.append(f"{path}: {_short(value)} is not one of {_short(enum)}")
    if "const" in schema and not _json_equal(value, schema["const"]):
        errors.append(f"{path}: expected {_short(schema['const'])}")

    if isinstance(value, Mapping):
        errors += _validate_object(value, schema, path, depth)
    elif isinstance(value, list):
        errors += _validate_array(value, schema, path, depth)
    elif isinstance(value, str):
        errors += _validate_string(value, schema, path)
    elif _TYPES["number"](value):
        errors += _validate_number(value, schema, path)

    for key in ("allOf", "anyOf", "oneOf"):
        subs = schema.get(key)
        if not isinstance(subs, list) or not subs:
            continue
        results = [validate(value, s, path, depth + 1) for s in subs]
        passed = sum(not r for r in results)
        if key == "allOf":
            errors += [e for r in results for e in r]
        elif key == "anyOf" and passed == 0:
            errors.append(f"{path}: matches none of anyOf")
        elif key == "oneOf" and passed != 1:
            errors.append(f"{path}: matches {passed} of oneOf, expected exactly 1")
    return errors


def _validate_object(value: Mapping[str, Any], schema: Mapping[str, Any], path: str, depth: int):
    errors = []
    props = schema.get("properties")
    props = props if isinstance(props, Mapping) else {}
    required = schema.get("required")
    if isinstance(required, list):
        errors += [
            f"{path}: missing required property {k!r}"
            for k in required
            if isinstance(k, str) and k not in value
        ]
    extra = schema.get("additionalProperties", True)
    for key, item in value.items():
        if key in props:
            errors += validate(item, props[key], f"{path}.{key}", depth + 1)
        elif extra is False:
            errors.append(f"{path}: unexpected property {key!r}")
        elif isinstance(extra, Mapping):
            errors += validate(item, extra, f"{path}.{key}", depth + 1)
    return errors


def _validate_array(value: list[Any], schema: Mapping[str, Any], path: str, depth: int):
    errors = []
    if isinstance(schema.get("minItems"), int) and len(value) < schema["minItems"]:
        errors.append(f"{path}: {len(value)} items, fewer than minItems {schema['minItems']}")
    if isinstance(schema.get("maxItems"), int) and len(value) > schema["maxItems"]:
        errors.append(f"{path}: {len(value)} items, more than maxItems {schema['maxItems']}")
    items = schema.get("items")
    if isinstance(items, Mapping):
        for i, item in enumerate(value):
            errors += validate(item, items, f"{path}[{i}]", depth + 1)
    return errors


def _validate_string(value: str, schema: Mapping[str, Any], path: str):
    errors = []
    if isinstance(schema.get("minLength"), int) and len(value) < schema["minLength"]:
        errors.append(f"{path}: shorter than minLength {schema['minLength']}")
    if isinstance(schema.get("maxLength"), int) and len(value) > schema["maxLength"]:
        errors.append(f"{path}: longer than maxLength {schema['maxLength']}")
    pattern = schema.get("pattern")
    if isinstance(pattern, str):
        try:
            if re.search(pattern, value) is None:
                errors.append(f"{path}: does not match pattern {pattern!r}")
        except re.error:
            pass  # a pattern Python cannot compile is ignored, not reported
    return errors


def _validate_number(value: float, schema: Mapping[str, Any], path: str):
    errors = []
    checks = (
        ("minimum", lambda b: value < b, "below minimum"),
        ("maximum", lambda b: value > b, "above maximum"),
        ("exclusiveMinimum", lambda b: value <= b, "not above exclusiveMinimum"),
        ("exclusiveMaximum", lambda b: value >= b, "not below exclusiveMaximum"),
    )
    for key, fails, label in checks:
        bound = schema.get(key)
        if _TYPES["number"](bound) and fails(bound):
            errors.append(f"{path}: {value} is {label} {bound}")
    return errors
