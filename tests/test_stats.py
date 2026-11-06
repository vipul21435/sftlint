import math
import random

import pytest

from sftlint.dataset_rules import check_duplicate_prompt, check_length_outlier, prompt_key
from sftlint.formats import normalize
from sftlint.stats import length_stats, modified_z, quantile, split_fields, summary


def pc(prompt, completion, line=1):
    return normalize({"prompt": prompt, "completion": completion}, line=line)


def test_quantile_matches_linear_interpolation():
    s = [1.0, 2.0, 3.0, 4.0]
    assert quantile(s, 0.5) == 2.5
    assert quantile(s, 0.0) == 1.0 and quantile(s, 1.0) == 4.0
    assert math.isnan(quantile([], 0.5))


def test_summary():
    out = summary([3, 1, 2])
    assert out["count"] == 3 and out["p50"] == 2 and out["mean"] == 2
    assert summary([]) == {"count": 0}


def test_modified_z_constant_and_scale():
    assert modified_z([5, 5, 5]) == [0.0, 0.0, 0.0]
    assert modified_z([]) == []
    z = modified_z([1, 2, 3, 4, 100])
    assert z[2] == 0 and z[-1] > 3.5


def test_modified_z_is_shift_and_scale_invariant():
    rng = random.Random(0)
    xs = [rng.gauss(0, 1) for _ in range(200)]
    a = modified_z(xs)
    b = modified_z([3 * x + 10 for x in xs])
    assert max(abs(p - q) for p, q in zip(a, b, strict=True)) < 1e-9


def test_false_positive_rate_on_normal_data_is_low():
    rng = random.Random(1)
    xs = [rng.gauss(0, 1) for _ in range(20000)]
    rate = sum(abs(z) > 3.5 for z in modified_z(xs)) / len(xs)
    assert rate < 0.002


def test_split_fields_without_final_assistant():
    c = normalize({"messages": [{"role": "user", "content": "a"}]})
    assert split_fields(c) == {"prompt": "a", "response": ""}


def test_length_stats_shape():
    st = length_stats([pc("one two", "three"), pc("x", "y z w")])
    assert st["prompt"]["words"]["max"] == 2
    assert st["response"]["chars"]["count"] == 2


def test_duplicate_prompt_exact_after_whitespace():
    convs = [pc("Hello  world", "a", 1), pc("Hello world", "b", 2), pc("hello world", "c", 3)]
    out = check_duplicate_prompt(convs)
    assert [(f.line, f.message) for f in out] == [(2, "same prompt as line 1")]


def test_prompt_key_includes_roles():
    a = normalize({"messages": [{"role": "user", "content": "x"}]})
    b = normalize({"messages": [{"role": "system", "content": "x"}]})
    assert prompt_key(a) != prompt_key(b)


@pytest.mark.parametrize("n", [5, 19])
def test_length_outlier_skips_small_sets(n):
    convs = [pc("p" * 10, "r", i) for i in range(n)] + [pc("p" * 10000, "r", 99)]
    assert check_length_outlier(convs) == []


def test_length_outlier_flags_long_and_short():
    rng = random.Random(2)
    convs = [pc("p" * rng.randint(80, 120), "r" * rng.randint(80, 120), i) for i in range(50)]
    convs.append(pc("p" * 20000, "r" * 100, 100))
    convs.append(pc("p" * 100, "r", 101))
    out = check_length_outlier(convs)
    got = {(f.line, f.message.split(":")[0]) for f in out}
    assert got == {(100, "prompt is long"), (101, "response is short")}
