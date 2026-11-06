import math
import random

import pytest

from sftlint.dataset_rules import check_duplicate_prompt, check_length_outlier, prompt_key
from sftlint.formats import normalize
from sftlint.stats import (
    length_stats,
    modified_z,
    quantile,
    split_fields,
    summary,
    turn_text,
)
from sftlint.tokens import get_counter, pretokens


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


@pytest.mark.parametrize("n", [5, 18])
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


def test_length_stats_has_tokens():
    st = length_stats([pc("Hello, world!", "ok")])
    assert st["prompt"]["tokens"]["max"] == 4
    assert "tokens" not in length_stats([pc("a", "b")], None)["prompt"]


def test_pretokens_follow_gpt2_pieces():
    assert pretokens("") == 0
    assert pretokens("Hello world") == 2
    assert pretokens("it's 2026!") == 4  # "it", "'s", " 2026", "!"
    assert pretokens("a_b") == 3
    assert pretokens("x    y") == 3


def test_pretokens_never_exceed_chars():
    import random

    rng = random.Random(3)
    for _ in range(500):
        s = "".join(rng.choice("ab 1,_\n'é") for _ in range(rng.randint(0, 40)))
        assert pretokens(s) <= len(s)


def test_get_counter():
    assert get_counter() is pretokens
    with pytest.raises(ValueError, match="unknown tokenizer"):
        get_counter("bpe")


def test_tiktoken_counter_missing_or_present(monkeypatch):
    import builtins

    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "tiktoken":
            raise ImportError
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(ValueError, match="pip install tiktoken"):
        get_counter("tiktoken:cl100k_base")


def test_tiktoken_counter_uses_encoding(monkeypatch):
    import sys
    import types

    class Enc:
        def encode(self, text, disallowed_special=()):
            return text.split()

    mod = types.SimpleNamespace(get_encoding=lambda name: Enc())
    monkeypatch.setitem(sys.modules, "tiktoken", types.ModuleType("tiktoken"))
    monkeypatch.setattr(sys.modules["tiktoken"], "get_encoding", mod.get_encoding, raising=False)
    assert get_counter("tiktoken:any")("a b c") == 3


def test_turn_text_includes_tool_calls():
    c = normalize(
        {
            "messages": [
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "content": "Calling.",
                    "tool_calls": [{"function": {"name": "f", "arguments": "{}"}}],
                },
            ]
        }
    )
    assert turn_text(c.messages[1]) == 'Calling.\n{"name": "f", "arguments": "{}"}'
    assert split_fields(c)["response"].startswith("Calling.")
    c.messages[1].content = ""
    assert turn_text(c.messages[1]) == '{"name": "f", "arguments": "{}"}'


def _tool_record(tools, call_args=None):
    msgs = [{"role": "user", "content": "Book me a table for two tonight."}]
    if call_args is not None:
        call = {"id": "c1", "function": {"name": "book_table", "arguments": call_args}}
        msgs += [
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        ]
    msgs.append({"role": "assistant", "content": "Done."})
    return {"messages": msgs, "tools": [{"name": t, "parameters": {}} for t in tools]}


def test_duplicate_prompt_sees_declared_tools_and_call_arguments():
    convs = [
        normalize(_tool_record(["book_table"]), line=1),
        normalize(_tool_record(["search_places"]), line=2),
        normalize(_tool_record(["book_table"], '{"city": "Oslo"}'), line=3),
        normalize(_tool_record(["book_table"], '{"city": "Bergen"}'), line=4),
        normalize(_tool_record(["book_table"], '{"city": "Bergen"}'), line=5),
    ]
    out = check_duplicate_prompt(convs)
    assert [(f.line, f.message) for f in out] == [(5, "same prompt as line 4")]


def test_modified_z_falls_back_to_mean_absolute_deviation_when_mad_is_zero():
    xs = [2.0] * 30 + [12.0]
    z = modified_z(xs)
    assert z[:30] == [0.0] * 30
    mean_ad = 10.0 / 31
    assert z[30] == pytest.approx(10.0 / (1.253314 * mean_ad))


def test_length_outlier_found_when_most_lengths_are_equal():
    convs = [pc(f"Review {i}: fine product.", "positive", i + 1) for i in range(30)]
    convs.append(pc("Review 30: fine product.", "positive " * 20000, 31))
    out = check_length_outlier(convs)
    assert [(f.line, f.message.split(":")[0]) for f in out] == [(31, "response is long")]
