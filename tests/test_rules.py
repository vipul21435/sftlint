import pytest

from sftlint.formats import normalize
from sftlint.rules import CHECKS, RULES, truncation_reason


def conv(*turns, **extra):
    msgs = []
    for t in turns:
        if isinstance(t, dict):
            msgs.append(t)
        else:
            role, content = t
            msgs.append({"role": role, "content": content})
    return normalize({"messages": msgs, **extra}, line=7)


def run(rule_id, c):
    return CHECKS[rule_id](c)


def test_registry_has_severity_and_summary():
    for rid, info in RULES.items():
        assert info.id == rid
        assert info.severity in {"error", "warning", "info"}
        assert info.summary


def test_clean_conversation_has_no_findings():
    c = conv(
        ("system", "s"),
        ("user", "Hi."),
        ("assistant", "Hello."),
        ("user", "Bye."),
        ("assistant", "Goodbye."),
    )
    for rid, fn in CHECKS.items():
        assert fn(c) == [], rid


def test_role_unknown():
    out = run("role-unknown", conv(("user", "a"), ("narrator", "b")))
    assert len(out) == 1 and out[0].turn == 1 and out[0].line == 7


def test_system_position():
    out = run("system-position", conv(("user", "a"), ("system", "b"), ("assistant", "c")))
    assert [f.turn for f in out] == [1]


@pytest.mark.parametrize(
    "turns,expected",
    [
        ([("assistant", "a")], "first turn is assistant"),
        ([("user", "a"), ("user", "b")], "two user turns"),
        ([("user", "a"), ("assistant", "b"), ("assistant", "c")], "two assistant turns"),
        ([("user", "a"), ("tool", "b")], "tool turn does not follow"),
    ],
)
def test_role_order_violations(turns, expected):
    out = run("role-order", conv(*turns))
    assert len(out) == 1 and expected in out[0].message


def test_role_order_tool_flow_ok_and_user_after_tool():
    call = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "1", "function": {"name": "f", "arguments": "{}"}}],
    }
    ok = conv(("user", "q"), call, ("tool", "r1"), ("tool", "r2"), ("assistant", "done."))
    assert run("role-order", ok) == []
    bad = conv(("user", "q"), call, ("tool", "r1"), ("user", "again"))
    assert "follows a tool result" in run("role-order", bad)[0].message
    # an assistant turn with tool calls may be followed by another assistant turn
    two = conv(("user", "q"), call, ("assistant", "done."))
    assert run("role-order", two) == []


def test_empty_turn_missing_vs_empty():
    out = run("empty-turn", conv(("user", "  "), {"role": "assistant", "content": None}))
    assert [f.message for f in out] == [
        "user turn content is empty",
        "assistant turn content is missing",
    ]


def test_duplicate_turn_whitespace_insensitive():
    c = conv(
        ("user", "same  text"), ("assistant", "x."), ("user", "same text"), ("assistant", "y.")
    )
    out = run("duplicate-turn", c)
    assert len(out) == 1 and out[0].turn == 2 and "turn 0" in out[0].message


def test_final_role():
    assert run("final-role", conv(("user", "a")))[0].turn == 0
    assert run("final-role", conv(("user", "a"), ("assistant", "b."))) == []


@pytest.mark.parametrize(
    "text,flagged",
    [
        ("The answer is 4.", False),
        ("Done :)", False),
        ("```python\nprint(1)\n```", False),
        ("```python\nprint(1", True),
        ("First, we compute", False),
        ("and then we add the", True),
        ("Here is the list:", True),
        ("x = foo(", True),
        ("", False),
    ],
)
def test_truncation_reason(text, flagged):
    assert (truncation_reason(text) is not None) is flagged


def test_truncated_final_only_checks_last_assistant():
    assert run("truncated-final", conv(("user", "list:"))) == []
    out = run("truncated-final", conv(("user", "q"), ("assistant", "Steps are:")))
    assert out[0].turn == 1


def test_truncated_final_reports_missing_end_token():
    c = normalize({"text": "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nHello there."})
    out = CHECKS["truncated-final"](c)
    assert [f.message for f in out] == ["no end-of-turn token after the final turn"]
    ok = normalize(
        {"text": "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nHello.<|im_end|>"}
    )
    assert CHECKS["truncated-final"](ok) == []


@pytest.mark.parametrize(
    "text",
    ["Matches: i-b, ii-d, iii-c, iv-a", "first line|\n\nsecond line||"],
)
def test_truncation_ignores_answer_keys_and_verse_endings(text):
    assert truncation_reason(text) is None


def test_truncation_still_catches_a_dangling_article():
    assert truncation_reason("He picked up a") == "ends with the word 'a'"
    assert truncation_reason("a") == "ends with the word 'a'"
    assert truncation_reason("one|") == "ends with '|'"
