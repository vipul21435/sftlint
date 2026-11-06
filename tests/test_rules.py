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


@pytest.mark.parametrize(
    "text",
    [
        "| n | square |\n|---|---|\n| 1 | 1 |\n| 2 | 4 |",
        "The language is C++",
        "See https://docs.python.org/3/",
        "Docs: <https://example.org/a/>",
    ],
)
def test_truncation_ignores_table_rows_cpp_and_urls(text):
    assert truncation_reason(text) is None


def test_truncation_still_catches_a_dangling_article():
    assert truncation_reason("He picked up a") == "ends with the word 'a'"
    assert truncation_reason("a") == "ends with the word 'a'"
    assert truncation_reason("one|") == "ends with '|'"
    assert truncation_reason("x = a +") == "ends with '+'"
    assert truncation_reason("divide by /") == "ends with '/'"
    assert truncation_reason("Sources: https://example.org/a,") == "ends with ','"


@pytest.mark.parametrize("text", ["The correct answer is A", "Answer: A", "Vitamin A", "(A)"])
def test_single_capital_letter_answer_is_not_truncated(text):
    assert truncation_reason(text) is None


def test_lower_case_article_at_end_is_still_truncated():
    assert truncation_reason("The answer is a") == "ends with the word 'a'"
    assert truncation_reason("It depends on The") == "ends with the word 'The'"


def test_empty_tool_result_is_not_an_empty_turn():
    from sftlint.formats import normalize
    from sftlint.rules import check_empty_turn

    call = {"id": "c1", "function": {"name": "reset", "arguments": "{}"}}
    rec = {
        "messages": [
            {"role": "user", "content": "Reset it."},
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "c1", "content": ""},
            {"role": "assistant", "content": ""},
        ]
    }
    assert [f.turn for f in check_empty_turn(normalize(rec))] == [3]


@pytest.mark.parametrize(
    "text",
    [
        "```python\nx = 1\n````",  # opened with three, closed with four
        "````python\nx = 1\n````",
        "````md\n```\n````",  # a lone ``` line shown inside a four-backtick block
        "````md\n```python\nx = 1\n```\n````",
        "Use ```x``` inline.",
    ],
)
def test_fences_pair_by_line_and_length(text):
    assert truncation_reason(text) is None


@pytest.mark.parametrize(
    "text",
    [
        "````python\nx = 1\n```",  # a shorter line does not close a longer fence
        "```py\na\n```\nThen:\n```py\nb = 2",  # one closed block, then an open one
        "```py\na\n```\n```py\nb\n```\n```py\nc = 3",
        "```py\nx = 1\n``` trailing",  # a line with text after the ticks does not close
    ],
)
def test_unclosed_fence_is_flagged(text):
    assert truncation_reason(text) == "unclosed code fence"


def test_image_only_turn_is_not_empty(write_jsonl):
    from sftlint.cli import main

    image = {"type": "image_url", "image_url": {"url": "https://example.com/dog.png"}}
    c = conv({"role": "user", "content": [image]}, ("assistant", "A dog."))
    assert c.messages[0].has_media and run("empty-turn", c) == []
    blank = conv({"role": "user", "content": [{"type": "text", "text": " "}]}, ("assistant", "x"))
    assert [f.turn for f in run("empty-turn", blank)] == [0]
    path = write_jsonl(
        [
            {
                "messages": [
                    {"role": "user", "content": [image]},
                    {"role": "assistant", "content": "A dog."},
                ]
            }
        ]
    )
    assert main(["check", str(path)]) == 0


def test_role_order_after_a_leading_system_turn():
    sys_turn = ("system", "Be brief.")
    two_users = run("role-order", conv(sys_turn, ("user", "a"), ("user", "b"), ("assistant", "c")))
    assert [(f.turn, f.message) for f in two_users] == [(2, "two user turns in a row")]
    stray_tool = run("role-order", conv(sys_turn, ("user", "a"), ("tool", "r"), ("assistant", "c")))
    assert [(f.turn, f.message) for f in stray_tool] == [
        (2, "tool turn does not follow an assistant tool call")
    ]


def test_tool_turn_after_an_assistant_turn_without_calls():
    out = run(
        "role-order", conv(("user", "a"), ("assistant", "b"), ("tool", "r"), ("assistant", "c"))
    )
    assert [(f.turn, f.message) for f in out] == [
        (2, "tool turn does not follow an assistant tool call")
    ]


def test_trailing_and_is_truncated():
    assert truncation_reason("Add the salt and") == "ends with the word 'and'"


def test_duplicate_turn_after_an_empty_turn():
    c = conv(("user", ""), ("assistant", "x"), ("user", "q"), ("assistant", "y"), ("user", "q"))
    assert [f.turn for f in run("duplicate-turn", c)] == [4]
