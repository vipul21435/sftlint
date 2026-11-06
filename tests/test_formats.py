import json

import pytest

from sftlint.formats import FormatError, detect_file_format, detect_format, normalize


def test_detect_messages():
    rec = {"messages": [{"role": "user", "content": "hi"}]}
    assert detect_format(rec) == "messages"
    assert detect_format({"chat": [{"role": "user", "content": "hi"}]}) == "messages"


def test_detect_sharegpt_beats_messages_key():
    rec = {"conversations": [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "yo"}]}
    assert detect_format(rec) == "sharegpt"


def test_detect_alpaca_and_prompt_completion():
    assert detect_format({"instruction": "a", "input": "", "output": "b"}) == "alpaca"
    assert detect_format({"prompt": "a", "completion": "b"}) == "prompt-completion"


def test_detect_none():
    assert detect_format({"text": "x"}) is None
    assert detect_format([1, 2]) is None
    assert detect_format({"messages": []}) is None


def test_file_vote():
    rows = [{"prompt": "a", "completion": "b"}] * 2 + [{"instruction": "a", "output": "b"}]
    assert detect_file_format(rows) == "prompt-completion"
    assert detect_file_format([{"x": 1}]) is None


def test_normalize_messages_content_parts_and_tools():
    rec = {
        "messages": [
            {"role": "developer", "content": "be brief"},
            {"role": "user", "content": [{"type": "text", "text": "a"}, {"text": "b"}]},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        ],
        "tools": '[{"type": "function", "function": {"name": "f"}}]',
    }
    conv = normalize(rec, line=3)
    assert [m.role for m in conv.messages] == ["system", "user", "assistant", "tool"]
    assert conv.messages[1].content == "ab"
    assert conv.messages[2].content_missing
    assert conv.messages[2].tool_calls[0].id == "c1"
    assert conv.messages[2].tool_calls[0].name == "f"
    assert conv.messages[3].tool_call_id == "c1"
    assert conv.tools and conv.tools[0]["function"]["name"] == "f"
    assert conv.line == 3


def test_normalize_sharegpt_function_call_and_system():
    rec = {
        "system": "sys",
        "conversations": [
            {"from": "human", "value": "weather?"},
            {"from": "function_call", "value": '{"name": "w", "arguments": {"city": "X"}}'},
            {"from": "observation", "value": "sunny"},
            {"from": "gpt", "value": "It is sunny."},
            {"from": "narrator", "value": "?"},
        ],
    }
    conv = normalize(rec)
    assert [m.role for m in conv.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "narrator",
    ]
    assert conv.messages[2].tool_calls[0].name == "w"
    assert conv.messages[2].tool_calls[0].arguments == {"city": "X"}
    assert conv.messages[2].content == ""


def test_sharegpt_unparsed_function_call():
    rec = {"conversations": [{"from": "function_call", "value": "not json"}]}
    calls = normalize(rec).messages[0].tool_calls
    assert calls[0].parse_error.startswith("invalid JSON")


def test_normalize_alpaca_joins_input():
    conv = normalize({"instruction": "Sum", "input": "1 2", "output": "3"})
    assert conv.messages[0].content == "Sum\n\n1 2"
    assert conv.messages[1].role == "assistant"


def test_normalize_prompt_completion():
    conv = normalize({"prompt": "p", "completion": "c"})
    assert [m.content for m in conv.messages] == ["p", "c"]


def test_normalize_wrong_format():
    with pytest.raises(FormatError):
        normalize({"prompt": "p", "completion": "c"}, "alpaca")
    with pytest.raises(FormatError):
        normalize({"text": "x"})


def test_hermes_tags_become_calls_results_and_tools():
    tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]
    rec = {
        "conversations": [
            {"from": "system", "value": "Use tools.\n<tools>\n" + json.dumps(tools) + "\n</tools>"},
            {"from": "human", "value": "go"},
            {
                "from": "gpt",
                "value": '<tool_call>\n{"name": "f", "arguments": {"a": 1}}\n</tool_call>\n'
                "<tool_call>\n{bad json}\n</tool_call>",
            },
            {
                "from": "tool",
                "value": '<tool_response>\n{"name": "f", "content": 2}\n</tool_response>',
            },
            {"from": "gpt", "value": "Done."},
        ]
    }
    conv = normalize(rec)
    call_turn = conv.messages[2]
    assert call_turn.content == ""
    assert [c.name for c in call_turn.tool_calls] == ["f", None]
    assert call_turn.tool_calls[0].arguments == {"a": 1}
    assert call_turn.tool_calls[1].parse_error
    assert [r.name for r in conv.messages[3].results()] == ["f"]
    assert conv.tools == tools


def test_unclosed_tool_call_runs_to_end():
    from sftlint.formats import parse_tagged_calls, parse_tagged_results

    calls, rest = parse_tagged_calls('Sure. <tool_call>{"name": "f", "argu')
    assert rest == "Sure."
    assert calls[0].parse_error
    assert parse_tagged_calls("plain text") == ([], "plain text")
    bad = parse_tagged_results("<tool_response>nope</tool_response><tool_response>[1]")
    assert bad[0].parse_error and bad[1].name is None


def test_openai_tool_message_result_from_id():
    rec = {
        "messages": [
            {"role": "user", "content": "x"},
            {"role": "tool", "tool_call_id": "c9", "name": "g", "content": "r"},
        ],
        "tools": "not json",
    }
    conv = normalize(rec)
    assert conv.tools is None
    res = conv.messages[1].results()
    assert (res[0].id, res[0].name) == ("c9", "g")
