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
    assert conv.messages[2].tool_calls[0]["id"] == "c1"
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
        "system", "user", "assistant", "tool", "assistant", "narrator",
    ]
    assert conv.messages[2].tool_calls[0]["function"]["name"] == "w"
    assert conv.messages[2].content == ""


def test_sharegpt_unparsed_function_call():
    rec = {"conversations": [{"from": "function_call", "value": "not json"}]}
    calls = normalize(rec).messages[0].tool_calls
    assert calls[0]["_unparsed"] is True


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
