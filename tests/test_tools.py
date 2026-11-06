import json

from sftlint.formats import ToolCall, normalize
from sftlint.tools import (
    check_tool_args_json,
    check_tool_args_schema,
    check_tool_result_match,
    check_tool_unknown,
    declared_tools,
    parse_arguments,
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                "required": ["city"],
            },
        },
    },
    {"name": "clock"},
]


def call(cid, name, args):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": args}}


def conv(*msgs, tools=TOOLS):
    rec = {"messages": list(msgs)}
    if tools is not None:
        rec["tools"] = tools
    return normalize(rec, line=4)


U = {"role": "user", "content": "q"}
A = {"role": "assistant", "content": "a"}


def asst(*calls):
    return {"role": "assistant", "content": None, "tool_calls": list(calls)}


def tool(cid, content="r", name=None):
    m = {"role": "tool", "tool_call_id": cid, "content": content}
    if name:
        m["name"] = name
    return m


def test_declared_tools_both_shapes():
    c = conv(U, A)
    assert set(declared_tools(c)) == {"weather", "clock"}
    assert declared_tools(c)["clock"] is None
    assert declared_tools(conv(U, A, tools=None)) is None


def test_parse_arguments():
    assert parse_arguments(ToolCall("f", '{"a": 1}')) == ({"a": 1}, None)
    assert parse_arguments(ToolCall("f", None)) == ({}, None)
    assert parse_arguments(ToolCall("f", "  ")) == ({}, None)
    assert parse_arguments(ToolCall("f", '{"a": '))[1].startswith("arguments are not valid JSON")
    assert "expected an object" in parse_arguments(ToolCall("f", "[1]"))[1]
    assert parse_arguments(ToolCall(None, "x", parse_error="bad"))[1] == "bad"


def test_clean_tool_conversation():
    c = conv(
        U,
        asst(call("c1", "weather", '{"city": "Pune"}'), call("c2", "clock", "{}")),
        tool("c2"),
        tool("c1"),
        A,
    )
    for fn in (
        check_tool_args_json,
        check_tool_unknown,
        check_tool_args_schema,
        check_tool_result_match,
    ):
        assert fn(c) == [], fn.__name__


def test_args_json_and_schema():
    c = conv(U, asst(call("c1", "weather", '{"city": '), call("c2", "weather", '{"days": "x"}')))
    j = check_tool_args_json(c)
    assert len(j) == 1 and j[0].turn == 1 and "'weather'" in j[0].message
    s = check_tool_args_schema(c)
    assert len(s) == 1
    assert "missing required property 'city'" in s[0].message
    assert "$.days: expected integer" in s[0].message


def test_schema_message_caps_errors():
    tools = [{"name": "f", "parameters": {"required": ["a", "b", "c", "d", "e"]}}]
    c = conv(U, asst(call("c1", "f", "{}")), tools=tools)
    assert "(+2 more)" in check_tool_args_schema(c)[0].message


def test_unknown_tool():
    c = conv(U, asst(call("c1", "search", "{}"), call("c2", None, "{}")))
    msgs = [f.message for f in check_tool_unknown(c)]
    assert msgs == ["'search' is not a declared tool", "tool call has no function name"]
    assert check_tool_unknown(conv(U, asst(call("c1", "search", "{}")), tools=None)) == []


def test_result_matching_by_id():
    c = conv(U, asst(call("c1", "weather", '{"city": "a"}')), tool("c9"), A)
    msgs = [(f.turn, f.message) for f in check_tool_result_match(c)]
    assert msgs == [
        (2, "result id 'c9' answers no pending call"),
        (1, "call 'weather' gets no result"),
    ]


def test_call_at_end_is_a_target_not_a_missing_result():
    c = conv(U, asst(call("c1", "weather", '{"city": "a"}')))
    assert check_tool_result_match(c) == []
    c2 = conv(U, asst(call("c1", "weather", '{"city": "a"}')), U, A)
    assert [f.message for f in check_tool_result_match(c2)] == ["call 'weather' gets no result"]


def test_hermes_results_match_by_name():
    tools = json.dumps(TOOLS[:1])
    rec = {
        "conversations": [
            {"from": "system", "value": f"<tools>{tools}</tools>"},
            {"from": "human", "value": "q"},
            {
                "from": "gpt",
                "value": '<tool_call>{"name": "weather", "arguments": {"city": "a"}}</tool_call>'
                '<tool_call>{"name": "weather", "arguments": {"city": "b"}}</tool_call>',
            },
            {
                "from": "tool",
                "value": '<tool_response>{"name": "weather", "content": 1}</tool_response>'
                '<tool_response>{"name": "clock", "content": 2}</tool_response>'
                "<tool_response>oops</tool_response>",
            },
            {"from": "gpt", "value": "done"},
        ]
    }
    c = normalize(rec)
    msgs = [f.message for f in check_tool_result_match(c)]
    assert msgs[0] == "result 'clock' answers no pending call"
    assert msgs[1].startswith("tool result: invalid JSON")
    assert msgs[2] == "call 'weather' gets no result"
    assert check_tool_unknown(c) == [] and check_tool_args_schema(c) == []


def test_unnamed_result_matches_by_position():
    rec = {
        "conversations": [
            {"from": "human", "value": "q"},
            {"from": "function_call", "value": '{"name": "f", "arguments": {}}'},
            {"from": "observation", "value": "r1"},
            {"from": "observation", "value": "r2"},
            {"from": "gpt", "value": "done"},
        ]
    }
    msgs = [f.message for f in check_tool_result_match(normalize(rec))]
    assert msgs == ["result 'result' answers no pending call"]


TAGGED_GET_USER = '<tool_call>{"name": "get_user", "arguments": {"user_id": 42}}</tool_call>'


def test_tool_response_id_field_is_payload_not_call_id():
    rec = {
        "conversations": [
            {"from": "human", "value": "Show user 42."},
            {"from": "gpt", "value": TAGGED_GET_USER},
            {"from": "tool", "value": '<tool_response>{"id": 42, "name": "Ada"}</tool_response>'},
            {"from": "gpt", "value": "User 42 is Ada."},
        ]
    }
    assert check_tool_result_match(normalize(rec)) == []


def test_id_less_tagged_call_answered_by_tool_call_id():
    rec = {
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": '<tool_call>{"name": "f"}</tool_call>'},
            {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
            {"role": "assistant", "content": "done"},
        ]
    }
    assert check_tool_result_match(normalize(rec)) == []


def test_nan_and_infinity_are_not_json():
    for bad in ('{"n": NaN}', '{"n": Infinity}', '{"n": -Infinity}'):
        args, err = parse_arguments(ToolCall("f", bad))
        assert args is None and err.startswith("arguments are not valid JSON: ")
    assert parse_arguments(ToolCall("f", '{"n": "NaN"}')) == ({"n": "NaN"}, None)


def test_payload_with_name_and_content_is_not_an_envelope():
    read = '<tool_call>{"name": "read_file", "arguments": {"path": "notes.txt"}}</tool_call>'
    rec = {
        "conversations": [
            {"from": "human", "value": "What is in notes.txt?"},
            {"from": "gpt", "value": read},
            {
                "from": "tool",
                "value": '<tool_response>{"name": "notes.txt", "content": "buy milk"}'
                "</tool_response>",
            },
            {"from": "gpt", "value": "It says: buy milk."},
        ]
    }
    assert check_tool_result_match(normalize(rec)) == []


def test_payload_with_extra_fields_is_matched_by_position():
    get = '<tool_call>{"name": "get_product", "arguments": {"id": 7}}</tool_call>'
    body = {"name": "Blue Widget", "content": {"colour": "blue"}, "price": 3}
    rec = {
        "messages": [
            {"role": "user", "content": "Price of product 7?"},
            {"role": "assistant", "content": get},
            {"role": "tool", "content": f"<tool_response>{json.dumps(body)}</tool_response>"},
            {"role": "assistant", "content": "It costs 3."},
        ]
    }
    assert check_tool_result_match(normalize(rec)) == []


def test_name_hint_picks_the_call_it_names():
    calls = (
        '<tool_call>{"name": "a", "arguments": {}}</tool_call>'
        '<tool_call>{"name": "b", "arguments": {}}</tool_call>'
    )
    rec = {
        "conversations": [
            {"from": "human", "value": "q"},
            {"from": "gpt", "value": calls},
            {
                "from": "tool",
                "value": '<tool_response>{"name": "b", "content": "x"}</tool_response>',
            },
            {"from": "gpt", "value": "done"},
        ]
    }
    assert [f.message for f in check_tool_result_match(normalize(rec))] == [
        "call 'a' gets no result"
    ]


def test_envelope_with_object_content_still_names_its_call():
    rec = {
        "conversations": [
            {"from": "human", "value": "q"},
            {"from": "gpt", "value": '<tool_call>{"name": "a", "arguments": {}}</tool_call>'},
            {
                "from": "tool",
                "value": '<tool_response>{"name": "z", "content": {}}</tool_response>',
            },
            {"from": "gpt", "value": "done"},
        ]
    }
    assert [f.message for f in check_tool_result_match(normalize(rec))] == [
        "result 'z' answers no pending call",
        "call 'a' gets no result",
    ]


def test_pending_call_after_a_final_tool_turn():
    c = conv(U, asst(call("1", "weather", "{}"), call("2", "clock", "{}")), tool("1"))
    assert [(f.turn, f.message) for f in check_tool_result_match(c)] == [
        (1, "call 'clock' gets no result")
    ]
    ends_on_call = conv(U, asst(call("1", "weather", "{}"), call("2", "clock", "{}")))
    assert check_tool_result_match(ends_on_call) == []
