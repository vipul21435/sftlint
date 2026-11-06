"""Tool-calling rules: arguments, declared schemas, and results matched to calls.

* ``tool-args-json``: a call that could not be read, or whose arguments are not a JSON
  object. In the OpenAI chat format ``arguments`` is a string holding JSON, so a
  truncated or hand-edited call shows up here.
* ``tool-unknown``: a call to a function the record does not declare in ``tools``.
  Skipped when the record declares no tools.
* ``tool-args-schema``: arguments that fail the declared ``parameters`` JSON Schema
  (see :mod:`sftlint.schema` for the supported keywords).
* ``tool-result-match``: a tool result that answers no pending call (by
  ``tool_call_id`` when the calls carry ids, else by function name, else by
  position), or a call that gets no
  result although the conversation goes on. A conversation that ends on the call
  turn is accepted: in single-turn function-calling data the call is the target.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from sftlint.findings import Finding
from sftlint.formats import Conversation, ToolCall
from sftlint.rules import RULES, rule
from sftlint.schema import validate


def declared_tools(conv: Conversation) -> dict[str, Any] | None:
    """Map each declared function name to its ``parameters`` schema (``None`` if absent).

    Accepts the nested OpenAI form ``{"type": "function", "function": {...}}`` and the
    flat ``{"name", "parameters"}`` form. Returns ``None`` when nothing is declared.
    """
    if not conv.tools:
        return None
    out: dict[str, Any] = {}
    for tool in conv.tools:
        fn = tool.get("function") if isinstance(tool.get("function"), Mapping) else tool
        name = fn.get("name")
        if isinstance(name, str):
            out[name] = fn.get("parameters")
    return out or None


def _reject_constant(name: str) -> Any:
    # Python's json module accepts NaN and Infinity; the JSON grammar has neither.
    raise ValueError(f"{name} is not a JSON value")


def parse_arguments(call: ToolCall) -> tuple[dict[str, Any] | None, str | None]:
    """Return ``(arguments, error)``. Missing arguments count as an empty object."""
    if call.parse_error:
        return None, call.parse_error
    args = call.arguments
    if args is None:
        return {}, None
    if isinstance(args, str):
        if not args.strip():
            return {}, None
        try:
            args = json.loads(args, parse_constant=_reject_constant)
        except json.JSONDecodeError as exc:
            return None, f"arguments are not valid JSON: {exc.msg} at column {exc.colno}"
        except ValueError as exc:
            return None, f"arguments are not valid JSON: {exc}"
    if not isinstance(args, Mapping):
        return None, f"arguments are a JSON {type(args).__name__}, expected an object"
    return dict(args), None


def _calls(conv: Conversation):
    for i, m in enumerate(conv.messages):
        if m.role == "assistant":
            for call in m.tool_calls:
                yield i, call


def _f(rule_id: str, conv: Conversation, message: str, turn: int) -> Finding:
    return Finding(rule_id, RULES[rule_id].severity, conv.line, message, turn)


def _label(call: ToolCall) -> str:
    return repr(call.name) if call.name else "unnamed call"


@rule("tool-args-json", "error", "a tool call cannot be read or its arguments are not an object")
def check_tool_args_json(conv: Conversation) -> list[Finding]:
    out = []
    for i, call in _calls(conv):
        _, err = parse_arguments(call)
        if err:
            out.append(_f("tool-args-json", conv, f"{_label(call)}: {err}", i))
    return out


@rule("tool-unknown", "error", "a tool call names a function the record does not declare")
def check_tool_unknown(conv: Conversation) -> list[Finding]:
    tools = declared_tools(conv)
    if tools is None:
        return []
    out = []
    for i, call in _calls(conv):
        if call.parse_error:
            continue
        if not call.name:
            out.append(_f("tool-unknown", conv, "tool call has no function name", i))
        elif call.name not in tools:
            out.append(_f("tool-unknown", conv, f"{call.name!r} is not a declared tool", i))
    return out


@rule("tool-args-schema", "error", "tool-call arguments fail the declared parameters schema")
def check_tool_args_schema(conv: Conversation, max_errors: int = 3) -> list[Finding]:
    tools = declared_tools(conv)
    if tools is None:
        return []
    out = []
    for i, call in _calls(conv):
        schema = tools.get(call.name or "")
        args, err = parse_arguments(call)
        if schema is None or args is None or err:
            continue
        errors = validate(args, schema)
        if errors:
            shown = "; ".join(errors[:max_errors])
            more = f" (+{len(errors) - max_errors} more)" if len(errors) > max_errors else ""
            out.append(_f("tool-args-schema", conv, f"{call.name!r}: {shown}{more}", i))
    return out


def _match(pending: list[ToolCall], rid: str | None, name: str | None) -> ToolCall | None:
    """By id when the pending calls carry ids, else by function name, else the first."""
    if rid is not None and any(c.id is not None for c in pending):
        return next((c for c in pending if c.id == rid), None)
    if name is not None:
        return next((c for c in pending if c.name == name), None)
    return pending[0] if pending else None


@rule("tool-result-match", "error", "a tool result answers no call, or a call gets no result")
def check_tool_result_match(conv: Conversation) -> list[Finding]:
    out: list[Finding] = []
    msgs = conv.messages
    i = 0
    while i < len(msgs):
        m = msgs[i]
        if m.role != "assistant" or not m.tool_calls:
            i += 1
            continue
        call_turn = i
        pending = [c for c in m.tool_calls if not c.parse_error]
        j = i + 1
        while j < len(msgs) and msgs[j].role == "tool":
            for res in msgs[j].results():
                if res.parse_error:
                    out.append(_f("tool-result-match", conv, f"tool result: {res.parse_error}", j))
                    continue
                hit = _match(pending, res.id, res.name)
                if hit is None:
                    key = f"id {res.id!r}" if res.id is not None else repr(res.name or "result")
                    out.append(
                        _f("tool-result-match", conv, f"result {key} answers no pending call", j)
                    )
                else:
                    pending.remove(hit)
            j += 1
        if j < len(msgs) or j > call_turn + 1:
            for call in pending:
                out.append(
                    _f("tool-result-match", conv, f"call {_label(call)} gets no result", call_turn)
                )
        i = j
    return out
