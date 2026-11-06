"""Detect the record format of a chat dataset and normalise it to one shape.

Supported layouts:

* ``messages``: a list of ``{"role": ..., "content": ...}`` dicts under ``messages``,
  ``conversation``, ``conversations`` or ``chat``. This is the OpenAI chat format and
  the input shape of Hugging Face chat templates. ``tool_calls``, ``tool_call_id`` and
  ``name`` are kept; a top-level ``tools`` list is kept as the declared tool schemas.
* ``sharegpt``: a list of ``{"from": ..., "value": ...}`` dicts under ``conversations``
  (or ``conversation`` / ``items``). ``function_call`` turns become assistant tool calls
  and ``observation`` turns become tool results.
* ``alpaca``: ``instruction``, optional ``input`` and ``output`` strings.
* ``prompt-completion``: ``prompt`` and ``completion`` strings.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

FORMATS = ("messages", "sharegpt", "alpaca", "prompt-completion")

ROLES = ("system", "user", "assistant", "tool")

_MESSAGE_KEYS = ("messages", "conversation", "conversations", "chat")
_SHAREGPT_KEYS = ("conversations", "conversation", "items")

_MESSAGES_ROLE_MAP = {
    "system": "system",
    "developer": "system",
    "user": "user",
    "assistant": "assistant",
    "tool": "tool",
    "function": "tool",
    "ipython": "tool",
}

_SHAREGPT_ROLE_MAP = {
    "system": "system",
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "chatgpt": "assistant",
    "model": "assistant",
    "bard": "assistant",
    "bing": "assistant",
    "function_call": "assistant",
    "observation": "tool",
    "tool": "tool",
    "function": "tool",
}


@dataclass
class Message:
    """One turn after normalisation.

    ``role`` is one of :data:`ROLES` when the source role was recognised; otherwise it
    is the source role unchanged, so the ``role-unknown`` rule can report it.
    """

    role: str
    content: str
    raw_role: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None
    content_missing: bool = False


@dataclass
class Conversation:
    """A normalised record: its messages, source format, declared tools and line number."""

    messages: list[Message]
    fmt: str
    line: int = 0
    tools: list[dict[str, Any]] | None = None
    raw: Mapping[str, Any] | None = None


class FormatError(ValueError):
    """Raised when a record does not match the requested format."""


def _list_of_dicts(value: Any) -> bool:
    return isinstance(value, list) and len(value) > 0 and all(isinstance(m, dict) for m in value)


def detect_format(record: Any) -> str | None:
    """Return the format name of one record, or ``None`` if it matches none."""
    if not isinstance(record, Mapping):
        return None
    for key in _SHAREGPT_KEYS:
        value = record.get(key)
        if _list_of_dicts(value) and all("from" in m and "value" in m for m in value):
            return "sharegpt"
    for key in _MESSAGE_KEYS:
        value = record.get(key)
        if _list_of_dicts(value) and all("role" in m for m in value):
            return "messages"
    if isinstance(record.get("instruction"), str) and isinstance(record.get("output"), str):
        return "alpaca"
    if isinstance(record.get("prompt"), str) and isinstance(record.get("completion"), str):
        return "prompt-completion"
    return None


def detect_file_format(records: Iterable[Any], sample: int = 200) -> str | None:
    """Majority vote of :func:`detect_format` over the first ``sample`` records."""
    counts: dict[str, int] = {}
    for i, record in enumerate(records):
        if i >= sample:
            break
        fmt = detect_format(record)
        if fmt is not None:
            counts[fmt] = counts.get(fmt, 0) + 1
    if not counts:
        return None
    return max(sorted(counts), key=lambda k: counts[k])


def _text(content: Any) -> tuple[str, bool]:
    """Flatten message content to text. Returns (text, missing)."""
    if content is None:
        return "", True
    if isinstance(content, str):
        return content, False
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, Mapping) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "".join(parts), False
    return json.dumps(content, ensure_ascii=False), False


def _parse_function_call(value: str) -> list[dict[str, Any]]:
    """Turn a ShareGPT ``function_call`` value into OpenAI-style tool calls."""
    try:
        obj = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return [{"function": {"name": None, "arguments": value}, "_unparsed": True}]
    items = obj if isinstance(obj, list) else [obj]
    calls = []
    for item in items:
        if not isinstance(item, Mapping):
            calls.append({"function": {"name": None, "arguments": item}, "_unparsed": True})
            continue
        args = item.get("arguments", item.get("parameters", {}))
        calls.append({"function": {"name": item.get("name"), "arguments": args}})
    return calls


def _messages(record: Mapping[str, Any]) -> list[Message]:
    key = next(k for k in _MESSAGE_KEYS if _list_of_dicts(record.get(k)))
    out = []
    for m in record[key]:
        raw_role = str(m.get("role", ""))
        role = _MESSAGES_ROLE_MAP.get(raw_role.lower(), raw_role)
        text, missing = _text(m.get("content"))
        calls = m.get("tool_calls") or []
        if not calls and isinstance(m.get("function_call"), Mapping):
            calls = [{"function": dict(m["function_call"])}]
        out.append(
            Message(
                role=role,
                content=text,
                raw_role=raw_role,
                tool_calls=list(calls) if isinstance(calls, list) else [],
                tool_call_id=m.get("tool_call_id"),
                name=m.get("name"),
                content_missing=missing,
            )
        )
    return out


def _sharegpt(record: Mapping[str, Any]) -> list[Message]:
    key = next(k for k in _SHAREGPT_KEYS if _list_of_dicts(record.get(k)))
    out = []
    for m in record[key]:
        raw_role = str(m.get("from", ""))
        role = _SHAREGPT_ROLE_MAP.get(raw_role.lower(), raw_role)
        text, missing = _text(m.get("value"))
        calls: list[dict[str, Any]] = []
        if raw_role.lower() == "function_call":
            calls = _parse_function_call(text)
            text = ""
        out.append(
            Message(role, text, raw_role=raw_role, tool_calls=calls, content_missing=missing)
        )
    if isinstance(record.get("system"), str) and record["system"].strip():
        out.insert(0, Message("system", record["system"], raw_role="system"))
    return out


def _alpaca(record: Mapping[str, Any]) -> list[Message]:
    prompt = record["instruction"]
    extra = record.get("input")
    if isinstance(extra, str) and extra.strip():
        prompt = f"{prompt}\n\n{extra}"
    msgs = []
    if isinstance(record.get("system"), str) and record["system"].strip():
        msgs.append(Message("system", record["system"], raw_role="system"))
    msgs.append(Message("user", prompt, raw_role="instruction"))
    msgs.append(Message("assistant", record["output"], raw_role="output"))
    return msgs


def _prompt_completion(record: Mapping[str, Any]) -> list[Message]:
    return [
        Message("user", record["prompt"], raw_role="prompt"),
        Message("assistant", record["completion"], raw_role="completion"),
    ]


_NORMALISERS = {
    "messages": _messages,
    "sharegpt": _sharegpt,
    "alpaca": _alpaca,
    "prompt-completion": _prompt_completion,
}


def normalize(record: Mapping[str, Any], fmt: str | None = None, line: int = 0) -> Conversation:
    """Normalise one record. With ``fmt=None`` the format is detected per record."""
    detected = detect_format(record)
    fmt = fmt or detected
    if fmt not in _NORMALISERS:
        raise FormatError("record matches none of: " + ", ".join(FORMATS))
    if detected != fmt:
        raise FormatError(f"record does not match format {fmt!r}")
    tools = record.get("tools")
    if isinstance(tools, str):
        try:
            tools = json.loads(tools)
        except json.JSONDecodeError:
            tools = None
    return Conversation(
        messages=_NORMALISERS[fmt](record),
        fmt=fmt,
        line=line,
        tools=tools if isinstance(tools, list) else None,
        raw=record,
    )
