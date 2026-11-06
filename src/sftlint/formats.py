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
* ``template-text``: a ``text`` string already rendered by a chat template, in one of
  three public layouts: ChatML (``<|im_start|>role ... <|im_end|>``), Zephyr
  (``<|user|> ... </s>``) or Llama 3 (``<|start_header_id|>role<|end_header_id|> ...
  <|eot_id|>``). A turn with no end-of-turn token is marked ``unterminated``.

Tool calls are normalised to :class:`ToolCall` whatever their source: an OpenAI
``tool_calls`` list (or the older ``function_call`` field), a ShareGPT
``function_call`` turn, or ``<tool_call>{...}</tool_call>`` blocks inside assistant
text, the layout of the Hermes function-calling data. ``<tool_response>`` blocks in a
tool turn become :class:`ToolResult` entries, and a ``<tools>[...]</tools>`` block in
the system prompt is read as the declared tools when the record has no ``tools`` field.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

FORMATS = ("messages", "sharegpt", "alpaca", "prompt-completion", "template-text")

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
class ToolCall:
    """One call an assistant turn makes.

    ``arguments`` is what the source holds: a JSON string in the OpenAI format, an
    object in most other layouts. ``parse_error`` is set when the call itself could
    not be read (for example a ``<tool_call>`` body that is not JSON).
    """

    name: str | None
    arguments: Any = None
    id: str | None = None
    parse_error: str | None = None


@dataclass
class ToolResult:
    """One result inside a tool turn: the call id and function name it answers, when given."""

    id: str | None = None
    name: str | None = None
    parse_error: str | None = None


@dataclass
class Message:
    """One turn after normalisation.

    ``role`` is one of :data:`ROLES` when the source role was recognised; otherwise it
    is the source role unchanged, so the ``role-unknown`` rule can report it.
    """

    role: str
    content: str
    raw_role: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None
    content_missing: bool = False
    tool_results: list[ToolResult] = field(default_factory=list)
    source_index: int | None = None  # position in the record's turn list, when it has one
    unterminated: bool = False  # rendered text only: no end-of-turn token after this turn

    def results(self) -> list[ToolResult]:
        """The results a tool turn carries: tagged blocks, else one from its id and name."""
        if self.tool_results:
            return self.tool_results
        return [ToolResult(self.tool_call_id, self.name)]


@dataclass
class Conversation:
    """A normalised record: its messages, source format, declared tools and line number."""

    messages: list[Message]
    fmt: str
    line: int = 0
    tools: list[dict[str, Any]] | None = None
    raw: Mapping[str, Any] | None = None
    turns_key: str | None = None  # the field holding the turn list (messages, sharegpt)


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
    text = record.get("text")
    if isinstance(text, str) and _template_layout(text) is not None:
        return "template-text"
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


def _call_from_obj(item: Any) -> ToolCall:
    """Read one call from a ``{"name", "arguments"}`` object (or OpenAI's nested form)."""
    if not isinstance(item, Mapping):
        return ToolCall(None, item, parse_error="call is not a JSON object")
    fn = item.get("function")
    if isinstance(fn, Mapping):
        return ToolCall(fn.get("name"), fn.get("arguments"), id=item.get("id"))
    args = item.get("arguments", item.get("parameters"))
    return ToolCall(item.get("name"), args, id=item.get("id"))


def _parse_function_call(value: str) -> list[ToolCall]:
    """Turn a ShareGPT ``function_call`` value into tool calls."""
    try:
        obj = json.loads(value)
    except (json.JSONDecodeError, TypeError) as exc:
        return [ToolCall(None, value, parse_error=f"invalid JSON: {exc}")]
    items = obj if isinstance(obj, list) else [obj]
    return [_call_from_obj(item) for item in items]


_TAG = {
    name: re.compile(rf"<{name}>\s*(.*?)\s*(?:</{name}>|$)", re.DOTALL)
    for name in ("tool_call", "tool_response", "tools")
}


def _tag_bodies(text: str, tag: str) -> list[str]:
    return [m.group(1) for m in _TAG[tag].finditer(text)] if f"<{tag}>" in text else []


def _strip_tags(text: str, tag: str) -> str:
    return _TAG[tag].sub("", text).strip()


def parse_tagged_calls(text: str) -> tuple[list[ToolCall], str]:
    """Extract ``<tool_call>`` blocks from assistant text. Returns (calls, remaining text).

    A block with no closing tag runs to the end of the text, so a cut-off call is
    still reported (as a parse error) rather than lost.
    """
    bodies = _tag_bodies(text, "tool_call")
    if not bodies:
        return [], text
    calls = []
    for body in bodies:
        try:
            calls.append(_call_from_obj(json.loads(body)))
        except json.JSONDecodeError as exc:
            calls.append(ToolCall(None, body, parse_error=f"invalid JSON: {exc.msg}"))
    return calls, _strip_tags(text, "tool_call")


def parse_tagged_results(text: str) -> list[ToolResult]:
    """Extract ``<tool_response>`` blocks from a tool turn."""
    out = []
    for body in _tag_bodies(text, "tool_response"):
        try:
            obj = json.loads(body)
        except json.JSONDecodeError as exc:
            out.append(ToolResult(parse_error=f"invalid JSON: {exc.msg}"))
            continue
        if isinstance(obj, Mapping):
            out.append(ToolResult(obj.get("tool_call_id", obj.get("id")), obj.get("name")))
        else:
            out.append(ToolResult())
    return out


def _apply_tags(msg: Message) -> Message:
    if msg.role == "assistant" and not msg.tool_calls:
        calls, rest = parse_tagged_calls(msg.content)
        if calls:
            msg.tool_calls, msg.content = calls, rest
    elif msg.role == "tool":
        msg.tool_results = parse_tagged_results(msg.content)
    return msg


def _messages(record: Mapping[str, Any]) -> list[Message]:
    key = next(k for k in _MESSAGE_KEYS if _list_of_dicts(record.get(k)))
    out = []
    for idx, m in enumerate(record[key]):
        raw_role = str(m.get("role", ""))
        role = _MESSAGES_ROLE_MAP.get(raw_role.lower(), raw_role)
        text, missing = _text(m.get("content"))
        raw_calls = m.get("tool_calls") or []
        if not raw_calls and isinstance(m.get("function_call"), Mapping):
            raw_calls = [{"function": m["function_call"]}]
        calls = [_call_from_obj(c) for c in raw_calls] if isinstance(raw_calls, list) else []
        msg = Message(
            role=role,
            content=text,
            raw_role=raw_role,
            tool_calls=calls,
            tool_call_id=m.get("tool_call_id"),
            name=m.get("name"),
            content_missing=missing,
            source_index=idx,
        )
        out.append(_apply_tags(msg))
    return out


def _sharegpt(record: Mapping[str, Any]) -> list[Message]:
    key = next(k for k in _SHAREGPT_KEYS if _list_of_dicts(record.get(k)))
    out = []
    for idx, m in enumerate(record[key]):
        raw_role = str(m.get("from", ""))
        role = _SHAREGPT_ROLE_MAP.get(raw_role.lower(), raw_role)
        text, missing = _text(m.get("value"))
        calls: list[ToolCall] = []
        if raw_role.lower() == "function_call":
            calls = _parse_function_call(text)
            text = ""
        msg = Message(role, text, raw_role, calls, content_missing=missing, source_index=idx)
        out.append(_apply_tags(msg))
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


_ZEPHYR_ROLES = r"(?:system|user|assistant)"
_TEMPLATES = {
    "chatml": re.compile(
        r"<\|im_start\|>(\w+)\n?(.*?)(<\|im_end\|>|(?=<\|im_start\|>)|$)", re.DOTALL
    ),
    "llama3": re.compile(
        r"<\|start_header_id\|>(\w+)<\|end_header_id\|>\n*(.*?)"
        r"(<\|eot_id\|>|(?=<\|start_header_id\|>)|$)",
        re.DOTALL,
    ),
    "zephyr": re.compile(
        rf"<\|({_ZEPHYR_ROLES})\|>\n?(.*?)(</s>|(?=<\|{_ZEPHYR_ROLES}\|>)|$)", re.DOTALL
    ),
}
_TEMPLATE_MARKERS = {
    "chatml": "<|im_start|>",
    "llama3": "<|start_header_id|>",
    "zephyr": "<|user|>",
}


def _template_layout(text: str) -> str | None:
    for name, marker in _TEMPLATE_MARKERS.items():
        if marker in text:
            return name
    return None


def _template_text(record: Mapping[str, Any]) -> list[Message]:
    text = record["text"]
    layout = _template_layout(text) or "chatml"
    out = []
    for m in _TEMPLATES[layout].finditer(text):
        raw_role, body, end = m.group(1), m.group(2), m.group(3)
        role = _MESSAGES_ROLE_MAP.get(raw_role.lower(), raw_role)
        msg = Message(role, body.rstrip("\n"), raw_role=raw_role, unterminated=not end)
        out.append(_apply_tags(msg))
    return out


_NORMALISERS = {
    "messages": _messages,
    "sharegpt": _sharegpt,
    "alpaca": _alpaca,
    "prompt-completion": _prompt_completion,
    "template-text": _template_text,
}


def normalize(record: Mapping[str, Any], fmt: str | None = None, line: int = 0) -> Conversation:
    """Normalise one record. With ``fmt=None`` the format is detected per record."""
    detected = detect_format(record)
    fmt = fmt or detected
    if fmt not in _NORMALISERS:
        raise FormatError("record matches none of: " + ", ".join(FORMATS))
    if detected != fmt:
        raise FormatError(f"record does not match format {fmt!r}")
    messages = _NORMALISERS[fmt](record)
    keys = _SHAREGPT_KEYS if fmt == "sharegpt" else _MESSAGE_KEYS
    turns_key = next((k for k in keys if _list_of_dicts(record.get(k))), None)
    tools = _load_tools(record.get("tools"))
    if tools is None:
        for m in messages:
            if m.role == "system":
                bodies = _tag_bodies(m.content, "tools")
                tools = _load_tools(bodies[0]) if bodies else None
                break
    return Conversation(messages, fmt, line, tools, record, turns_key)


def _load_tools(value: Any) -> list[dict[str, Any]] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if isinstance(value, list) and all(isinstance(t, Mapping) for t in value):
        return list(value)
    return None
