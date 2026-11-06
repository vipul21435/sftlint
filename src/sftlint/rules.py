"""Per-conversation structural rules.

Each rule is a function ``(Conversation) -> list[Finding]`` registered in :data:`RULES`
with its default severity and a one-line summary. Dataset-level rules live in
:mod:`sftlint.dataset_rules`.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from sftlint.findings import Finding
from sftlint.formats import ROLES, Conversation


@dataclass(frozen=True)
class RuleInfo:
    id: str
    severity: str
    summary: str


CheckFn = Callable[[Conversation], list[Finding]]

RULES: dict[str, RuleInfo] = {}
CHECKS: dict[str, CheckFn] = {}

# Load-time rules emitted by sftlint.loader, registered here so they can be listed.
LOAD_RULES = {
    "json-parse": RuleInfo("json-parse", "error", "line is not valid JSON"),
    "format-mismatch": RuleInfo(
        "format-mismatch", "error", "record does not match the detected or given format"
    ),
}
RULES.update(LOAD_RULES)


def rule(rule_id: str, severity: str, summary: str) -> Callable[[CheckFn], CheckFn]:
    def wrap(fn: CheckFn) -> CheckFn:
        RULES[rule_id] = RuleInfo(rule_id, severity, summary)
        CHECKS[rule_id] = fn
        return fn

    return wrap


def _f(rule_id: str, conv: Conversation, message: str, turn: int | None = None) -> Finding:
    return Finding(rule_id, RULES[rule_id].severity, conv.line, message, turn)


@rule("role-unknown", "error", "a turn has a role outside system, user, assistant, tool")
def check_role_unknown(conv: Conversation) -> list[Finding]:
    return [
        _f("role-unknown", conv, f"unknown role {m.role!r}", i)
        for i, m in enumerate(conv.messages)
        if m.role not in ROLES
    ]


@rule("system-position", "error", "a system turn appears after the first turn")
def check_system_position(conv: Conversation) -> list[Finding]:
    return [
        _f("system-position", conv, "system turn is not the first turn", i)
        for i, m in enumerate(conv.messages)
        if m.role == "system" and i > 0
    ]


@rule(
    "role-order", "error", "turns do not alternate user and assistant, or a tool turn is orphaned"
)
def check_role_order(conv: Conversation) -> list[Finding]:
    """Expected grammar after an optional leading system turn::

        user (assistant-with-tool-calls tool+)* assistant user ...

    A user turn follows an assistant turn; an assistant turn follows a user or tool
    turn; a tool turn follows an assistant turn that made tool calls, or another tool
    turn. Unknown roles and misplaced system turns are skipped (other rules own them).
    """
    out: list[Finding] = []
    prev = None  # the previous checked turn
    for i, m in enumerate(conv.messages):
        if m.role not in ROLES or m.role == "system":
            continue
        if prev is None:
            if m.role != "user":
                out.append(_f("role-order", conv, f"first turn is {m.role}, expected user", i))
        elif m.role == "user" and prev.role == "user":
            out.append(_f("role-order", conv, "two user turns in a row", i))
        elif m.role == "user" and prev.role == "tool":
            out.append(_f("role-order", conv, "user turn follows a tool result", i))
        elif m.role == "assistant" and prev.role == "assistant" and not prev.tool_calls:
            out.append(_f("role-order", conv, "two assistant turns in a row", i))
        elif m.role == "tool" and not (
            prev.role == "tool" or (prev.role == "assistant" and prev.tool_calls)
        ):
            out.append(
                _f("role-order", conv, "tool turn does not follow an assistant tool call", i)
            )
        prev = m
    return out


@rule("empty-turn", "error", "a turn has no text and no tool calls")
def check_empty_turn(conv: Conversation) -> list[Finding]:
    out = []
    for i, m in enumerate(conv.messages):
        if not m.content.strip() and not m.tool_calls:
            what = "missing" if m.content_missing else "empty"
            out.append(_f("empty-turn", conv, f"{m.role} turn content is {what}", i))
    return out


def _norm(text: str) -> str:
    return " ".join(text.split())


@rule("duplicate-turn", "warning", "a turn repeats an earlier turn of the same role verbatim")
def check_duplicate_turn(conv: Conversation) -> list[Finding]:
    seen: dict[tuple[str, str], int] = {}
    out = []
    for i, m in enumerate(conv.messages):
        key = (m.role, _norm(m.content))
        if not key[1]:
            continue
        if key in seen:
            out.append(_f("duplicate-turn", conv, f"{m.role} turn repeats turn {seen[key]}", i))
        else:
            seen[key] = i
    return out


@rule("final-role", "warning", "the last turn is not an assistant turn")
def check_final_role(conv: Conversation) -> list[Finding]:
    if not conv.messages:
        return [_f("final-role", conv, "conversation has no turns")]
    last = conv.messages[-1]
    if last.role != "assistant":
        return [_f("final-role", conv, f"last turn is {last.role}", len(conv.messages) - 1)]
    return []


_DANGLING_CHARS = ",;:([{\\-/&|+="
_DANGLING_WORDS = frozenset(
    "a an the and or but of to in on for with by from as at is are was were that which "
    "because if then than".split()
)
_LAST_WORD = re.compile(r"([A-Za-z]+)\s*$")


def truncation_reason(text: str) -> str | None:
    """Return why ``text`` looks cut off, or ``None``.

    Signals, each cheap and conservative: an odd number of triple-backtick fences
    (an unclosed code block); a final character that cannot end a sentence
    (comma, colon, open bracket, operator); or a final word that is an article,
    conjunction or preposition.
    """
    stripped = text.rstrip()
    if not stripped:
        return None
    if stripped.count("```") % 2 == 1:
        return "unclosed code fence"
    if stripped[-1] in _DANGLING_CHARS and not stripped.endswith(("--", ":)", ":-)")):
        return f"ends with {stripped[-1]!r}"
    match = _LAST_WORD.search(stripped)
    if match and match.group(1).lower() in _DANGLING_WORDS:
        return f"ends with the word {match.group(1)!r}"
    return None


@rule("truncated-final", "warning", "the final assistant turn looks cut off mid-output")
def check_truncated_final(conv: Conversation) -> list[Finding]:
    if not conv.messages or conv.messages[-1].role != "assistant":
        return []
    last = conv.messages[-1]
    reason = truncation_reason(last.content)
    if reason is None and last.unterminated and last.content.strip():
        reason = "no end-of-turn token after the final turn"
    if reason is None:
        return []
    return [_f("truncated-final", conv, reason, len(conv.messages) - 1)]
