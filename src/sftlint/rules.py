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


@rule("empty-turn", "error", "a non-tool turn has no text, no tool calls and no media")
def check_empty_turn(conv: Conversation) -> list[Finding]:
    """Tool turns are exempt: a function that returns nothing gives an empty result.

    A turn whose content list holds a non-text part (an image-only user turn in the
    OpenAI vision format) is not empty.
    """
    out = []
    for i, m in enumerate(conv.messages):
        if m.role == "tool" or m.has_media:
            continue
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
_FENCE_LINE = re.compile(r"^ {0,3}(`{3,})(.*)$")
_TICK_RUN = re.compile(r"`{3,}")
_LAST_WORD = re.compile(r"(?:^|\s)([A-Za-z]+)\s*$")
_URL_END = re.compile(r"(?:^|\s)<?https?://\S*[^\s,;:(\[{]$")


def _closed_ending(stripped: str) -> bool:
    """Endings whose last character is in ``_DANGLING_CHARS`` but that are complete:
    a Markdown table row (``| 1 | 2 |``), a name such as ``C++``, a URL ending in ``/``,
    a ``--`` sign-off, a smiley, or a ``||`` verse end."""
    last_line = stripped.rsplit("\n", 1)[-1].strip()
    if len(last_line) > 1 and last_line.startswith("|") and last_line.endswith("|"):
        return True
    if _URL_END.search(stripped):
        return True
    # "||" closes a verse in texts that write the danda as a pipe; ":)" is a smiley.
    return stripped.endswith(("++", "--", ":)", ":-)", "||"))


def unclosed_fence(text: str) -> bool:
    """Whether a backtick code fence is still open at the end of ``text``.

    Fences are paired by line, as CommonMark pairs them: a line that starts (after at
    most three spaces) with three or more backticks opens a block, and only a line of
    at least as many backticks with nothing after them closes it. So a block opened
    with three backticks and closed with four is closed, and a four-backtick block
    that shows a lone triple-backtick line inside it is not cut off. An opening line
    whose info string holds a backtick (`` ```x``` `` on one line) is inline code, not
    a fence.

    One step goes past CommonMark, for model output that writes a fence after text
    on the same line (`` end. ``` `` closing a block, or `` as follows: ``` `` opening
    one): a line that does not start with a fence but ends with a run of
    three or more backticks, and holds an odd number of such runs (so ```` ```x``` ````
    pairs are inline code), closes an open block whose fence is no longer than the
    run, or opens a block when none is open. A fence in the middle of a line, with
    text after it, is not seen.
    """
    open_len = 0
    for line in text.split("\n"):
        m = _FENCE_LINE.match(line)
        if m is None:
            runs = _TICK_RUN.findall(line)
            if len(runs) % 2 and line.rstrip().endswith(runs[-1]):
                ticks = len(runs[-1])
                if open_len == 0:
                    open_len = ticks
                elif ticks >= open_len:
                    open_len = 0
            continue
        ticks, rest = len(m.group(1)), m.group(2)
        if open_len == 0:
            if "`" not in rest:
                open_len = ticks
        elif ticks >= open_len and not rest.strip():
            open_len = 0
    return open_len > 0


def truncation_reason(text: str) -> str | None:
    """Return why ``text`` looks cut off, or ``None``.

    Signals, each cheap and conservative: a code fence left open (see
    ``unclosed_fence``); a final character that cannot end a sentence
    (comma, colon, open bracket, operator) unless the ending is a known complete
    form (see ``_closed_ending``); or a final word in ``_DANGLING_WORDS`` (articles,
    conjunctions, prepositions, copulas and relative words; a single capital ``A``, an
    answer label, does not count).
    """
    stripped = text.rstrip()
    if not stripped:
        return None
    if unclosed_fence(stripped):
        return "unclosed code fence"
    if stripped[-1] in _DANGLING_CHARS and not _closed_ending(stripped):
        return f"ends with {stripped[-1]!r}"
    match = _LAST_WORD.search(stripped)
    word = match.group(1) if match else ""
    # A lone capital letter is an answer label ("The answer is A", "Vitamin A").
    if word.lower() in _DANGLING_WORDS and not (len(word) == 1 and word.isupper()):
        return f"ends with the word {word!r}"
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
