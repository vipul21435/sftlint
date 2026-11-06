"""``--fix``: write a copy of the file with only what the named rules flag removed.

Each rule has one fix action:

* ``trim`` (``final-role``, ``empty-turn``): cut the turn list where the finding points.
  ``final-role`` cuts every turn after the last assistant turn; ``empty-turn`` cuts
  from the first flagged empty turn, which is allowed only when every turn from there
  on is empty too, so no real turn is lost. If a trim would leave a record that does
  not end in a non-empty assistant turn, or the record has no turn list to cut (Alpaca,
  prompt/completion), the record is dropped instead and counted as such.
* ``drop`` (every other rule): drop each record with a finding of that rule.

Only findings of the rules passed in ``rules`` are acted on; other findings are left
alone, and records without a selected finding are written back unchanged (blank lines are
skipped). The
fixed file is not re-linted: run ``sftlint check`` on it again.
"""

from __future__ import annotations

import copy
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from sftlint.findings import Finding
from sftlint.formats import Conversation
from sftlint.loader import Dataset

TRIM_RULES = frozenset({"final-role", "empty-turn"})


@dataclass
class FixResult:
    """What a fix run did. A record with findings of two rules counts under both."""

    written: int = 0
    records_dropped: int = 0
    records_trimmed: int = 0
    dropped: Counter[str] = field(default_factory=Counter)
    trimmed: Counter[str] = field(default_factory=Counter)

    def summary(self) -> str:
        parts = [
            f"{self.written} records written",
            f"{self.records_dropped} dropped",
            f"{self.records_trimmed} trimmed",
        ]
        parts += [f"{n} dropped for {r}" for r, n in sorted(self.dropped.items())]
        parts += [f"{n} trimmed for {r}" for r, n in sorted(self.trimmed.items())]
        return ", ".join(parts)


def fix_action(rule_id: str) -> str:
    """``trim`` or ``drop``."""
    return "trim" if rule_id in TRIM_RULES else "drop"


def _cut_point(conv: Conversation, findings: list[Finding]) -> int | None:
    """Index in ``conv.messages`` to cut at, or ``None`` if the trim is not safe."""
    msgs = conv.messages
    cut = len(msgs)
    for f in findings:
        if f.rule == "final-role":
            last = max((i for i, m in enumerate(msgs) if m.role == "assistant"), default=-1)
            cut = min(cut, last + 1)
        elif f.rule == "empty-turn" and f.turn is not None:
            tail = msgs[f.turn :]
            if any(m.content.strip() or m.tool_calls for m in tail):
                return None
            cut = min(cut, f.turn)
    kept = msgs[:cut]
    if not kept or kept[-1].role != "assistant":
        return None
    if not (kept[-1].content.strip() or kept[-1].tool_calls):
        return None
    if conv.turns_key is None or any(m.source_index is None for m in msgs[cut:]):
        return None
    return cut


def apply_fixes(
    path: str | Path, data: Dataset, findings: Iterable[Finding], rules: Iterable[str], out: str
) -> FixResult:
    """Write the fixed copy of ``path`` to ``out`` and return what was done."""
    wanted = set(rules)
    by_line: dict[int, list[Finding]] = defaultdict(list)
    for f in findings:
        if f.rule in wanted:
            by_line[f.line].append(f)
    convs = {c.line: c for c in data.conversations}
    result = FixResult()
    with open(path, encoding="utf-8") as src, open(out, "w", encoding="utf-8") as dst:
        for lineno, text in enumerate(src, start=1):
            if not text.strip():
                continue
            hits = by_line.get(lineno)
            if not hits:
                dst.write(text if text.endswith("\n") else text + "\n")
                result.written += 1
                continue
            drops = sorted({f.rule for f in hits if fix_action(f.rule) == "drop"})
            conv = convs.get(lineno)
            rule_ids = sorted({f.rule for f in hits})
            cut = None if drops or conv is None else _cut_point(conv, hits)
            if cut is None or conv is None or conv.turns_key is None:
                result.records_dropped += 1
                result.dropped.update(rule_ids)
                continue
            record = copy.deepcopy(dict(conv.raw or {}))
            first_cut = conv.messages[cut].source_index if cut < len(conv.messages) else None
            if first_cut is not None:
                record[conv.turns_key] = record[conv.turns_key][:first_cut]
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")
            result.written += 1
            result.records_trimmed += 1
            result.trimmed.update(rule_ids)
    return result
