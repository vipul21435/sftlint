"""Run every selected rule over a loaded dataset."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from sftlint.content import check_refusal_rate, content_stats
from sftlint.dataset_rules import check_duplicate_prompt, check_length_outlier
from sftlint.findings import Finding, severity_rank
from sftlint.loader import Dataset
from sftlint.rules import CHECKS, RULES
from sftlint.stats import length_stats
from sftlint.tokens import TokenCounter, pretokens


@dataclass
class Report:
    """Findings, per-rule counts and length statistics for one file."""

    path: str
    fmt: str | None
    records: int
    findings: list[Finding]
    stats: dict[str, Any] = field(default_factory=dict)

    def counts(self) -> dict[str, int]:
        c = Counter(f.rule for f in self.findings)
        return {rule: c[rule] for rule in sorted(c)}

    def severity_counts(self) -> dict[str, int]:
        c = Counter(f.severity for f in self.findings)
        return {s: c.get(s, 0) for s in ("error", "warning", "info")}

    def flagged_records(self) -> int:
        return len({f.line for f in self.findings})

    def worst(self) -> str | None:
        if not self.findings:
            return None
        return max((f.severity for f in self.findings), key=severity_rank)


def selected_rules(select: Iterable[str] | None, ignore: Iterable[str] | None) -> set[str]:
    """Resolve ``--select`` / ``--ignore`` to a set of rule ids. Unknown ids raise."""
    chosen = set(select) if select else set(RULES)
    dropped = set(ignore or ())
    unknown = (chosen | dropped) - set(RULES)
    if unknown:
        raise ValueError("unknown rule id: " + ", ".join(sorted(unknown)))
    return chosen - dropped


def lint(
    data: Dataset,
    path: str = "",
    select: Iterable[str] | None = None,
    ignore: Iterable[str] | None = None,
    outlier_z: float = 3.5,
    count_tokens: TokenCounter = pretokens,
    tokenizer: str = "pretokens",
    max_refusal_rate: float | None = None,
) -> Report:
    """Lint ``data`` and return a :class:`Report` with findings sorted by line.

    With ``max_refusal_rate`` set, a refusal rate above it adds one dataset-level
    ``refusal-rate`` finding (line 0).
    """
    rules = selected_rules(select, ignore)
    findings = [f for f in data.findings if f.rule in rules]
    for conv in data.conversations:
        for rule_id, check in CHECKS.items():
            if rule_id in rules:
                findings.extend(check(conv))
    if "duplicate-prompt" in rules:
        findings.extend(check_duplicate_prompt(data.conversations))
    if "length-outlier" in rules:
        findings.extend(check_length_outlier(data.conversations, threshold=outlier_z))
    stats: dict[str, Any] = {"tokenizer": tokenizer}
    stats["lengths"] = length_stats(data.conversations, count_tokens)
    stats["content"] = content_stats(data.conversations)
    if "refusal-rate" in rules:
        findings.extend(check_refusal_rate(stats["content"], max_refusal_rate))
    findings.sort(key=lambda f: (f.line, -1 if f.turn is None else f.turn, f.rule))
    return Report(
        path=path,
        fmt=data.fmt,
        records=data.records,
        findings=findings,
        stats=stats,
    )
