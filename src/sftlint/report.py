"""Render a :class:`~sftlint.lint.Report` as JSON or Markdown."""

from __future__ import annotations

import json
from typing import Any

from sftlint import __version__
from sftlint.lint import Report
from sftlint.rules import RULES


def to_dict(report: Report, max_examples: int | None = None) -> dict[str, Any]:
    findings = report.findings if max_examples is None else report.findings[:max_examples]
    return {
        "tool": "sftlint",
        "version": __version__,
        "path": report.path,
        "format": report.fmt,
        "records": report.records,
        "flagged_records": report.flagged_records(),
        "severity_counts": report.severity_counts(),
        "rule_counts": report.counts(),
        "stats": report.stats,
        "findings": [f.to_dict() for f in findings],
        "findings_truncated": len(findings) < len(report.findings),
    }


def to_json(report: Report, max_examples: int | None = None) -> str:
    return json.dumps(to_dict(report, max_examples), indent=2, ensure_ascii=False) + "\n"


def _fmt_num(x: float) -> str:
    return f"{x:.0f}" if float(x).is_integer() else f"{x:.1f}"


def to_markdown(report: Report, max_examples: int = 20) -> str:
    sev = report.severity_counts()
    lines = [
        f"# sftlint report: {report.path or 'input'}",
        "",
        f"- format: {report.fmt or 'unknown'}",
        f"- records: {report.records}",
        f"- records with findings: {report.flagged_records()}",
        f"- errors: {sev['error']}, warnings: {sev['warning']}, info: {sev['info']}",
        "",
        "## Findings by rule",
        "",
    ]
    counts = report.counts()
    if counts:
        lines += ["| rule | severity | count | what it flags |", "|---|---|---:|---|"]
        for rule, n in counts.items():
            info = RULES[rule]
            lines.append(f"| {rule} | {info.severity} | {n} | {info.summary} |")
    else:
        lines.append("No findings.")
    lengths = report.stats.get("lengths", {})
    if lengths:
        lines += [
            "",
            "## Lengths",
            "",
            "| field | unit | mean | p50 | p90 | p99 | max |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for name, units in lengths.items():
            for unit, s in units.items():
                if s.get("count"):
                    cells = " | ".join(_fmt_num(s[k]) for k in ("mean", "p50", "p90", "p99", "max"))
                    lines.append(f"| {name} | {unit} | {cells} |")
    if report.findings:
        lines += ["", f"## First {min(max_examples, len(report.findings))} findings", ""]
        for f in report.findings[:max_examples]:
            turn = "" if f.turn is None else f" turn {f.turn}"
            lines.append(f"- line {f.line}{turn}: `{f.rule}` {f.message}")
    return "\n".join(lines) + "\n"
