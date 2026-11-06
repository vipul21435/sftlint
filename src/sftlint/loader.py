"""Read a JSONL file into normalised conversations plus load-time findings."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sftlint.findings import Finding
from sftlint.formats import Conversation, FormatError, detect_file_format, normalize


@dataclass
class Dataset:
    """The parsed file: conversations, findings from loading, and the file format."""

    conversations: list[Conversation]
    findings: list[Finding] = field(default_factory=list)
    fmt: str | None = None
    records: int = 0


def iter_jsonl(path: str | Path) -> Iterator[tuple[int, Any, str | None]]:
    """Yield ``(line_number, object, error)`` for each non-blank line. A leading UTF-8
    byte-order mark is skipped."""
    with open(path, encoding="utf-8-sig") as fh:
        for lineno, text in enumerate(fh, start=1):
            if not text.strip():
                continue
            try:
                yield lineno, json.loads(text), None
            except json.JSONDecodeError as exc:
                yield lineno, None, f"invalid JSON: {exc.msg} at column {exc.colno}"


def load(path: str | Path, fmt: str = "auto") -> Dataset:
    """Load a JSONL file.

    With ``fmt="auto"`` the file format is a majority vote over the first records and
    each record that does not match it is reported under ``format-mismatch``.
    """
    rows = list(iter_jsonl(path))
    findings: list[Finding] = []
    file_fmt = (
        detect_file_format(obj for _, obj, err in rows if err is None) if fmt == "auto" else fmt
    )
    conversations = []
    for lineno, obj, err in rows:
        if err is not None:
            findings.append(Finding("json-parse", "error", lineno, err))
            continue
        if file_fmt is None:
            findings.append(
                Finding("format-mismatch", "error", lineno, "record matches no known format")
            )
            continue
        try:
            conversations.append(normalize(obj, file_fmt, line=lineno))
        except FormatError as exc:
            findings.append(Finding("format-mismatch", "error", lineno, str(exc)))
    return Dataset(conversations, findings, file_fmt, len(rows))
