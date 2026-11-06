"""Rules that need the whole dataset: exact duplicate prompts and length outliers.

Exact duplicates only. Near-duplicate detection (MinHash and similar) belongs to
datasieve: https://github.com/vipul21435/datasieve
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

from sftlint.findings import Finding
from sftlint.formats import Conversation
from sftlint.rules import RULES, RuleInfo
from sftlint.stats import length_table, modified_z

DATASET_RULES = {
    "duplicate-prompt": RuleInfo(
        "duplicate-prompt", "warning", "the same prompt appears in an earlier record"
    ),
    "length-outlier": RuleInfo(
        "length-outlier", "info", "prompt or response length is a robust outlier for the set"
    ),
}
RULES.update(DATASET_RULES)


def prompt_key(conv: Conversation) -> str:
    """Hash of every turn before the final assistant turn, roles included,
    with whitespace collapsed. Case is kept: casing differences are real data."""
    msgs = conv.messages
    if msgs and msgs[-1].role == "assistant":
        msgs = msgs[:-1]
    h = hashlib.sha256()
    for m in msgs:
        h.update(m.role.encode())
        h.update(b"\x00")
        h.update(" ".join(m.content.split()).encode())
        h.update(b"\x01")
    return h.hexdigest()


def check_duplicate_prompt(convs: Sequence[Conversation]) -> list[Finding]:
    first: dict[str, int] = {}
    out = []
    sev = DATASET_RULES["duplicate-prompt"].severity
    for conv in convs:
        key = prompt_key(conv)
        if key in first:
            out.append(
                Finding("duplicate-prompt", sev, conv.line, f"same prompt as line {first[key]}")
            )
        else:
            first[key] = conv.line
    return out


def check_length_outlier(
    convs: Sequence[Conversation], threshold: float = 3.5, min_records: int = 20
) -> list[Finding]:
    """Flag records whose character length has ``|modified z| > threshold`` on the log
    scale. Skipped below ``min_records`` records, where the median is not stable."""
    if len(convs) < min_records:
        return []
    sev = DATASET_RULES["length-outlier"].severity
    out = []
    for name, units in length_table(convs).items():
        chars = units["chars"]
        scores = modified_z([math.log1p(c) for c in chars])
        for conv, n, z in zip(convs, chars, scores, strict=True):
            if abs(z) > threshold:
                side = "long" if z > 0 else "short"
                out.append(
                    Finding(
                        "length-outlier",
                        sev,
                        conv.line,
                        f"{name} is {side}: {n} chars, modified z {z:+.1f}",
                    )
                )
    return out
