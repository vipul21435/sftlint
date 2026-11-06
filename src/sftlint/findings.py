"""The finding record every rule emits."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

SEVERITIES = ("error", "warning", "info")
_RANK = {"info": 0, "warning": 1, "error": 2}


def severity_rank(severity: str) -> int:
    """Order severities: info < warning < error."""
    return _RANK[severity]


@dataclass(frozen=True)
class Finding:
    """One problem in one record (``line`` is 1-based; ``turn`` is a 0-based index)."""

    rule: str
    severity: str
    line: int
    message: str
    turn: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
