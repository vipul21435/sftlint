"""Lint chat, SFT and tool-calling datasets."""

__version__ = "0.1.0"

from sftlint import content, dataset_rules, pii, tools  # noqa: E402, F401  (registers rules)
from sftlint.fix import FixResult, apply_fixes  # noqa: E402
from sftlint.formats import Conversation, Message, detect_format, normalize  # noqa: E402
from sftlint.lint import Report, lint  # noqa: E402
from sftlint.loader import Dataset, load  # noqa: E402
from sftlint.report import to_json, to_markdown  # noqa: E402

__all__ = [
    "Conversation",
    "Dataset",
    "FixResult",
    "Message",
    "Report",
    "apply_fixes",
    "detect_format",
    "lint",
    "load",
    "normalize",
    "to_json",
    "to_markdown",
]
