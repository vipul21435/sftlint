"""Lint the bundled sample file from Python and print a per-rule summary.

Run from the repository root:  uv run python examples/lint_sample.py
"""

from pathlib import Path

from sftlint.lint import lint
from sftlint.loader import load
from sftlint.report import to_markdown

path = Path(__file__).with_name("sample.jsonl")
report = lint(load(path), path.name)
print(to_markdown(report, max_examples=10))
