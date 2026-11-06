"""Lint the bundled sample, then write a copy without what three rules flag.

Run from the repository root:  uv run python examples/fix_sample.py
"""

import tempfile
from pathlib import Path

from sftlint.fix import apply_fixes, fix_action
from sftlint.lint import lint
from sftlint.loader import load

path = Path(__file__).with_name("sample.jsonl")
data = load(path)
report = lint(data, path.name)
rules = ["final-role", "tool-result-match", "duplicate-prompt"]
for rule in rules:
    print(f"{rule}: {report.counts().get(rule, 0)} findings, fix action {fix_action(rule)}")

with tempfile.TemporaryDirectory() as tmp:
    out = Path(tmp) / "fixed.jsonl"
    result = apply_fixes(path, data, report.findings, rules, str(out))
    print(result.summary())
    after = lint(load(out), select=rules)
    print(f"after the fix: {len(after.findings)} findings of those rules")
