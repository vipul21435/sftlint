"""Lint the bundled sample, then write a copy without what three rules flag.

Run from the repository root:  uv run python examples/fix_sample.py
"""

import tempfile
from pathlib import Path

from sftlint import apply_fixes, lint, load
from sftlint.fix import fix_action

path = Path(__file__).with_name("sample.jsonl")
data = load(path)
report = lint(data, path.name)
rules = ["final-role", "tool-result-match", "duplicate-prompt"]
for rule in rules:
    n = report.counts().get(rule, 0)
    print(f"{rule}: {n} finding{'' if n == 1 else 's'}, fix action {fix_action(rule)}")

with tempfile.TemporaryDirectory() as tmp:
    out = Path(tmp) / "fixed.jsonl"
    result = apply_fixes(path, data, report.findings, rules, str(out))
    print(result.summary())
    after = lint(load(out), select=rules)
    n = len(after.findings)
    print(f"after the fix: {n} finding{'' if n == 1 else 's'} of those rules")
