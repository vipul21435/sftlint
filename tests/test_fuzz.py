"""Seeded random records through load, lint, reports and fix: no crash, and the fixed
copy keeps each record as a prefix of its input and re-lints clean for the fixed rules."""

import json
import random

from sftlint.fix import apply_fixes
from sftlint.lint import lint
from sftlint.loader import load
from sftlint.report import to_json, to_markdown

ROLES = ["user", "assistant", "tool", "system", "gpt", None]
TEXTS = ["", " ", "Hi.", "The answer is", "```py\nx = [1,", "api_key = settings.KEY", "A"]


def _turn(rng):
    role = rng.choice(ROLES)
    turn = {"role": role, "content": rng.choice(TEXTS + [None, ["part"], {"x": 1}])}
    if role == "assistant" and rng.random() < 0.3:
        name = rng.choice(["f", "g", ["f"], None, 3])
        args = rng.choice(['{"a": 1}', "{bad", '{"n": NaN}', None, [1], {"a": 1}])
        turn["tool_calls"] = [
            {"id": rng.choice(["c1", None, 7]), "function": {"name": name, "arguments": args}}
        ]
    if role == "tool" and rng.random() < 0.5:
        turn["tool_call_id"] = rng.choice(["c1", "c2", None])
    return {k: v for k, v in turn.items() if v is not None or k == "content"}


def _record(rng):
    # One format per file (messages) plus junk lines: dropping records can change a
    # mixed file's majority format, which would change what the re-lint checks.
    if rng.random() < 0.85:
        rec = {"messages": [_turn(rng) for _ in range(rng.randint(1, 6))]}
        if rng.random() < 0.4:
            rec["tools"] = [{"name": "f", "parameters": {"type": "object"}}]
        return json.dumps(rec)
    return rng.choice(["{", "[]", "null", '"x"', "[" * 50 + "]" * 50])


def test_random_files_lint_and_fix_cleanly(tmp_path):
    rng = random.Random(7)
    rules = [
        "final-role",
        "empty-turn",
        "role-order",
        "tool-result-match",
        "json-parse",
        "format-mismatch",
    ]
    for case in range(40):
        path = tmp_path / f"in{case}.jsonl"
        lines = [_record(rng) for _ in range(rng.randint(1, 30))]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        data = load(path)
        report = lint(data, str(path))
        to_json(report)
        to_markdown(report)
        out = tmp_path / f"out{case}.jsonl"
        apply_fixes(path, data, report.findings, rules, str(out))
        fixed = lint(load(out), str(out), select=rules)
        assert fixed.findings == [], (case, fixed.findings[:3])
        originals = [json.loads(x) for x in lines if x.startswith('{"')]
        for text in out.read_text(encoding="utf-8").splitlines():
            rec = json.loads(text)
            assert any(
                o.get("messages", [])[: len(rec["messages"])] == rec["messages"]
                and {k: v for k, v in o.items() if k != "messages"}
                == {k: v for k, v in rec.items() if k != "messages"}
                for o in originals
            )
