import runpy
from pathlib import Path

from sftlint.lint import lint
from sftlint.loader import load

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def test_sample_counts():
    report = lint(load(EXAMPLES / "sample.jsonl"))
    assert report.records == 13
    assert report.counts() == {
        "duplicate-prompt": 1,
        "empty-turn": 1,
        "final-role": 1,
        "language-mix": 1,
        "refusal": 1,
        "role-order": 1,
        "system-position": 1,
        "tool-args-json": 1,
        "tool-args-schema": 1,
        "tool-result-match": 2,
        "tool-unknown": 1,
        "truncated-final": 2,
    }


def test_example_script_runs(capsys):
    runpy.run_path(str(EXAMPLES / "lint_sample.py"), run_name="__main__")
    assert "# sftlint report: sample.jsonl" in capsys.readouterr().out


def test_fix_example_runs(capsys):
    runpy.run_path(str(EXAMPLES / "fix_sample.py"), run_name="__main__")
    out = capsys.readouterr().out
    assert "11 records written, 2 dropped, 1 trimmed" in out
    assert "after the fix: 0 findings of those rules" in out
