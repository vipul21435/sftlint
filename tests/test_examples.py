import runpy
from pathlib import Path

from sftlint.lint import lint
from sftlint.loader import load

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def test_sample_counts():
    report = lint(load(EXAMPLES / "sample.jsonl"))
    assert report.records == 9
    assert report.counts() == {
        "duplicate-prompt": 1,
        "empty-turn": 1,
        "final-role": 1,
        "role-order": 1,
        "system-position": 1,
        "truncated-final": 2,
    }


def test_example_script_runs(capsys):
    runpy.run_path(str(EXAMPLES / "lint_sample.py"), run_name="__main__")
    assert "# sftlint report: sample.jsonl" in capsys.readouterr().out
