import json

import pytest

from sftlint.cli import main
from sftlint.lint import lint, selected_rules
from sftlint.loader import load
from sftlint.report import to_dict, to_markdown

ROWS = [
    {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello."}]},
    {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hey."}]},
    {"messages": [{"role": "user", "content": "Q"}, {"role": "user", "content": "Q2"}]},
    "{broken",
]


def test_lint_counts_and_order(write_jsonl):
    path = write_jsonl(ROWS)
    report = lint(load(path), str(path))
    assert report.counts() == {
        "duplicate-prompt": 1,
        "final-role": 1,
        "json-parse": 1,
        "role-order": 1,
    }
    assert [f.line for f in report.findings] == sorted(f.line for f in report.findings)
    assert report.worst() == "error"
    assert report.flagged_records() == 3


def test_select_and_ignore():
    assert selected_rules(["role-order"], None) == {"role-order"}
    assert "role-order" not in selected_rules(None, ["role-order"])
    with pytest.raises(ValueError):
        selected_rules(["nope"], None)


def test_reports(write_jsonl):
    report = lint(load(write_jsonl(ROWS)))
    d = to_dict(report, max_examples=2)
    assert d["findings_truncated"] and len(d["findings"]) == 2
    assert d["severity_counts"] == {"error": 2, "warning": 2, "info": 0}
    md = to_markdown(report)
    assert "| role-order | error | 1 |" in md and "## Lengths" in md
    clean = lint(load(write_jsonl(ROWS[:1], "c.jsonl")))
    assert "No findings." in to_markdown(clean)
    assert clean.worst() is None


def test_cli_exit_codes(write_jsonl, capsys):
    path = str(write_jsonl(ROWS))
    assert main(["check", path]) == 1
    assert main(["check", path, "--fail-on", "never"]) == 0
    assert main(["check", path, "--ignore", "json-parse,role-order"]) == 0
    assert main(["check", path, "--ignore", "json-parse,role-order", "--fail-on", "warning"]) == 1
    assert main(["check", path, "--select", "bogus"]) == 2
    assert main(["check", path + ".missing"]) == 2
    out = capsys.readouterr().out
    assert "4 records" in out and "role-order" in out


def test_cli_writes_reports(write_jsonl, tmp_path, capsys):
    path = str(write_jsonl(ROWS))
    js, md = tmp_path / "r.json", tmp_path / "r.md"
    main(["check", path, "--json", str(js), "--markdown", str(md)])
    assert json.loads(js.read_text())["rule_counts"]["json-parse"] == 1
    assert md.read_text().startswith("# sftlint report")
    capsys.readouterr()
    main(["check", path, "--json", "-"])
    assert json.loads(capsys.readouterr().out)["records"] == 4


def test_cli_rules_lists_every_rule(capsys):
    assert main(["rules"]) == 0
    out = capsys.readouterr().out
    for rid in ("role-order", "truncated-final", "duplicate-prompt", "length-outlier"):
        assert rid in out


def test_cli_detect_reports_io_errors(tmp_path, capsys):
    assert main(["detect", str(tmp_path / "missing.jsonl")]) == 2
    bad = tmp_path / "latin1.jsonl"
    bad.write_bytes(b'{"prompt": "caf\xe9", "completion": "ok"}\n')
    assert main(["detect", str(bad)]) == 2
    assert main(["check", str(bad)]) == 2
    err = capsys.readouterr().err
    assert err.count("sftlint:") == 3 and "Traceback" not in err


@pytest.mark.parametrize(
    "flags",
    [["--max-refusal-rate", "-1"], ["--max-refusal-rate", "1.5"], ["--max-examples", "-3"]],
)
def test_cli_rejects_out_of_range_values(write_jsonl, flags, capsys):
    path = str(write_jsonl(ROWS))
    with pytest.raises(SystemExit) as exc:
        main(["check", path, *flags])
    assert exc.value.code == 2
    assert "error:" in capsys.readouterr().err


def test_cli_report_write_error_exits_2(write_jsonl, tmp_path):
    path = str(write_jsonl(ROWS))
    assert main(["check", path, "--json", str(tmp_path / "no" / "dir" / "r.json")]) == 2
