import json

from sftlint.cli import main
from sftlint.fix import apply_fixes, fix_action
from sftlint.lint import lint
from sftlint.loader import load


def m(role, content):
    return {"role": role, "content": content}


def run_fix(write_jsonl, tmp_path, rows, rules):
    path = write_jsonl(rows)
    data = load(path)
    report = lint(data, str(path))
    out = tmp_path / "fixed.jsonl"
    result = apply_fixes(path, data, report.findings, rules, str(out))
    lines = out.read_text(encoding="utf-8").splitlines()
    return result, lines


def test_fix_action():
    assert fix_action("final-role") == "trim" and fix_action("empty-turn") == "trim"
    assert fix_action("duplicate-prompt") == "drop"


def test_untouched_records_are_written_byte_for_byte(write_jsonl, tmp_path):
    raw = '{"messages":[{"role":"user","content":"a"},{"role":"assistant","content":"b"}]}'
    result, lines = run_fix(write_jsonl, tmp_path, [raw], ["final-role"])
    assert lines == [raw] and result.written == 1


def test_trim_final_role_and_trailing_empty(write_jsonl, tmp_path):
    rows = [
        {"messages": [m("user", "a"), m("assistant", "b"), m("user", "c")], "id": 1},
        {"messages": [m("user", "a"), m("assistant", "b"), m("user", "c"), m("assistant", " ")]},
        {"messages": [m("user", "a"), m("assistant", ""), m("user", "c"), m("assistant", "d")]},
    ]
    result, lines = run_fix(write_jsonl, tmp_path, rows, ["final-role", "empty-turn"])
    assert json.loads(lines[0]) == {"messages": [m("user", "a"), m("assistant", "b")], "id": 1}
    # Cutting the trailing empty answer would leave a user turn last: dropped instead.
    # The empty turn in the middle cannot be cut without losing real turns: dropped.
    assert len(lines) == 1
    assert result.records_trimmed == 1 and result.records_dropped == 2
    assert result.dropped == {"empty-turn": 2}
    assert result.trimmed == {"final-role": 1}
    assert "1 trimmed" in result.summary()


def test_trim_trailing_empty_turns(write_jsonl, tmp_path):
    row = {"messages": [m("user", "a"), m("assistant", "b"), m("user", ""), m("assistant", "")]}
    result, lines = run_fix(write_jsonl, tmp_path, [row], ["empty-turn"])
    assert json.loads(lines[0]) == {"messages": [m("user", "a"), m("assistant", "b")]}
    assert result.trimmed == {"empty-turn": 1}


def test_only_named_rules_act(write_jsonl, tmp_path):
    rows = [
        {"messages": [m("user", "q"), m("assistant", "a")]},
        {"messages": [m("user", "q"), m("assistant", "a")]},
        {"messages": [m("user", "x"), m("user", "y"), m("assistant", "z")]},
        "not json",
    ]
    result, lines = run_fix(write_jsonl, tmp_path, rows, ["duplicate-prompt", "json-parse"])
    assert len(lines) == 2 and result.records_dropped == 2
    assert result.dropped == {"duplicate-prompt": 1, "json-parse": 1}


def test_trim_falls_back_to_drop_without_turn_list(write_jsonl, tmp_path):
    rows = [{"instruction": "q", "output": " "}, {"instruction": "q2", "output": "a"}]
    result, lines = run_fix(write_jsonl, tmp_path, rows, ["empty-turn"])
    assert result.records_dropped == 1 and len(lines) == 1


def test_sharegpt_trim_respects_system_field(write_jsonl, tmp_path):
    row = {
        "system": "s",
        "conversations": [
            {"from": "human", "value": "a"},
            {"from": "gpt", "value": "b"},
            {"from": "human", "value": "c"},
        ],
    }
    _, lines = run_fix(write_jsonl, tmp_path, [row], ["final-role"])
    assert json.loads(lines[0])["conversations"] == row["conversations"][:2]


def test_cli_fix(write_jsonl, tmp_path, capsys):
    path = write_jsonl([{"messages": [m("user", "a"), m("assistant", "b"), m("user", "c")]}])
    out = tmp_path / "o.jsonl"
    code = main(["check", str(path), "--fix", str(out), "--fix-rules", "final-role"])
    assert code == 0
    assert "1 trimmed" in capsys.readouterr().err
    assert main(["check", str(path), "--fix", str(out)]) == 2
    assert main(["check", str(path), "--fix", str(out), "--fix-rules", "nope"]) == 2
