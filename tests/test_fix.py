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


def test_fix_in_place_keeps_the_data(write_jsonl, tmp_path, capsys):
    rows = [
        {"messages": [m("user", "a"), m("assistant", "b")]},
        {"messages": [m("user", "a"), m("assistant", "b"), m("user", "c")]},
    ]
    path = write_jsonl(rows)
    rc = main(
        ["check", str(path), "--fail-on", "never", "--fix", str(path), "--fix-rules", "final-role"]
    )
    assert rc == 0
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 and json.loads(lines[1])["messages"][-1]["content"] == "b"
    assert "2 records written" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == [path]


def test_fix_failure_leaves_no_partial_output(write_jsonl, tmp_path):
    path = write_jsonl([{"messages": [m("user", "a"), m("assistant", "b")]}])
    data = load(path)
    out = tmp_path / "missing-dir" / "fixed.jsonl"
    try:
        apply_fixes(path, data, [], ["final-role"], str(out))
    except OSError:
        pass
    assert list(tmp_path.iterdir()) == [path]


def test_fix_removes_its_temporary_file_when_reading_fails(write_jsonl, tmp_path):
    path = write_jsonl([{"messages": [m("user", "a"), m("assistant", "b")]}])
    data = load(path)
    path.unlink()
    try:
        apply_fixes(path, data, [], ["final-role"], str(tmp_path / "fixed.jsonl"))
    except OSError:
        pass
    assert list(tmp_path.iterdir()) == []


def test_empty_turn_trim_keeps_a_later_image_turn(write_jsonl, tmp_path):
    image = {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "u"}}]}
    rows = [{"messages": [m("user", "q"), m("assistant", "a"), m("assistant", ""), image]}]
    result, lines = run_fix(write_jsonl, tmp_path, rows, ["empty-turn"])
    assert lines == [] and result.records_dropped == 1 and result.records_trimmed == 0


def test_final_role_trim_keeps_every_exchange(write_jsonl, tmp_path):
    turns = [m("user", "q1"), m("assistant", "a1"), m("user", "q2"), m("assistant", "a2")]
    rows = [{"messages": turns + [m("user", "q3")]}]
    result, lines = run_fix(write_jsonl, tmp_path, rows, ["final-role"])
    assert [json.loads(x)["messages"] for x in lines] == [turns]
    assert result.records_trimmed == 1


def test_empty_turn_fix_drops_when_a_real_turn_follows(write_jsonl, tmp_path):
    turns = [m("user", "q"), m("assistant", "a1"), m("user", ""), m("assistant", "a2")]
    result, lines = run_fix(write_jsonl, tmp_path, [{"messages": turns}], ["empty-turn"])
    assert lines == [] and result.records_dropped == 1 and result.records_trimmed == 0


def test_blank_line_between_records_does_not_stop_the_copy(write_jsonl, tmp_path):
    good = {"messages": [m("user", "q"), m("assistant", "a.")]}
    bad = {"messages": [m("user", "q"), m("assistant", "a."), m("user", "more")]}
    rows = [json.dumps(good), "", "   ", json.dumps(bad), json.dumps(good)]
    result, lines = run_fix(write_jsonl, tmp_path, rows, ["final-role"])
    assert len(lines) == 3 and result.written == 3 and result.records_trimmed == 1


def test_fix_keeps_crlf_line_endings(tmp_path):
    good = json.dumps({"messages": [m("user", "q"), m("assistant", "a.")]})
    trim = json.dumps({"messages": [m("user", "q"), m("assistant", "a."), m("user", "more")]})
    path = tmp_path / "crlf.jsonl"
    path.write_bytes(("\r\n".join([good, trim, good, good]) + "\r\n").encode())
    data = load(path)
    report = lint(data, str(path))
    out = tmp_path / "fixed.jsonl"
    result = apply_fixes(path, data, report.findings, ["final-role", "tool-args-schema"], str(out))
    raw = out.read_bytes()
    assert result.written == 4 and result.records_trimmed == 1
    assert raw.count(b"\r\n") == 4 and raw.count(b"\n") == 4
    assert raw.split(b"\r\n")[0] == good.encode()


def test_fix_adds_the_files_line_ending_to_an_unterminated_last_line(tmp_path):
    good = json.dumps({"messages": [m("user", "q"), m("assistant", "a.")]})
    path = tmp_path / "crlf.jsonl"
    path.write_bytes(f"{good}\r\n{good}".encode())
    data = load(path)
    out = tmp_path / "fixed.jsonl"
    apply_fixes(path, data, lint(data, str(path)).findings, ["final-role"], str(out))
    assert out.read_bytes() == f"{good}\r\n{good}\r\n".encode()
