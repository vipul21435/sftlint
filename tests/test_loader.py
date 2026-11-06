from sftlint.cli import main
from sftlint.loader import load


def test_load_reports_bad_json_and_mismatch(write_jsonl):
    path = write_jsonl(
        [
            {"prompt": "a", "completion": "b"},
            "{not json",
            "",
            {"prompt": "c", "completion": "d"},
            {"instruction": "x", "output": "y"},
        ]
    )
    data = load(path)
    assert data.fmt == "prompt-completion"
    assert data.records == 4
    assert len(data.conversations) == 2
    rules = sorted(f.rule for f in data.findings)
    assert rules == ["format-mismatch", "json-parse"]
    assert {f.line for f in data.findings} == {2, 5}


def test_load_unknown_file(write_jsonl):
    data = load(write_jsonl([{"text": "x"}]))
    assert data.fmt is None
    assert data.findings[0].rule == "format-mismatch"


def test_cli_detect(write_jsonl, capsys):
    path = write_jsonl([{"instruction": "x", "output": "y"}])
    assert main(["detect", str(path)]) == 0
    assert "format: alpaca" in capsys.readouterr().out
    assert main(["detect", str(write_jsonl([{"t": 1}], "b.jsonl"))]) == 1


def test_utf8_byte_order_mark_is_accepted(tmp_path):
    path = tmp_path / "bom.jsonl"
    rec = '{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"Hello."}]}'
    path.write_bytes(b"\xef\xbb\xbf" + rec.encode() + b"\n" + rec.replace("hi", "yo").encode())
    data = load(path)
    assert data.fmt == "messages" and data.findings == [] and len(data.conversations) == 2
    out = tmp_path / "out.jsonl"
    args = ["check", str(path), "--fix", str(out), "--fix-rules", "empty-turn"]
    assert main(args) == 0
    assert out.read_bytes().startswith(b'{"messages"')
