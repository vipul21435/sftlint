import email.message
import importlib.util
import io
import urllib.error
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "fetch_data.py"
spec = importlib.util.spec_from_file_location("fetch_data", SCRIPT)
fetch_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch_data)


def http_error(code, retry_after=None):
    headers = email.message.Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError("https://host.test/x", code, "err", headers, io.BytesIO())


def fake_urlopen(outcomes):
    def _open(url, timeout):
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return io.BytesIO(outcome)

    return _open


def test_get_honours_retry_after_and_backs_off(monkeypatch):
    outcomes = [http_error(429, "30"), http_error(429), http_error(503), b"ok"]
    monkeypatch.setattr(fetch_data.urllib.request, "urlopen", fake_urlopen(outcomes))
    waits = []
    assert fetch_data._get("https://host.test/x", sleep=waits.append) == b"ok"
    assert waits == [30.0, 4.0, 8.0]


def test_get_caps_waits_and_raises_other_client_errors(monkeypatch):
    outcomes = [http_error(429, "600"), http_error(404)]
    monkeypatch.setattr(fetch_data.urllib.request, "urlopen", fake_urlopen(outcomes))
    waits = []
    with pytest.raises(urllib.error.HTTPError):
        fetch_data._get("https://host.test/x", sleep=waits.append)
    assert waits == [60.0]


def test_atomic_write_leaves_nothing_on_failure(tmp_path):
    path = tmp_path / "rows.jsonl"
    with pytest.raises(RuntimeError):
        with fetch_data._atomic_write(path) as fh:
            fh.write("partial\n")
            raise RuntimeError("network")
    assert list(tmp_path.iterdir()) == []
    with fetch_data._atomic_write(path) as fh:
        fh.write("done\n")
    assert path.read_text() == "done\n" and list(tmp_path.iterdir()) == [path]


def test_glaive_row_becomes_messages_with_tools_and_calls():
    row = {
        "system": 'SYSTEM: Use these if needed -\n{"name": "get_time", "parameters": '
        '{"type": "object", "properties": {"zone": {"type": "string"}}}}\n\n'
        '{"name": "add", "parameters": {"type": "object"}}\n',
        "chat": "USER: What time is it in Lima?\n\n\n"
        "ASSISTANT: Let me check. <|endoftext|>\n\n\n"
        "ASSISTANT: <functioncall> "
        '{"name": "get_time", "arguments": \'{"zone": "it\\\'s Lima"}\'} <|endoftext|>\n\n\n'
        'FUNCTION RESPONSE: {"time": "10:00"}\n\n\n'
        "ASSISTANT: It is 10:00 there.",
    }
    stats = {}
    rec = fetch_data.glaive_to_messages(row, stats)
    assert [t["function"]["name"] for t in rec["tools"]] == ["get_time", "add"]
    roles = [m["role"] for m in rec["messages"]]
    assert roles == ["system", "user", "assistant", "assistant", "tool", "assistant"]
    assert rec["messages"][0]["content"] == "Use these if needed -"
    assert rec["messages"][2]["content"] == "Let me check."
    call = rec["messages"][3]["tool_calls"][0]["function"]
    assert call == {"name": "get_time", "arguments": '{"zone": "it\'s Lima"}'}
    assert rec["messages"][3]["content"] == ""
    assert rec["messages"][4]["content"] == '{"time": "10:00"}'
    assert stats == {"quoted": 1, "no_end": 1}


@pytest.mark.parametrize(
    "body, expected, how",
    [
        ('{"name": "f", "arguments": {"a": 1}}', {"name": "f", "arguments": '{"a": 1}'}, "json"),
        (
            '{"name": "f", "arguments": "{\\"a\\": 1}"}',
            {"name": "f", "arguments": '{"a": 1}'},
            "json",
        ),
        (
            '{"name": "f", "arguments": {a: 1}}',
            {"name": "f", "arguments": '{"name": "f", "arguments": {a: 1}}'},
            "raw",
        ),
        ("not a call", {"name": None, "arguments": "not a call"}, "raw"),
    ],
)
def test_glaive_call_layouts(body, expected, how):
    assert fetch_data._glaive_call(body) == (expected, how)


def test_glaive_system_without_functions_declares_no_tools():
    rec = fetch_data.glaive_to_messages(
        {
            "system": "SYSTEM: You are a helpful assistant.",
            "chat": "USER: Hi\n\n\nASSISTANT: Hello. <|endoftext|>",
        }
    )
    assert rec == {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello."},
        ],
        "tools": [],
    }


def test_main_rejects_unknown_source(monkeypatch, tmp_path):
    monkeypatch.setattr("sys.argv", ["fetch_data.py", "--out", str(tmp_path), "--only", "nope"])
    with pytest.raises(SystemExit) as exc:
        fetch_data.main()
    assert exc.value.code == 2
