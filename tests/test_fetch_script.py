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
