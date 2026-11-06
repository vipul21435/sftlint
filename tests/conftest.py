import json
from pathlib import Path

import pytest


@pytest.fixture
def write_jsonl(tmp_path: Path):
    def _write(rows, name="data.jsonl"):
        path = tmp_path / name
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(row if isinstance(row, str) else json.dumps(row))
                fh.write("\n")
        return path

    return _write
