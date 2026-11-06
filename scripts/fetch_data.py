"""Download the small public slices used for the results in the README.

* smoltalk, config smol-magpie-ultra, train split, first --rows rows, through the
  Hugging Face datasets-server rows API (100 rows per request). Licence: Apache 2.0
  for this subset, per https://huggingface.co/datasets/HuggingFaceTB/smoltalk
* hermes-function-calling-v1, file func-calling-singleturn.json (about 18 MB).
  Licence: Apache 2.0, per https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1

Writes JSONL files into --out (default: ./data, which .gitignore excludes). Only the
fields the linter reads are kept; per-record ids are dropped.

    uv run python scripts/fetch_data.py --out data --rows 2000
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROWS_API = "https://datasets-server.huggingface.co/rows"
HERMES_URL = (
    "https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1/"
    "resolve/main/func-calling-singleturn.json"
)


def _get(url: str) -> bytes:
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return resp.read()
        except OSError:
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def fetch_smoltalk(out: Path, rows: int) -> Path:
    path = out / "smoltalk-magpie-ultra.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for offset in range(0, rows, 100):
            query = urllib.parse.urlencode(
                {
                    "dataset": "HuggingFaceTB/smoltalk",
                    "config": "smol-magpie-ultra",
                    "split": "train",
                    "offset": offset,
                    "length": min(100, rows - offset),
                }
            )
            page = json.loads(_get(f"{ROWS_API}?{query}"))
            for item in page["rows"]:
                fh.write(json.dumps({"messages": item["row"]["messages"]}, ensure_ascii=False))
                fh.write("\n")
    return path


def fetch_hermes(out: Path) -> Path:
    path = out / "hermes-func-calling-singleturn.jsonl"
    records = json.loads(_get(HERMES_URL))
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            row = {"conversations": rec["conversations"], "tools": rec.get("tools")}
            fh.write(json.dumps(row, ensure_ascii=False))
            fh.write("\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="data")
    parser.add_argument("--rows", type=int, default=2000)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for path in (fetch_smoltalk(out, args.rows), fetch_hermes(out)):
        print(f"{path}: {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
