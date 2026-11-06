"""Download the small public slices used for the results in the README.

* smoltalk, config smol-magpie-ultra, train split, first --rows rows, through the
  Hugging Face datasets-server rows API (100 rows per request). Licence: Apache 2.0
  for this subset, per https://huggingface.co/datasets/HuggingFaceTB/smoltalk
* hermes-function-calling-v1, files func-calling-singleturn.json (about 18 MB) and
  func-calling.json (about 21 MB, multi-turn with tool results). Licence: Apache 2.0, per https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1

* aya_dataset, test split (1,750 rows in seven languages), through the same rows API,
  written as prompt/completion records with the language label kept. Licence: Apache
  2.0, per https://huggingface.co/datasets/CohereLabs/aya_dataset

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
HERMES_BASE = "https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1/resolve/main"
HERMES_FILES = {
    "func-calling-singleturn.json": "hermes-func-calling-singleturn.jsonl",
    "func-calling.json": "hermes-func-calling.jsonl",
}


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


def _rows(dataset: str, config: str, split: str, rows: int):
    """Yield up to ``rows`` rows of a split, 100 per request."""
    for offset in range(0, rows, 100):
        query = urllib.parse.urlencode(
            {
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": offset,
                "length": min(100, rows - offset),
            }
        )
        page = json.loads(_get(f"{ROWS_API}?{query}"))
        if not page["rows"]:
            return
        for item in page["rows"]:
            yield item["row"]


def fetch_smoltalk(out: Path, rows: int) -> Path:
    path = out / "smoltalk-magpie-ultra.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for row in _rows("HuggingFaceTB/smoltalk", "smol-magpie-ultra", "train", rows):
            fh.write(json.dumps({"messages": row["messages"]}, ensure_ascii=False))
            fh.write("\n")
    return path


def fetch_aya(out: Path, rows: int = 1750) -> Path:
    """The Aya test split as prompt/completion; the per-row annotator id is dropped."""
    path = out / "aya-test.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for row in _rows("CohereLabs/aya_dataset", "default", "test", rows):
            rec = {"prompt": row["inputs"], "completion": row["targets"]}
            rec["language"] = row["language"]
            fh.write(json.dumps(rec, ensure_ascii=False))
            fh.write("\n")
    return path


def fetch_hermes(out: Path, source: str) -> Path:
    path = out / HERMES_FILES[source]
    records = json.loads(_get(f"{HERMES_BASE}/{source}"))
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
    paths = [fetch_smoltalk(out, args.rows)] + [fetch_hermes(out, f) for f in HERMES_FILES]
    paths.append(fetch_aya(out))
    for path in paths:
        print(f"{path}: {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
