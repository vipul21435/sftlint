"""Download the small public slices used for the results in the README.

* smoltalk, config smol-magpie-ultra, train split, first --rows rows, through the
  Hugging Face datasets-server rows API (100 rows per request). Licence: Apache 2.0
  for this subset, per https://huggingface.co/datasets/HuggingFaceTB/smoltalk
* hermes-function-calling-v1, files func-calling-singleturn.json (about 18 MB) and
  func-calling.json (about 21 MB, multi-turn with tool results). Licence: Apache 2.0, per https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1

* aya_dataset, test split (1,750 rows in seven languages), through the same rows API,
  written as prompt/completion records with the language label kept. Licence: Apache
  2.0, per https://huggingface.co/datasets/CohereLabs/aya_dataset
* ultrachat_200k, train_sft split, first --rows rows, through the rows API; only the
  ``messages`` list is kept (``prompt_id`` is dropped). Licence: MIT, per
  https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k
* glaive-function-calling-v2, train split, first --rows rows, through the rows API.
  Each row is a ``system`` string (an instruction followed by the declared functions
  as JSON objects) and a ``chat`` string of ``USER:``, ``ASSISTANT:`` and ``FUNCTION
  RESPONSE:`` turns; ``glaive_to_messages`` rewrites it as a ``messages`` record with a
  ``tools`` list (see its docstring for what it assumes). Licence: Apache 2.0, per
  https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2

Writes JSONL files into --out (default: ./data, which .gitignore excludes). Only the
fields the linter reads are kept; per-record ids are dropped.

    uv run python scripts/fetch_data.py --out data --rows 2000
    uv run python scripts/fetch_data.py --out data --rows 2000 --only ultrachat,glaive
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import TextIO

ROWS_API = "https://datasets-server.huggingface.co/rows"
HERMES_BASE = "https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1/resolve/main"
HERMES_FILES = {
    "func-calling-singleturn.json": "hermes-func-calling-singleturn.jsonl",
    "func-calling.json": "hermes-func-calling.jsonl",
}


ATTEMPTS = 7
MAX_WAIT = 60.0


def _retry_after(exc: urllib.error.HTTPError) -> float | None:
    """Seconds from a numeric Retry-After header, or None."""
    value = exc.headers.get("Retry-After") if exc.headers else None
    try:
        return max(0.0, float(value)) if value is not None else None
    except ValueError:
        return None


def _get(url: str, sleep=time.sleep) -> bytes:
    """GET with retries. HTTP 429 (rate limited) waits for Retry-After when the server
    sends one, else backs off 2, 4, 8 ... seconds up to a minute; other network errors
    back off the same way. Client errors other than 429 are raised at once."""
    for attempt in range(ATTEMPTS):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if attempt == ATTEMPTS - 1 or (400 <= exc.code < 500 and exc.code != 429):
                raise
            wait = _retry_after(exc) if exc.code == 429 else None
            sleep(min(MAX_WAIT, wait if wait is not None else 2.0 ** (attempt + 1)))
        except OSError:
            if attempt == ATTEMPTS - 1:
                raise
            sleep(min(MAX_WAIT, 2.0 ** (attempt + 1)))
    raise RuntimeError("unreachable")


@contextlib.contextmanager
def _atomic_write(path: Path) -> Iterator[TextIO]:
    """Write to ``path.part`` and rename it to ``path`` only once the block finishes,
    so a failed download never leaves a partial file under the final name."""
    tmp = path.with_name(path.name + ".part")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            yield fh
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


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
    with _atomic_write(path) as fh:
        for row in _rows("HuggingFaceTB/smoltalk", "smol-magpie-ultra", "train", rows):
            fh.write(json.dumps({"messages": row["messages"]}, ensure_ascii=False))
            fh.write("\n")
    return path


def fetch_aya(out: Path, rows: int = 1750) -> Path:
    """The Aya test split as prompt/completion; the per-row annotator id is dropped."""
    path = out / "aya-test.jsonl"
    with _atomic_write(path) as fh:
        for row in _rows("CohereLabs/aya_dataset", "default", "test", rows):
            rec = {"prompt": row["inputs"], "completion": row["targets"]}
            rec["language"] = row["language"]
            fh.write(json.dumps(rec, ensure_ascii=False))
            fh.write("\n")
    return path


def fetch_hermes(out: Path, source: str) -> Path:
    path = out / HERMES_FILES[source]
    records = json.loads(_get(f"{HERMES_BASE}/{source}"))
    with _atomic_write(path) as fh:
        for rec in records:
            row = {"conversations": rec["conversations"], "tools": rec.get("tools")}
            fh.write(json.dumps(row, ensure_ascii=False))
            fh.write("\n")
    return path


def fetch_ultrachat(out: Path, rows: int) -> Path:
    path = out / "ultrachat-sft.jsonl"
    with _atomic_write(path) as fh:
        for row in _rows("HuggingFaceH4/ultrachat_200k", "default", "train_sft", rows):
            fh.write(json.dumps({"messages": row["messages"]}, ensure_ascii=False))
            fh.write("\n")
    return path


_GLAIVE_TURN = re.compile(r"(?:^|\n+)(USER|ASSISTANT|FUNCTION RESPONSE): ")
_GLAIVE_ROLE = {"USER": "user", "ASSISTANT": "assistant", "FUNCTION RESPONSE": "tool"}
# The call layout the dataset uses: arguments as a JSON string in single quotes.
_GLAIVE_CALL = re.compile(r'\{"name":\s*"([^"]*)",\s*"arguments":\s*\'(.*)\'\}', re.DOTALL)
_GLAIVE_NAME = re.compile(r'"name":\s*"([^"]*)"')
_END = "<|endoftext|>"


def _glaive_tools(system: str) -> tuple[str, list[dict]]:
    """Split a glaive system string into its instruction text and the JSON objects
    that follow it, read one after another with ``raw_decode``. Text that does not
    decode ends the list."""
    text = system.removeprefix("SYSTEM: ")
    start = text.find("{")
    if start < 0:
        return text.strip(), []
    decoder, tools, pos = json.JSONDecoder(), [], start
    while pos < len(text):
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos == len(text):
            break
        try:
            obj, pos = decoder.raw_decode(text, pos)
        except json.JSONDecodeError:
            break
        if isinstance(obj, dict):
            tools.append({"type": "function", "function": obj})
    return text[:start].strip(), tools


def _glaive_call(body: str) -> tuple[dict, str]:
    """Read the text after ``<functioncall>`` as a call. Returns the call and how it was
    read: ``quoted`` (the dataset's ``"arguments": '<json>'`` layout; the quoted
    string, with each ``\\'`` read as a quote as in a Python string literal, is kept as
    the arguments, parsed or not), ``json`` (the body is a JSON
    object) or ``raw`` (neither; the whole body is kept as the arguments, so
    ``tool-args-json`` reports it)."""
    body = body.strip()
    match = _GLAIVE_CALL.fullmatch(body)
    if match:
        args = match.group(2).replace("\\'", "'")
        return {"name": match.group(1), "arguments": args}, "quoted"
    try:
        obj = json.loads(body)
    except json.JSONDecodeError:
        obj = None
    if isinstance(obj, dict) and isinstance(obj.get("name"), str):
        args = obj.get("arguments")
        args = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
        return {"name": obj["name"], "arguments": args}, "json"
    name = _GLAIVE_NAME.search(body)
    return {"name": name.group(1) if name else None, "arguments": body}, "raw"


def glaive_to_messages(row: dict, stats: dict | None = None) -> dict:
    """Rewrite one glaive row as ``{"messages": [...], "tools": [...]}``.

    Turns are split at ``USER:``, ``ASSISTANT:`` and ``FUNCTION RESPONSE:`` at the start
    of the chat or after a newline. A trailing ``<|endoftext|>`` is removed from an
    assistant turn (an assistant turn without one is counted in ``stats`` as
    ``no_end``). An assistant turn holding ``<functioncall>`` becomes the text before
    it plus one tool call (see ``_glaive_call``); FUNCTION RESPONSE turns become tool
    turns with no id, matched to calls by position. No text is otherwise changed.
    """
    stats = stats if stats is not None else {}
    instruction, tools = _glaive_tools(row.get("system") or "")
    messages = [{"role": "system", "content": instruction}] if instruction else []
    chat = row.get("chat") or ""
    parts = _GLAIVE_TURN.split(chat)
    for label, text in zip(parts[1::2], parts[2::2], strict=True):
        role = _GLAIVE_ROLE[label]
        text = text.strip()
        msg: dict = {"role": role}
        if role == "assistant":
            if text.endswith(_END):
                text = text[: -len(_END)].rstrip()
            else:
                stats["no_end"] = stats.get("no_end", 0) + 1
            head, sep, body = text.partition("<functioncall>")
            if sep:
                call, how = _glaive_call(body)
                stats[how] = stats.get(how, 0) + 1
                msg["tool_calls"] = [{"type": "function", "function": call}]
                text = head.strip()
        msg["content"] = text
        messages.append(msg)
    return {"messages": messages, "tools": tools}


def fetch_glaive(out: Path, rows: int) -> tuple[Path, dict]:
    path, stats = out / "glaive-fc-v2.jsonl", {}
    with _atomic_write(path) as fh:
        for row in _rows("glaiveai/glaive-function-calling-v2", "default", "train", rows):
            fh.write(json.dumps(glaive_to_messages(row, stats), ensure_ascii=False))
            fh.write("\n")
    return path, stats


SOURCES = ("smoltalk", "hermes", "aya", "ultrachat", "glaive")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="data")
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument(
        "--only",
        default=",".join(SOURCES),
        help=f"comma-separated subset of {','.join(SOURCES)} (default: all)",
    )
    args = parser.parse_args()
    only = [name.strip() for name in args.only.split(",") if name.strip()]
    unknown = sorted(set(only) - set(SOURCES))
    if unknown:
        parser.error(f"unknown source(s): {', '.join(unknown)}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    if "smoltalk" in only:
        paths.append(fetch_smoltalk(out, args.rows))
    if "hermes" in only:
        paths += [fetch_hermes(out, f) for f in HERMES_FILES]
    if "aya" in only:
        paths.append(fetch_aya(out))
    if "ultrachat" in only:
        paths.append(fetch_ultrachat(out, args.rows))
    if "glaive" in only:
        path, stats = fetch_glaive(out, args.rows)
        paths.append(path)
        print(f"glaive calls read: {json.dumps(stats, sort_keys=True)}")
    for path in paths:
        print(f"{path}: {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
