"""Print the breakdowns behind the tool-rule counts for a Hermes function-calling file.

    uv run python scripts/explain_hermes.py data/hermes-func-calling-singleturn.jsonl
    uv run python scripts/explain_hermes.py data/hermes-func-calling.jsonl

Every number in the README bullets on hermes-function-calling-v1 is printed here.
The script re-reads the data with sftlint's own loader and rules, so it does not
depend on a report file (JSON reports cap the number of findings they list).

Definitions:

* a body "starts with a literal backslash-n" when its first two characters are a
  backslash and the letter n, not a newline;
* a body "single-quotes a list item" when a ``[`` or ``,`` followed by optional
  whitespace is followed by ``'`` (regex ``[\\[,]\\s*'``), the way Python's ``repr``
  writes a list of strings;
* a ``tool-args-schema`` finding is put under the cause of its first error.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter

from sftlint.loader import load
from sftlint.tools import check_tool_args_json, check_tool_args_schema, check_tool_result_match

_SINGLE_QUOTED_ITEM = re.compile(r"[\[,]\s*'")
_NAME = re.compile(r'"name":\s*"([^"]+)"')


def schema_cause(error: str) -> str:
    if error.endswith("is not one of [None]"):
        return 'value not in "enum": [null]'
    if error.endswith("is not one of []"):
        return 'value not in "enum": []'
    if m := re.search(r"expected (\w+), got null", error):
        return f"null where the schema declares a {m.group(1)}"
    if "missing required property" in error:
        return "required property missing"
    return "other: " + error.split(": ", 1)[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("path", help="Hermes function-calling JSONL file")
    path = ap.parse_args().path
    ds = load(path)

    bodies = [
        call.arguments
        for conv in ds.conversations
        for m in conv.messages
        for call in m.tool_calls
        if call.parse_error and isinstance(call.arguments, str)
    ]
    json_findings = sum(len(check_tool_args_json(c)) for c in ds.conversations)
    names = Counter(m.group(1) if (m := _NAME.search(b)) else "(no name)" for b in bodies)
    literal = [b for b in bodies if b.startswith("\\n")]
    quoted = [b for b in bodies if _SINGLE_QUOTED_ITEM.search(b)]
    parse_after, still_bad = 0, []
    for b in bodies:
        try:
            json.loads(b.replace("\\n", "\n"))
            parse_after += 1
        except json.JSONDecodeError:
            still_bad.append(b)
    print(f"{path}: {ds.records} records")
    print(f"tool-args-json findings: {json_findings}; unreadable call bodies: {len(bodies)}")
    for name, n in names.most_common():
        print(f"  function name in body {name!r}: {n}")
    print(f"  body starts with a literal backslash-n: {len(literal)}")
    print(f"  body single-quotes a list item: {len(quoted)}")
    print(f"  parses once literal \\n is replaced by a newline: {parse_after}")
    print(
        f"  still not JSON after the replacement: {len(still_bad)}, of which single-quote "
        f"a list item: {sum(bool(_SINGLE_QUOTED_ITEM.search(b)) for b in still_bad)}"
    )

    causes: Counter[str] = Counter()
    schema_records = 0
    schema_findings = 0
    for conv in ds.conversations:
        found = check_tool_args_schema(conv, max_errors=1)
        schema_records += bool(found)
        schema_findings += len(found)
        for f in found:
            first = re.sub(r" \(\+\d+ more\)$", "", f.message.split(": ", 1)[1])
            causes[schema_cause(first)] += 1
    print(f"tool-args-schema findings: {schema_findings} in {schema_records} records")
    for cause, n in causes.most_common():
        print(f"  {cause}: {n}")

    kinds: Counter[str] = Counter()
    match_records = 0
    for conv in ds.conversations:
        found = check_tool_result_match(conv)
        match_records += bool(found)
        for f in found:
            if "answers no pending call" in f.message:
                kinds["result answers no pending call"] += 1
            elif "gets no result" in f.message:
                kinds["call gets no result"] += 1
            else:
                kinds["result could not be read"] += 1
    print(f"tool-result-match findings: {sum(kinds.values())} in {match_records} records")
    for kind, n in kinds.most_common():
        print(f"  {kind}: {n}")


if __name__ == "__main__":
    main()
