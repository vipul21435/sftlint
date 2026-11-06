"""Tabulate the dominant script sftlint assigns to answers, per labelled language.

Reads a JSONL file whose records carry a ``language`` field (the Aya test split as
written by fetch_data.py) and prints, per language, how many answers get each
dominant script and how many records ``language-mix`` flags.

    uv run python scripts/script_by_language.py data/aya-test.jsonl
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict

from sftlint.content import check_language_mix, dominant_script
from sftlint.loader import load


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path")
    args = parser.parse_args()
    data = load(args.path)
    scripts: dict[str, Counter[str]] = defaultdict(Counter)
    flagged: Counter[str] = Counter()
    for conv in data.conversations:
        lang = str((conv.raw or {}).get("language", "unknown"))
        scripts[lang][dominant_script(conv.messages[-1].content) or "none"] += 1
        flagged[lang] += bool(check_language_mix(conv))
    print("| language | records | dominant script of answers | language-mix |")
    print("|---|---:|---|---:|")
    for lang in sorted(scripts):
        n = sum(scripts[lang].values())
        cells = ", ".join(f"{s} {c}" for s, c in scripts[lang].most_common())
        print(f"| {lang} | {n} | {cells} | {flagged[lang]} |")


if __name__ == "__main__":
    main()
