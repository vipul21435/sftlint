"""Command-line entry point."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from sftlint import __version__
from sftlint.formats import FORMATS
from sftlint.loader import load


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sftlint", description=__doc__)
    parser.add_argument("--version", action="version", version=f"sftlint {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    detect = sub.add_parser("detect", help="print the detected format of a JSONL file")
    detect.add_argument("path")
    detect.add_argument("--format", default="auto", choices=("auto", *FORMATS))
    return parser


def _cmd_detect(args: argparse.Namespace) -> int:
    data = load(args.path, args.format)
    print(f"format: {data.fmt or 'unknown'}")
    print(f"records: {data.records}")
    print(f"normalised: {len(data.conversations)}")
    print(f"load findings: {len(data.findings)}")
    return 0 if data.fmt else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "detect":
        return _cmd_detect(args)
    return 2  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
