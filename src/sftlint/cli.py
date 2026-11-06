"""sftlint: lint chat, SFT and tool-calling datasets in JSONL."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from sftlint import __version__
from sftlint.findings import severity_rank
from sftlint.fix import apply_fixes
from sftlint.formats import FORMATS
from sftlint.lint import lint
from sftlint.loader import load
from sftlint.report import to_json, to_markdown
from sftlint.rules import RULES
from sftlint.tokens import get_counter


def _ids(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sftlint", description=__doc__)
    parser.add_argument("--version", action="version", version=f"sftlint {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="lint a JSONL file")
    check.add_argument("path")
    check.add_argument("--format", default="auto", choices=("auto", *FORMATS))
    check.add_argument("--select", type=_ids, help="comma-separated rule ids to run")
    check.add_argument("--ignore", type=_ids, help="comma-separated rule ids to skip")
    check.add_argument("--json", metavar="FILE", help="write the JSON report ('-' for stdout)")
    check.add_argument(
        "--markdown", metavar="FILE", help="write the Markdown report ('-' for stdout)"
    )
    check.add_argument(
        "--fail-on",
        default="error",
        choices=("error", "warning", "info", "never"),
        help="exit 1 when a finding of this severity or worse exists (default: error)",
    )
    check.add_argument(
        "--outlier-z", type=float, default=3.5, help="modified z threshold (default 3.5)"
    )
    check.add_argument(
        "--max-refusal-rate",
        type=float,
        metavar="RATE",
        help="add a refusal-rate error when this share of answers refuse (0 to 1; off by default)",
    )
    check.add_argument(
        "--tokenizer",
        default="pretokens",
        help="token counter for length stats: pretokens (default) or tiktoken:<encoding>",
    )
    check.add_argument(
        "--fix", metavar="OUT", help="write a copy with what --fix-rules flag dropped or trimmed"
    )
    check.add_argument(
        "--fix-rules", type=_ids, help="comma-separated rule ids whose findings --fix acts on"
    )
    check.add_argument(
        "--max-examples", type=int, default=20, help="findings listed in reports (default 20)"
    )

    sub.add_parser("rules", help="list rule ids, default severities and summaries")

    detect = sub.add_parser("detect", help="print the detected format of a JSONL file")
    detect.add_argument("path")
    detect.add_argument("--format", default="auto", choices=("auto", *FORMATS))
    return parser


def _write(target: str, text: str) -> None:
    if target == "-":
        sys.stdout.write(text)
    else:
        with open(target, "w", encoding="utf-8") as fh:
            fh.write(text)


def _cmd_check(args: argparse.Namespace) -> int:
    if bool(args.fix) != bool(args.fix_rules):
        print("sftlint: --fix and --fix-rules go together", file=sys.stderr)
        return 2
    try:
        counter = get_counter(args.tokenizer)
        data = load(args.path, args.format)
        report = lint(
            data,
            args.path,
            args.select,
            args.ignore,
            args.outlier_z,
            counter,
            args.tokenizer,
            args.max_refusal_rate,
        )
        if args.fix:
            unknown = set(args.fix_rules) - set(RULES)
            if unknown:
                raise ValueError("unknown rule id: " + ", ".join(sorted(unknown)))
            fixed = apply_fixes(args.path, data, report.findings, args.fix_rules, args.fix)
    except (OSError, ValueError) as exc:
        print(f"sftlint: {exc}", file=sys.stderr)
        return 2
    if args.json:
        _write(args.json, to_json(report, args.max_examples))
    if args.markdown:
        _write(args.markdown, to_markdown(report, args.max_examples))
    if "-" not in (args.json, args.markdown):
        sev = report.severity_counts()
        print(
            f"{args.path}: {report.records} records, format {report.fmt or 'unknown'}, "
            f"{report.flagged_records()} with findings "
            f"({sev['error']} errors, {sev['warning']} warnings, {sev['info']} info)"
        )
        for rule, n in report.counts().items():
            print(f"  {rule:<18} {RULES[rule].severity:<8} {n}")
    if args.fix:
        print(f"fix: {fixed.summary()} -> {args.fix}", file=sys.stderr)
    worst = report.worst()
    if args.fail_on == "never" or worst is None:
        return 0
    return 1 if severity_rank(worst) >= severity_rank(args.fail_on) else 0


def _cmd_rules() -> int:
    for info in RULES.values():
        print(f"{info.id:<18} {info.severity:<8} {info.summary}")
    return 0


def _cmd_detect(args: argparse.Namespace) -> int:
    data = load(args.path, args.format)
    print(f"format: {data.fmt or 'unknown'}")
    print(f"records: {data.records}")
    print(f"normalised: {len(data.conversations)}")
    print(f"load findings: {len(data.findings)}")
    return 0 if data.fmt else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "check":
        return _cmd_check(args)
    if args.command == "rules":
        return _cmd_rules()
    return _cmd_detect(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
