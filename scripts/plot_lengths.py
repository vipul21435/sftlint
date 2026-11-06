"""Plot prompt and response length histograms with the length-outlier cut-offs.

Reads a JSONL file through sftlint, bins log10(1 + chars) for each field, and marks
where |modified z| = threshold falls, so the flagged tails are visible.

    uv run --with matplotlib python scripts/plot_lengths.py \
        data/smoltalk-magpie-ultra.jsonl docs/lengths-smoltalk.png
"""

from __future__ import annotations

import argparse
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sftlint.loader import load  # noqa: E402
from sftlint.stats import length_table, quantile  # noqa: E402

BAR = "#2a78d6"
INK = "#0b0b0b"
MUTED = "#52514e"


def cutoffs(chars: list[int], threshold: float) -> tuple[float, float]:
    """Character lengths where the modified z-score on log1p(length) reaches +-threshold."""
    x = sorted(math.log1p(c) for c in chars)
    med = quantile(x, 0.5)
    mad = quantile(sorted(abs(v - med) for v in x), 0.5)
    half = threshold * mad / 0.6745
    return math.expm1(med - half), math.expm1(med + half)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path")
    parser.add_argument("out")
    parser.add_argument("--threshold", type=float, default=3.5)
    args = parser.parse_args()

    data = load(args.path)
    table = length_table(data.conversations)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=False)
    for ax, field in zip(axes, ("prompt", "response"), strict=True):
        chars = table[field]["chars"]
        lo, hi = cutoffs(chars, args.threshold)
        vals = [math.log10(1 + c) for c in chars]
        ax.hist(vals, bins=50, color=BAR, edgecolor="white", linewidth=0.6)
        for edge, label in ((lo, "short cut-off"), (hi, "long cut-off")):
            if edge > 0:
                ax.axvline(math.log10(1 + edge), color=MUTED, linestyle="--", linewidth=1.2)
                ax.annotate(
                    f"{label}\n{edge:,.0f} chars",
                    xy=(math.log10(1 + edge), 1),
                    xycoords=("data", "axes fraction"),
                    xytext=(4, -4),
                    textcoords="offset points",
                    va="top",
                    fontsize=8,
                    color=MUTED,
                )
        flagged = sum(c < lo or c > hi for c in chars)
        ax.set_title(f"{field}: {len(chars):,} records, {flagged} outside", color=INK, fontsize=10)
        ax.set_xlabel("log10(1 + characters)", color=MUTED)
        ax.set_ylabel("records", color=MUTED)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED)
        ax.grid(axis="y", color="#e6e5e1", linewidth=0.6)
        ax.set_axisbelow(True)
    fig.suptitle(f"Length distribution, |modified z| > {args.threshold}", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
