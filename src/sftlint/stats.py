"""Per-field length statistics and robust outlier scores.

Fields are ``prompt`` (every turn before the final assistant turn, joined) and
``response`` (the final assistant turn). Lengths are counted in characters and in
whitespace-separated words; words are a cheap stand-in for tokens, not a tokenizer.

Outliers use the modified z-score of Iglewicz and Hoaglin on ``log1p(length)``::

    M_i = 0.6745 * (x_i - median(x)) / MAD,   MAD = median(|x_i - median(x)|)

and a record is an outlier when ``|M_i|`` exceeds a threshold (3.5 by default). The
log makes the score symmetric for lengths, which are right-skewed; the median and MAD
keep a handful of huge records from hiding each other the way a mean and standard
deviation would.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from sftlint.formats import Conversation


def split_fields(conv: Conversation) -> dict[str, str]:
    """Return the ``prompt`` and ``response`` text of a conversation."""
    msgs = conv.messages
    if msgs and msgs[-1].role == "assistant":
        prompt_msgs, response = msgs[:-1], msgs[-1].content
    else:
        prompt_msgs, response = msgs, ""
    prompt = "\n".join(m.content for m in prompt_msgs)
    return {"prompt": prompt, "response": response}


def quantile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile of an already sorted sequence."""
    if not sorted_values:
        return math.nan
    pos = (len(sorted_values) - 1) * q
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def summary(values: Sequence[float]) -> dict[str, float]:
    """count, mean, min, p50, p90, p99 and max of ``values``."""
    s = sorted(values)
    if not s:
        return {"count": 0}
    return {
        "count": len(s),
        "mean": sum(s) / len(s),
        "min": s[0],
        "p50": quantile(s, 0.5),
        "p90": quantile(s, 0.9),
        "p99": quantile(s, 0.99),
        "max": s[-1],
    }


def modified_z(values: Sequence[float]) -> list[float]:
    """Iglewicz-Hoaglin modified z-scores. All zeros when MAD is zero."""
    if not values:
        return []
    med = quantile(sorted(values), 0.5)
    mad = quantile(sorted(abs(v - med) for v in values), 0.5)
    if mad == 0:
        return [0.0] * len(values)
    return [0.6745 * (v - med) / mad for v in values]


def length_table(convs: Sequence[Conversation]) -> dict[str, dict[str, list[int]]]:
    """Per field, the character and word length of every conversation (in order)."""
    table: dict[str, dict[str, list[int]]] = {
        f: {"chars": [], "words": []} for f in ("prompt", "response")
    }
    for conv in convs:
        for name, text in split_fields(conv).items():
            table[name]["chars"].append(len(text))
            table[name]["words"].append(len(text.split()))
    return table


def length_stats(convs: Sequence[Conversation]) -> dict[str, dict[str, dict[str, float]]]:
    """Summary statistics of :func:`length_table`."""
    return {
        name: {unit: summary(vals) for unit, vals in units.items()}
        for name, units in length_table(convs).items()
    }
