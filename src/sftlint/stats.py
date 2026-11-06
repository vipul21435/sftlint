"""Per-field length statistics and robust outlier scores.

Fields are ``prompt`` (every turn before the final assistant turn, joined) and
``response`` (the final assistant turn). A turn's text includes its tool calls,
serialised as JSON, since those are tokens the model is trained on. Lengths are
counted in characters, whitespace-separated words and tokens (see
:mod:`sftlint.tokens`; the default counter is a GPT-2-style pre-tokenizer).

Outliers use the modified z-score of Iglewicz and Hoaglin on ``log1p(length)``::

    M_i = 0.6745 * (x_i - median(x)) / MAD,   MAD = median(|x_i - median(x)|)

and a record is an outlier when ``|M_i|`` exceeds a threshold (3.5 by default). The
log makes the score symmetric for lengths, which are right-skewed; the median and MAD
keep a handful of huge records from hiding each other the way a mean and standard
deviation would.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence

from sftlint.formats import Conversation, Message
from sftlint.tokens import TokenCounter, pretokens


def turn_text(m: Message) -> str:
    """The text of a turn plus each tool call as ``{"name", "arguments"}`` JSON."""
    if not m.tool_calls:
        return m.content
    calls = [
        json.dumps({"name": c.name, "arguments": c.arguments}, ensure_ascii=False, default=str)
        for c in m.tool_calls
    ]
    return "\n".join([m.content, *calls]) if m.content else "\n".join(calls)


def split_fields(conv: Conversation) -> dict[str, str]:
    """Return the ``prompt`` and ``response`` text of a conversation."""
    msgs = conv.messages
    if msgs and msgs[-1].role == "assistant":
        prompt_msgs, response = msgs[:-1], turn_text(msgs[-1])
    else:
        prompt_msgs, response = msgs, ""
    prompt = "\n".join(turn_text(m) for m in prompt_msgs)
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


def length_table(
    convs: Sequence[Conversation], count_tokens: TokenCounter | None = None
) -> dict[str, dict[str, list[int]]]:
    """Per field, the character, word and token length of every conversation (in order).

    ``count_tokens=None`` skips the token column (the outlier rule needs only chars).
    """
    units = ("chars", "words", "tokens") if count_tokens else ("chars", "words")
    table: dict[str, dict[str, list[int]]] = {
        f: {u: [] for u in units} for f in ("prompt", "response")
    }
    for conv in convs:
        for name, text in split_fields(conv).items():
            table[name]["chars"].append(len(text))
            table[name]["words"].append(len(text.split()))
            if count_tokens:
                table[name]["tokens"].append(count_tokens(text))
    return table


def length_stats(
    convs: Sequence[Conversation], count_tokens: TokenCounter | None = pretokens
) -> dict[str, dict[str, dict[str, float]]]:
    """Summary statistics of :func:`length_table`."""
    return {
        name: {unit: summary(vals) for unit, vals in units.items()}
        for name, units in length_table(convs, count_tokens).items()
    }
