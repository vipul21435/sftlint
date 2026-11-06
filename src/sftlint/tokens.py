"""Token counters for the length statistics.

The default, ``pretokens``, needs no download: it counts the pieces of the GPT-2
pre-tokenization pattern (a word with its leading space, a run of digits, a run of
punctuation, a run of whitespace), written with the standard ``re`` module. GPT-2's
BPE never merges across these pieces, so for that tokenizer a pretoken count is a
lower bound. Other vocabularies pre-tokenize differently (cl100k_base, for one, splits
digits into groups of three and attaches a leading punctuation mark to a word), so for
them it is an approximation; the README gives a measured comparison.

``tiktoken:<encoding>`` (for example ``tiktoken:cl100k_base``) counts real BPE tokens
when the optional ``tiktoken`` package is installed; the encoding file is downloaded
by tiktoken on first use.
"""

from __future__ import annotations

import re
from collections.abc import Callable

TokenCounter = Callable[[str], int]

# GPT-2's pattern uses \p{L} and \p{N}; [^\W\d_] is "letter" and \d is "digit" in re.
_PRETOKEN = re.compile(r"'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+")


def pretokens(text: str) -> int:
    """Number of GPT-2-style pre-tokens in ``text``."""
    return sum(1 for _ in _PRETOKEN.finditer(text))


def get_counter(spec: str = "pretokens") -> TokenCounter:
    """Return a counter for ``pretokens`` or ``tiktoken:<encoding>``.

    Raises ``ValueError`` for an unknown spec or when tiktoken is not installed.
    """
    if spec == "pretokens":
        return pretokens
    if spec.startswith("tiktoken:"):
        try:
            import tiktoken
        except ImportError as exc:
            raise ValueError("--tokenizer tiktoken:... needs: pip install tiktoken") from exc
        enc = tiktoken.get_encoding(spec.split(":", 1)[1])
        return lambda text: len(enc.encode(text, disallowed_special=()))
    raise ValueError(f"unknown tokenizer {spec!r}; use pretokens or tiktoken:<encoding>")
