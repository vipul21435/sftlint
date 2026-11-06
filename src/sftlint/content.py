"""Content rules on the final answer: refusal phrases and language (script) mix.

* ``refusal``: the final assistant turn opens with a refusal phrase within its first
  ``REFUSAL_WINDOW`` characters. The phrase list follows the prefix list Zou et al.
  (2023) match to decide whether a model refused, narrowed so that ordinary openings
  ("As an example", "Sorry to hear that", "I can't help but wonder") do not count.
  Refusals are not errors in themselves, so the rule is ``info`` and the report gives
  the rate over the file: a refusal rate far from what the dataset intends is the
  signal.
* ``language-mix``: the dominant script of the final answer differs from that of the
  last user turn, for example a question in Cyrillic answered in Latin script.
  Scripts approximate the Unicode Script property (UAX #24) from character names,
  since the standard library has no Script lookup. Code (fenced blocks and inline
  backticks) is removed first, and both sides need ``MIN_LETTERS`` letters.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

from sftlint.findings import Finding
from sftlint.formats import Conversation
from sftlint.rules import RULES, RuleInfo, rule

REFUSAL_WINDOW = 300
MIN_LETTERS = 20
MIN_SHARE = 0.6

_APOSTROPHE = "(?:'|’)"
_CANNOT = (
    rf"i (?:cannot|can{_APOSTROPHE}?t|can not|won{_APOSTROPHE}?t|will not|am unable to|"
    rf"{_APOSTROPHE}m unable to|am not able to|{_APOSTROPHE}m not able to)"
)
_REFUSAL_VERB = (
    r"(?:help(?! but)|assist|provide|fulfil|comply|create|generate|write|support|engage|"
    r"do that|answer)"
)
# An apology or hedge that may open the refusal clause: "I'm really sorry, I can't ...".
_SORRY = rf"(?:(?:i{_APOSTROPHE}?m|i am) (?:really |very |so |truly )?sorry|sorry|unfortunately)"

_REFUSAL = re.compile(
    r"(?i)(?:^|(?<=[.!?\n]\s)|(?<=\n)|(?<=^\W))(?:"
    rf"(?:i{_APOSTROPHE}?m|i am) (?:really |very |so |truly )?sorry,? but|"
    rf"sorry,? but i (?:can{_APOSTROPHE}?t|cannot|won{_APOSTROPHE}?t)|"
    r"i apologi[sz]e,? but|"
    rf"{_SORRY},?\s+(?:but\s+)?{_CANNOT} {_REFUSAL_VERB}|"
    rf"{_CANNOT} {_REFUSAL_VERB}|"
    r"as an ai(?: language model| assistant| model)?,|"
    rf"i{_APOSTROPHE}?m (?:just )?an ai\b|i am (?:just )?an ai\b|"
    r"as a responsible ai|"
    rf"(?:it{_APOSTROPHE}?s|it is) not appropriate (?:for me|to (?:help|assist|"
    r"provide|share|give|disclose|reveal|create|generate|write|make)\b)"
    r")"
)


def refusal_phrase(text: str, window: int = REFUSAL_WINDOW) -> str | None:
    """Return the refusal phrase found near the start of ``text``, or ``None``."""
    m = _REFUSAL.search(text.lstrip()[:window])
    return m.group(0) if m else None


@lru_cache(maxsize=65536)
def char_script(ch: str) -> str | None:
    """Approximate Unicode script of a letter, or ``None`` for non-letters.

    Han ideographs are ``Han``; hiragana and katakana (full or half width) are
    ``Kana``; modifier, superscript and subscript letters are ``None``; otherwise the
    first word of the character name (``LATIN``, ``CYRILLIC``, ``ARABIC`` ...).
    """
    if ch.isascii():
        return "Latin" if ch.isalpha() else None
    if not unicodedata.category(ch).startswith("L"):
        return None
    name = unicodedata.name(ch, "")
    if not name:
        return None
    if name.startswith("CJK"):
        return "Han"
    words = name.split()
    if words[0] in ("FULLWIDTH", "HALFWIDTH") and len(words) > 1:
        words = words[1:]
    if words[0] in ("HIRAGANA", "KATAKANA"):
        return "Kana"
    if words[0] in ("MODIFIER", "SUPERSCRIPT", "SUBSCRIPT"):
        return None  # mostly Latin or Common in UAX #24; not counted either way
    return words[0].capitalize()


_CODE = re.compile(r"```.*?(?:```|$)|`[^`\n]*`", re.DOTALL)


def script_counts(text: str) -> Counter[str]:
    """Letters per script, code removed."""
    return Counter(s for s in map(char_script, _CODE.sub(" ", text)) if s)


def dominant_script(text: str, min_letters: int = MIN_LETTERS) -> str | None:
    """The script with at least ``MIN_SHARE`` of the letters, or ``None``.

    Japanese mixes Han and kana, so a text with kana for at least a tenth of its
    letters is labelled ``Japanese`` and its Han letters are counted with it.
    """
    counts = script_counts(text)
    total = sum(counts.values())
    if total < min_letters:
        return None
    if counts["Kana"] >= 0.1 * total:
        counts["Japanese"] = counts.pop("Kana") + counts.pop("Han", 0)
    script, n = counts.most_common(1)[0]
    return script if n >= MIN_SHARE * total else None


def _final_pair(conv: Conversation) -> tuple[int, str, str] | None:
    msgs = conv.messages
    if not msgs or msgs[-1].role != "assistant" or not msgs[-1].content.strip():
        return None
    user = next((m.content for m in reversed(msgs[:-1]) if m.role == "user"), "")
    return len(msgs) - 1, user, msgs[-1].content


@rule("refusal", "info", "the final answer opens with a refusal phrase")
def check_refusal(conv: Conversation) -> list[Finding]:
    pair = _final_pair(conv)
    phrase = refusal_phrase(pair[2]) if pair else None
    if pair is None or phrase is None:
        return []
    return [Finding("refusal", RULES["refusal"].severity, conv.line, f"{phrase!r}", pair[0])]


@rule("language-mix", "info", "the final answer is in a different script from the user turn")
def check_language_mix(conv: Conversation) -> list[Finding]:
    pair = _final_pair(conv)
    if pair is None:
        return []
    turn, user, answer = pair
    a, b = dominant_script(user), dominant_script(answer)
    if a is None or b is None or a == b:
        return []
    msg = f"user turn is {a}, answer is {b}"
    return [Finding("language-mix", RULES["language-mix"].severity, conv.line, msg, turn)]


def content_stats(convs: Sequence[Conversation]) -> dict[str, Any]:
    """Refusal rate over records that end in a text answer, and the answers' scripts."""
    answered = 0
    refusals = 0
    scripts: Counter[str] = Counter()
    for conv in convs:
        pair = _final_pair(conv)
        if pair is None:
            continue
        answered += 1
        refusals += refusal_phrase(pair[2]) is not None
        scripts[dominant_script(pair[2]) or "none"] += 1
    return {
        "answered": answered,
        "refusals": refusals,
        "refusal_rate": refusals / answered if answered else 0.0,
        "answer_scripts": dict(scripts.most_common()),
    }


RULES["refusal-rate"] = RuleInfo(
    "refusal-rate", "error", "the share of answers that refuse is above --max-refusal-rate"
)


def check_refusal_rate(stats: dict[str, Any], max_rate: float | None) -> list[Finding]:
    """One dataset-level finding (line 0) when the refusal rate exceeds ``max_rate``."""
    if max_rate is None or stats["refusal_rate"] <= max_rate:
        return []
    msg = (
        f"{stats['refusals']} of {stats['answered']} answers refuse "
        f"({100 * stats['refusal_rate']:.2f} %), above {100 * max_rate:.2f} %"
    )
    return [Finding("refusal-rate", RULES["refusal-rate"].severity, 0, msg)]
