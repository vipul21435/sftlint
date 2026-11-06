"""PII and secret patterns, each with a validation step to cut false positives.

* ``pii-email``: an address-shaped string whose domain is not one of the names
  reserved for documentation and testing by RFC 2606 / RFC 6761 (``example.com``,
  ``.test``, ``.invalid``, ``localhost`` ...). Examples in instruction data mostly use
  those, and they are not personal data.
* ``pii-card``: 13 to 19 digits, optionally grouped by single spaces or hyphens, that
  start with a digit from 2 to 6 (the card-network industry identifiers), pass the
  Luhn checksum, and do not consist of one repeated digit.
* ``secret-key``: a string with a known credential prefix (cloud access key ids,
  personal access tokens, chat-app bot tokens, PEM private-key headers), or a value
  assigned to a key-like name (``api_key = ...``, ``token: ...``) that is at least 20
  characters long, mixes at least two of lower case, upper case and digits, contains
  no placeholder word (``your``, ``example``, ``xxxx`` ...), and has Shannon entropy
  of at least 3.5 bits per character.

Each distinct match is reported once per conversation, at the first turn it appears.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable

from sftlint.findings import Finding
from sftlint.formats import Conversation
from sftlint.rules import rule

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,})\b")
_RESERVED_DOMAINS = ("example.com", "example.org", "example.net", "example.edu")
_RESERVED_TLDS = (".test", ".example", ".invalid", ".localhost", ".local")

# Major industry identifiers 2-6 cover the card networks (ISO/IEC 7812); this also
# keeps ISBN-13 numbers (978/979...) and long digit runs starting with 0, 1 or 9 out.
_CARD_FIRST_DIGITS = "23456"
_CARD = re.compile(r"(?<![\d-])(?:\d[ -]?){12,18}\d(?![\d-])")

_PREFIXED_SECRETS = [
    ("cloud access key id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("access token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b")),
    ("access token", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{60,}\b")),
    ("bot token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b")),
    ("private key", re.compile(r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----")),
]
_ASSIGNED_SECRET = re.compile(
    r"(?i)\b(api[_-]?key|secret(?:[_-]?key)?|access[_-]?token|auth[_-]?token|token|"
    r"password|passwd)\b[\"']?\s*[:=]\s*[\"']?([A-Za-z0-9_\-+/=.]{20,})"
)


def luhn_valid(digits: str) -> bool:
    """Luhn checksum: double every second digit from the right, subtract 9 from
    results above 9, and require the sum to be divisible by 10."""
    if not digits.isdigit() or len(digits) < 2:
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def shannon_entropy(text: str) -> float:
    """Bits per character of the empirical character distribution of ``text``."""
    if not text:
        return 0.0
    n = len(text)
    return -sum(c / n * math.log2(c / n) for c in Counter(text).values())


def _reserved(domain: str) -> bool:
    d = domain.lower()
    return (
        d in _RESERVED_DOMAINS
        or d.endswith(tuple("." + r for r in _RESERVED_DOMAINS))
        or (d.endswith(_RESERVED_TLDS))
    )


def find_emails(text: str) -> list[str]:
    return [m.group(0) for m in _EMAIL.finditer(text) if not _reserved(m.group(1))]


def find_cards(text: str) -> list[str]:
    out = []
    for m in _CARD.finditer(text):
        raw = m.group(0)
        if re.search(r"[ -]{2}", raw) or (" " in raw and "-" in raw):
            continue
        digits = re.sub(r"\D", "", raw)
        if (
            13 <= len(digits) <= 19
            and digits[0] in _CARD_FIRST_DIGITS
            and len(set(digits)) > 1
            and luhn_valid(digits)
        ):
            out.append(raw)
    return out


def find_secrets(text: str, min_entropy: float = 3.5) -> list[str]:
    """Return a short description of each secret-like string found."""
    out = [label for label, rx in _PREFIXED_SECRETS if rx.search(text)]
    for m in _ASSIGNED_SECRET.finditer(text):
        name, value = m.group(1), m.group(2)
        if _looks_random(value) and shannon_entropy(value) >= min_entropy:
            out.append(f"high-entropy value assigned to {name!r}")
    return out


_PLACEHOLDER = re.compile(r"(?i)your|example|placeholder|replace|xxxx|dummy|sample|here")


def _looks_random(value: str) -> bool:
    """At least two of lower case, upper case and digits, and no placeholder word.
    Rules out ``YOUR_API_KEY``-style placeholders and plain identifiers."""
    classes = sum(any(f(c) for c in value) for f in (str.islower, str.isupper, str.isdigit))
    return classes >= 2 and not _PLACEHOLDER.search(value)


def _first_seen(conv: Conversation, finder: Callable[[str], list[str]]) -> list[tuple[int, str]]:
    """``(turn, value)`` for each distinct value, at the first turn it occurs."""
    seen: set[str] = set()
    out = []
    for i, m in enumerate(conv.messages):
        for value in finder(m.content):
            if value not in seen:
                seen.add(value)
                out.append((i, value))
    return out


def _mask(value: str) -> str:
    return value[:2] + "*" * max(len(value) - 4, 0) + value[-2:]


@rule("pii-email", "warning", "a turn contains an email address outside reserved domains")
def check_pii_email(conv: Conversation) -> list[Finding]:
    return [
        Finding("pii-email", "warning", conv.line, f"email address {_mask(e)}", i)
        for i, e in _first_seen(conv, find_emails)
    ]


@rule("pii-card", "warning", "a turn contains a Luhn-valid payment card number")
def check_pii_card(conv: Conversation) -> list[Finding]:
    return [
        Finding("pii-card", "warning", conv.line, f"Luhn-valid card number {_mask(c)}", i)
        for i, c in _first_seen(conv, find_cards)
    ]


@rule("secret-key", "error", "a turn contains a credential-shaped or high-entropy secret")
def check_secret_key(conv: Conversation) -> list[Finding]:
    return [
        Finding("secret-key", "error", conv.line, s, i) for i, s in _first_seen(conv, find_secrets)
    ]
