"""PII and secret patterns, each with a validation step to cut false positives.

* ``pii-email``: an address-shaped string whose domain is not one of the names
  reserved for documentation and testing by RFC 2606 / RFC 6761 (``example.com``,
  ``.test``, ``.invalid``, ``localhost`` ...). Examples in instruction data mostly use
  those, and they are not personal data. Placeholder local parts (``your-email@``,
  ``recipient@``) are skipped for the same reason.
* ``pii-card``: 13 to 19 digits, solid or grouped by single spaces or hyphens into
  fours (or the 4-6-5 and 4-6-4 layouts), that
  start with a digit from 2 to 6 (the card-network industry identifiers), pass the
  Luhn checksum, and do not consist of one repeated digit. A digit run that touches
  a letter, digit, underscore or hyphen is skipped as part of an identifier.
* ``pii-iban``: an International Bank Account Number: a country code, two check
  digits and an account part, written solid or in groups of four, whose length is the
  one listed for that country and whose ISO 7064 mod-97 check gives 1.
* ``secret-key``: a string with a known credential prefix (cloud access key ids and
  API keys, personal access tokens, chat-app bot tokens, hosted-model API keys,
  model-hub tokens, live payment API keys, PEM private-key headers), or a value
  assigned to a key-like name (``api_key = ...``, ``token: ...``) that is at least 20
  characters long, is not a name (an attribute path such as ``settings.API_KEY`` or an
  environment-variable name such as ``PAYMENTS_API_KEY``), mixes at least two of
  lower case, upper case and digits, contains no placeholder word (``your``,
  ``example``, ``xxxx`` ...), and has Shannon entropy of at least 3.5 bits per
  character.

Each distinct match is reported once per conversation, at the first turn it appears;
card numbers are compared by their digits only.
Tool-call arguments are scanned along with the text of each turn.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable

from sftlint.findings import Finding
from sftlint.formats import Conversation
from sftlint.rules import rule
from sftlint.stats import turn_text

# The look-behind starts a match only at the start of a run of local-part characters,
# which keeps the scan linear on long dotted runs with no "@".
_EMAIL = re.compile(
    r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,})\b"
)
_RESERVED_DOMAINS = ("example.com", "example.org", "example.net", "example.edu")
_RESERVED_TLDS = (".test", ".example", ".invalid", ".localhost", ".local")

# Major industry identifiers 2-6 cover the card networks (ISO/IEC 7812); this also
# keeps ISBN-13 numbers (978/979...) and long digit runs starting with 0, 1 or 9 out.
_CARD_FIRST_DIGITS = "23456"
# A digit run glued to letters or underscores (a hex object id, a hash, a variable
# name) is part of an identifier, not a card number written out.
_CARD = re.compile(r"(?<![\w-])(?:\d[ -]?){12,18}\d(?![\w-])")

# IBAN length per country code, from the format table on the Wikipedia IBAN page.
_IBAN_LENGTHS = {
    cc: int(n)
    for cc, n in (
        item.split(":")
        for item in (
            "AL:28 AD:24 AT:20 AZ:28 BH:22 BY:28 BE:16 BA:20 BR:29 BG:22 BI:27 CR:22 HR:21 "
            "CY:28 CZ:24 DK:18 DJ:27 DO:28 TL:23 EG:29 SV:28 EE:20 FK:18 FO:18 FI:18 FR:27 "
            "GE:22 DE:22 GI:23 GR:27 GL:18 GT:28 HN:28 HU:28 IS:26 IQ:23 IE:22 IL:23 IT:27 "
            "JO:30 KZ:20 XK:20 KW:30 LV:21 LB:28 LY:25 LI:21 LT:20 LU:20 MT:31 MR:27 MU:30 "
            "MC:27 MD:24 MN:20 ME:22 NL:18 NI:28 MK:19 NO:15 OM:23 PK:24 PS:29 PL:28 PT:25 "
            "QA:29 RO:24 RU:33 LC:32 SM:27 ST:25 SA:24 RS:22 SC:31 SK:24 SI:19 SO:23 ES:24 "
            "SD:18 SE:24 CH:21 TN:24 TR:26 UA:29 AE:23 GB:22 VA:22 VG:24 YE:30"
        ).split()
    )
}
_IBAN = re.compile(r"(?<![\w-])[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,3})?(?![\w-])")

# Prefix shapes follow the public gitleaks rule set (config/gitleaks.toml), simplified:
# the fixed prefix and a minimum length are enough to keep false positives rare.
_PREFIXED_SECRETS = [
    ("cloud access key id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("cloud API key", re.compile(r"\bAIza[\w-]{35}(?![\w-])")),
    ("access token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b")),
    ("access token", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{60,}\b")),
    ("bot token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b")),
    ("model API key", re.compile(r"\bsk-ant-(?:api|admin)\d\d-[A-Za-z0-9_-]{80,}")),
    (
        "model API key",
        re.compile(r"\bsk-(?:proj-|svcacct-|admin-)?[A-Za-z0-9_-]{20,}T3BlbkFJ[A-Za-z0-9_-]{20,}"),
    ),
    ("model-hub access token", re.compile(r"\bhf_[A-Za-z]{34}\b")),
    ("live payment API key", re.compile(r"\b(?:sk|rk)_live_[A-Za-z0-9]{10,99}\b")),
    ("private key", re.compile(r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----")),
]
_ASSIGNED_SECRET = re.compile(
    r"(?i)\b(api[_-]?key|secret(?:[_-]?key)?|access[_-]?token|auth[_-]?token|token|"
    r"password|passwd)\b[\"']?\s*[:=]\s*[\"']?([^\s\"'`,;()\[\]{}<>]{20,})"
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


_PLACEHOLDER_LOCAL = re.compile(
    r"(?i)your|example|sample|placeholder|recipient|sender|someone|username"
)


_SSH_REMOTE = re.compile(r":[\w~./-]")


def find_emails(text: str) -> list[str]:
    """Addresses outside reserved domains whose local part is not a placeholder
    word such as ``your-email`` or ``recipient``."""
    out = []
    for m in _EMAIL.finditer(text):
        local = m.group(0).split("@", 1)[0]
        if _SSH_REMOTE.match(text, m.end()):
            continue  # git@github.com:org/repo.git is an SSH remote, not an address
        if not _reserved(m.group(1)) and not _PLACEHOLDER_LOCAL.search(local):
            out.append(m.group(0))
    return out


def _card_grouping(raw: str) -> bool:
    """Solid digits, groups of four (the last may be shorter), or the 4-6-5 / 4-6-4
    layouts of 15- and 14-digit cards. Rules out single digits written with spaces."""
    lens = [len(g) for g in re.split(r"[ -]", raw)]
    if len(lens) == 1:
        return True
    return lens in ([4, 6, 5], [4, 6, 4]) or (all(n == 4 for n in lens[:-1]) and 1 <= lens[-1] <= 4)


def find_cards(text: str) -> list[str]:
    """Digit runs in ``text`` that pass the card checks (first digit, length, Luhn)."""
    out = []
    for m in _CARD.finditer(text):
        raw = m.group(0)
        if re.search(r"[ -]{2}", raw) or (" " in raw and "-" in raw):
            continue
        if not _card_grouping(raw):
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


def iban_valid(iban: str) -> bool:
    """ISO 7064 mod 97-10 as IBAN uses it: move the first four characters to the end,
    replace each letter by two digits (A = 10 ... Z = 35), and require the number to
    leave remainder 1 on division by 97. The length must match the country's."""
    code = iban.replace(" ", "")
    if _IBAN_LENGTHS.get(code[:2]) != len(code) or not code.isalnum() or not code.isascii():
        return False
    rearranged = code[4:] + code[:4]
    return int("".join(str(int(ch, 36)) for ch in rearranged)) % 97 == 1


def find_ibans(text: str) -> list[str]:
    """Each IBAN in ``text`` that passes `iban_valid`, with grouping spaces removed."""
    return [m.group(0).replace(" ", "") for m in _IBAN.finditer(text) if iban_valid(m.group(0))]


def find_secrets(text: str, min_entropy: float = 3.5) -> list[tuple[str, str]]:
    """Return ``(description, matched value)`` for each secret-like string found."""
    out = [
        (label, m.group(0))
        for label, rx in _PREFIXED_SECRETS
        for m in rx.finditer(text)
        if not _PLACEHOLDER.search(m.group(0))  # documentation keys such as ...EXAMPLE
    ]
    for m in _ASSIGNED_SECRET.finditer(text):
        name, value = m.group(1), m.group(2)
        if _NAME_VALUE.fullmatch(value):
            continue
        if _looks_random(value) and shannon_entropy(value) >= min_entropy:
            out.append((f"high-entropy value assigned to {name!r}", value))
    return out


# Code that reads a key instead of holding one: an attribute path rooted at a lower-case
# name (``settings.API_KEY``, ``self.cleaned_data.Password1``) or an upper-case
# environment-variable name with underscores (``PAYMENTS_API_KEY_2024``).
_NAME_VALUE = re.compile(r"[a-z_][a-z0-9_]*(?:\.[A-Za-z_]\w*)+|[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+")

_PLACEHOLDER = re.compile(r"(?i)your|example|placeholder|replace|xxxx|dummy|sample|here")


def _looks_random(value: str) -> bool:
    """At least two of lower case, upper case and digits, and no placeholder word.
    Rules out ``YOUR_API_KEY``-style placeholders and plain identifiers."""
    classes = sum(any(f(c) for c in value) for f in (str.islower, str.isupper, str.isdigit))
    return classes >= 2 and not _PLACEHOLDER.search(value)


def _first_seen(
    conv: Conversation,
    finder: Callable[[str], list[str]],
    key: Callable[[str], str] = lambda v: v,
) -> list[tuple[int, str]]:
    """``(turn, value)`` for each distinct value, at the first turn it occurs. Values
    with the same ``key`` (a card number's digits) count as one."""
    seen: set[str] = set()
    out = []
    for i, m in enumerate(conv.messages):
        for value in finder(turn_text(m)):
            if key(value) not in seen:
                seen.add(key(value))
                out.append((i, value))
    return out


def _digits(value: str) -> str:
    return re.sub(r"\D", "", value)


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
        for i, c in _first_seen(conv, find_cards, _digits)
    ]


@rule("pii-iban", "warning", "a turn contains an IBAN that passes its mod-97 check")
def check_pii_iban(conv: Conversation) -> list[Finding]:
    return [
        Finding("pii-iban", "warning", conv.line, f"IBAN {_mask(b)}", i)
        for i, b in _first_seen(conv, find_ibans)
    ]


@rule("secret-key", "error", "a turn contains a credential-shaped or high-entropy secret")
def check_secret_key(conv: Conversation) -> list[Finding]:
    seen: set[str] = set()
    out = []
    for i, m in enumerate(conv.messages):
        for label, value in find_secrets(turn_text(m)):
            if value not in seen:
                seen.add(value)
                message = label if label == "private key" else f"{label} {_mask(value)}"
                out.append(Finding("secret-key", "error", conv.line, message, i))
    return out
