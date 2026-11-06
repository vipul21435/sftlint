import random
import string

import pytest

from sftlint.formats import normalize
from sftlint.pii import (
    check_pii_card,
    check_pii_email,
    check_secret_key,
    find_cards,
    find_emails,
    find_secrets,
    luhn_valid,
    shannon_entropy,
)

RNG = random.Random(7)


def luhn_complete(prefix: str) -> str:
    """Append the check digit that makes ``prefix`` Luhn-valid."""
    for d in "0123456789":
        if luhn_valid(prefix + d):
            return prefix + d
    raise AssertionError


def rand(alphabet: str, n: int) -> str:
    return "".join(RNG.choice(alphabet) for _ in range(n))


def test_luhn_known_values():
    assert luhn_valid("79927398713")  # the worked example on the Luhn algorithm page
    assert not luhn_valid("79927398710")
    assert not luhn_valid("12a4") and not luhn_valid("7")


def test_luhn_detects_every_single_digit_error():
    number = luhn_complete("4" + rand(string.digits, 14))
    for i in range(len(number)):
        for d in string.digits:
            if d != number[i]:
                assert not luhn_valid(number[:i] + d + number[i + 1 :])


def test_find_cards_grouping_and_rejections():
    card = luhn_complete("5" + rand(string.digits, 14))
    grouped = " ".join(card[i : i + 4] for i in range(0, 16, 4))
    assert find_cards(f"pay with {card} today") == [card]
    assert find_cards(f"card {grouped}.") == [grouped]
    assert find_cards("0000 0000 0000 0000") == []  # one repeated digit
    bad = card[:-1] + str((int(card[-1]) + 1) % 10)
    assert find_cards(bad) == []
    assert find_cards("order 123456789012") == []  # too short
    assert find_cards(card[:4] + " -" + card[4:]) == []


def test_find_emails_skips_reserved_domains():
    text = "write to ana@mail.example.com or bo@example.org or x@host.test or cy@uni.edu.in"
    assert find_emails(text) == ["cy@uni.edu.in"]


def test_entropy():
    assert shannon_entropy("") == 0.0
    assert shannon_entropy("aaaa") == 0.0
    assert shannon_entropy("abcd") == pytest.approx(2.0)


def test_find_secrets_prefixes_and_assignment():
    key_id = "AKIA" + rand(string.ascii_uppercase + string.digits, 16)
    token = "ghp_" + rand(string.ascii_letters + string.digits, 36)
    pem = "-----BEGIN " + "RSA PRIVATE" + " KEY-----"
    value = rand(string.ascii_letters + string.digits, 32)
    assert find_secrets(f"id {key_id}") == ["cloud access key id"]
    assert find_secrets(token) == ["access token"]
    assert find_secrets(pem) == ["private key"]
    assert find_secrets(f'api_key = "{value}"') == ["high-entropy value assigned to 'api_key'"]
    assert find_secrets("password: aaaaaaaaaaaaaaaaaaaaaaaa") == []  # low entropy
    assert find_secrets("token = short") == []


def test_rule_wrappers_report_turn_and_mask():
    card = luhn_complete("4" + rand(string.digits, 14))
    conv = normalize(
        {
            "messages": [
                {"role": "user", "content": f"my card is {card}, mail me at dee@corp.co"},
                {"role": "assistant", "content": "ghp_" + rand(string.ascii_letters, 40)},
            ]
        },
        line=3,
    )
    cards = check_pii_card(conv)
    assert cards[0].turn == 0 and card not in cards[0].message and "*" in cards[0].message
    assert check_pii_email(conv)[0].message.startswith("email address de")
    assert check_secret_key(conv)[0].turn == 1


def test_isbn_and_low_first_digit_are_not_cards():
    isbn = luhn_complete("978" + rand(string.digits, 9))
    assert len(isbn) == 13 and find_cards(isbn) == []
    assert find_cards(luhn_complete("1" + rand(string.digits, 14))) == []


def test_placeholders_are_not_secrets():
    assert find_secrets('apiKey = "YOUR_WEATHER_SERVICE_API_KEY"') == []
    assert find_secrets("token = Replace" + rand(string.ascii_letters + string.digits, 20)) == []
    assert find_secrets("secret = " + rand(string.ascii_lowercase, 30)) == []


def test_repeated_value_reported_once_per_conversation():
    conv = normalize(
        {
            "messages": [
                {"role": "user", "content": "contact ann@corp.co"},
                {"role": "assistant", "content": "I will email ann@corp.co and bob@corp.co."},
            ]
        }
    )
    found = check_pii_email(conv)
    assert [f.turn for f in found] == [0, 1]
    assert found[1].message.startswith("email address bo")
