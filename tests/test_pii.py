import random
import string

import pytest

from sftlint.formats import normalize
from sftlint.pii import (
    check_pii_card,
    check_pii_email,
    check_pii_iban,
    check_secret_key,
    find_cards,
    find_emails,
    find_ibans,
    find_secrets,
    iban_valid,
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
    assert luhn_valid("17893729974")  # the worked example on the Luhn algorithm page
    assert luhn_valid("79927398713")
    assert not luhn_valid("17893729975")
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
    assert find_emails("send to your-email@mail.co or recipient@mail.co") == []


def test_entropy():
    assert shannon_entropy("") == 0.0
    assert shannon_entropy("aaaa") == 0.0
    assert shannon_entropy("abcd") == pytest.approx(2.0)


def test_find_secrets_prefixes_and_assignment():
    key_id = "AKIA" + rand(string.ascii_uppercase + string.digits, 16)
    token = "ghp_" + rand(string.ascii_letters + string.digits, 36)
    pem = "-----BEGIN " + "RSA PRIVATE" + " KEY-----"
    value = rand(string.ascii_letters + string.digits, 32)
    assert find_secrets(f"id {key_id}") == [("cloud access key id", key_id)]
    assert find_secrets(token) == [("access token", token)]
    assert find_secrets(pem) == [("private key", pem)]
    assert find_secrets(f'api_key = "{value}"') == [
        ("high-entropy value assigned to 'api_key'", value)
    ]
    assert find_secrets("password: aaaaaaaaaaaaaaaaaaaaaaaa") == []  # low entropy
    assert find_secrets("token = short") == []


ALNUM = string.ascii_letters + string.digits


@pytest.mark.parametrize(
    "label,build",
    [
        ("cloud API key", lambda: "AI" + "za" + rand(ALNUM + "_-", 35)),
        ("model API key", lambda: "sk-" + rand(ALNUM, 20) + "T3Blbk" + "FJ" + rand(ALNUM, 20)),
        ("model API key", lambda: "sk-" + "ant-api03-" + rand(ALNUM, 93) + "AA"),
        ("model-hub access token", lambda: "hf" + "_" + rand(string.ascii_letters, 34)),
        ("live payment API key", lambda: "sk" + "_live_" + rand(ALNUM, 24)),
    ],
)
def test_more_secret_prefixes(label, build):
    value = build()
    assert find_secrets(f"key {value} end") == [(label, value)]


def test_secret_prefix_near_misses():
    assert find_secrets("sk" + "_test_" + rand(ALNUM, 24)) == []  # test-mode keys
    assert find_secrets("hf" + "_" + rand(string.ascii_letters, 20)) == []  # too short
    assert find_secrets("sk-" + rand(ALNUM, 40)) == []  # no marker


def test_two_distinct_secrets_of_one_kind_are_both_reported():
    a, b = ("ghp_" + rand(ALNUM, 36) for _ in range(2))
    conv = normalize(
        {
            "messages": [
                {"role": "user", "content": f"tokens {a} and {b}, again {a}"},
                {"role": "assistant", "content": f"Revoke {b}."},
            ]
        }
    )
    found = check_secret_key(conv)
    assert [f.turn for f in found] == [0, 0]
    assert all(a not in f.message and b not in f.message for f in found)
    assert found[0].message.startswith("access token gh")


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


def test_digits_inside_identifiers_are_not_cards():
    card = "5817629592874208"
    assert luhn_valid(card) and find_cards(card) == [card]
    assert find_cards("id 5fa9e56f" + card + " found") == []  # hex object id
    assert find_cards(card + "abc") == [] and find_cards("ref_" + card) == []
    assert find_cards(f"({card}).") == [card]


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


def test_tool_call_arguments_are_scanned():
    from sftlint.formats import normalize
    from sftlint.pii import check_pii_email

    c = normalize(
        {
            "conversations": [
                {"from": "human", "value": "Email Ana."},
                {
                    "from": "gpt",
                    "value": '<tool_call>{"name": "send", '
                    '"arguments": {"to": "ana@corp.io"}}</tool_call>',
                },
            ]
        }
    )
    out = check_pii_email(c)
    assert len(out) == 1 and out[0].turn == 1


def make_iban(country: str, bban: str) -> str:
    """Compute the two check digits for ``country`` + ``bban`` (98 - mod 97)."""
    digits = "".join(str(int(ch, 36)) for ch in bban + country + "00")
    return f"{country}{98 - int(digits) % 97:02d}{bban}"


def test_iban_published_example_and_check_digits():
    example = "GB82 WEST 1234 5698 7654 32"  # the fictitious example on the IBAN page
    assert iban_valid(example) and iban_valid(example.replace(" ", ""))
    assert make_iban("GB", "WEST12345698765432") == "GB82WEST12345698765432"
    assert not iban_valid("GB83WEST12345698765432")
    assert not iban_valid("GB82WEST1234569876543")  # one character short for GB
    assert not iban_valid("ZZ82WEST12345698765432")  # unknown country code


def test_iban_detects_every_single_character_change():
    iban = make_iban("DE", rand(string.digits, 18))
    assert iban_valid(iban)
    for i in range(4, len(iban)):
        for d in string.digits:
            if d != iban[i]:
                assert not iban_valid(iban[:i] + d + iban[i + 1 :])


def test_find_ibans_in_text_and_rule():
    nl = make_iban("NL", "TEST" + rand(string.digits, 10))
    grouped = " ".join(nl[i : i + 4] for i in range(0, len(nl), 4))
    assert find_ibans(f"Send it to {grouped}.") == [nl]
    assert find_ibans(f"ref{nl}") == [] and find_ibans(nl.lower()) == []
    conv = normalize(
        {
            "messages": [
                {"role": "user", "content": f"My IBAN is {grouped}"},
                {"role": "assistant", "content": f"Noted: {nl}."},
            ]
        }
    )
    found = check_pii_iban(conv)
    assert len(found) == 1 and found[0].turn == 0 and nl not in found[0].message


@pytest.mark.parametrize(
    "text",
    [
        "api_key = settings.OPENAI_API_KEY",
        "token = request.headers.Authorization",
        "password = self.cleaned_data.Password1",
        "api_key = PAYMENTS_API_KEY_2024",
        "```python\nclient = Client(api_key=config.secrets.payment_api_key_v2)\n```",
    ],
)
def test_names_assigned_to_key_like_names_are_not_secrets(text):
    assert find_secrets(text) == []


def test_random_values_with_dots_are_still_secrets():
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5NXgL0n3I9P"
    assert find_secrets(f"token = {jwt}")
    assert find_secrets("secret = Zk3qP9vLm2Xw8RtY5nBc7HdJ")
