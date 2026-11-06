import pytest

from sftlint.content import (
    char_script,
    check_language_mix,
    check_refusal,
    content_stats,
    dominant_script,
    refusal_phrase,
    script_counts,
)
from sftlint.formats import normalize


def qa(q, a):
    return normalize(
        {"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]}
    )


@pytest.mark.parametrize(
    "text",
    [
        "I'm sorry, but I can't help with that.",
        "I cannot provide instructions for that.",
        "As an AI language model, I do not have opinions.",
        "Sure. I can’t assist with that request though.",
        "I am unable to comply.",
        "It's not appropriate to share someone's address.",
        "I'm sorry, I can't assist with that.",
        "Sorry, I can't help with that.",
        "I'm really sorry, but I can't help with that.",
        "Unfortunately, I cannot provide that information.",
        "I am so sorry, I won’t write that.",
    ],
)
def test_refusals(text):
    assert refusal_phrase(text)


@pytest.mark.parametrize(
    "text",
    [
        "As an example, take x = 2.",
        "Sorry to hear that. Here is a plan.",
        "I cannot stress this enough: test first.",
        "I can't help but wonder why.",
        "Unfortunately, I can't help but laugh at that one.",
        "Unfortunately, the bridge is closed.",
        "Sorry for the delay. Here it is.",
        "Here is the answer. " + "x" * 400 + " I cannot help with that.",
        "",
    ],
)
def test_not_refusals(text):
    assert refusal_phrase(text) is None


def test_check_refusal_turn_and_message():
    out = check_refusal(qa("How?", "I'm sorry, but I can't help with that."))
    assert len(out) == 1 and out[0].turn == 1 and out[0].severity == "info"
    assert check_refusal(qa("How?", "Like this.")) == []
    assert check_refusal(normalize({"messages": [{"role": "user", "content": "x"}]})) == []


def test_char_script():
    assert char_script("a") == "Latin" and char_script("1") is None
    assert char_script("é") == "Latin"
    assert char_script("ж") == "Cyrillic"
    assert char_script("中") == "Han"
    assert char_script("か") == "Kana" and char_script("カ") == "Kana"
    assert char_script("한") == "Hangul"
    assert char_script("Ａ") == "Latin"
    assert char_script("—") is None


def test_dominant_script_ignores_code_and_short_text():
    assert dominant_script("hi") is None
    ru = "Привет, как дела? Расскажи мне о погоде сегодня."
    assert dominant_script(ru) == "Cyrillic"
    assert (
        dominant_script(ru + "\n```python\nprint('hello world, long code here')\n```") == "Cyrillic"
    )
    assert script_counts("`abc` d") == {"Latin": 1}
    assert (
        dominant_script("東京は日本の首都です。とても大きな都市で、人口が多いです。") == "Japanese"
    )
    mixed = "Hello there my friend " + "Привет как дела мой друг"
    assert dominant_script(mixed) is None


def test_language_mix():
    q = "Какая сегодня погода в Москве? Расскажи подробно."
    out = check_language_mix(qa(q, "The weather in Moscow today is sunny and warm."))
    assert [f.message for f in out] == ["user turn is Cyrillic, answer is Latin"]
    assert (
        check_language_mix(qa(q, "Сегодня в Москве солнечно и тепло, около двадцати градусов."))
        == []
    )
    assert check_language_mix(qa("short", "The weather in Moscow today is sunny and warm.")) == []
    assert check_language_mix(normalize({"messages": [{"role": "user", "content": q}]})) == []


def test_content_stats():
    convs = [
        qa("q", "I'm sorry, but I can't help with that request today, friend."),
        qa("q", "Here is a long enough answer written in plain Latin script."),
        normalize({"messages": [{"role": "user", "content": "x"}]}),
    ]
    st = content_stats(convs)
    assert st["answered"] == 2 and st["refusals"] == 1 and st["refusal_rate"] == 0.5
    assert st["answer_scripts"] == {"Latin": 2}
    assert content_stats([])["refusal_rate"] == 0.0


def test_refusal_rate_gate(write_jsonl):
    from sftlint.cli import main
    from sftlint.content import check_refusal_rate
    from sftlint.lint import lint
    from sftlint.loader import load
    from sftlint.report import to_markdown

    rows = [
        {"prompt": "q1", "completion": "I'm sorry, but I can't help with that."},
        {"prompt": "q2", "completion": "Sure, here it is."},
    ]
    path = write_jsonl(rows)
    report = lint(load(path), max_refusal_rate=0.25)
    gate = [f for f in report.findings if f.rule == "refusal-rate"]
    assert len(gate) == 1 and gate[0].line == 0
    assert "1 of 2 answers refuse (50.00 %), above 25.00 %" in gate[0].message
    assert "- dataset: `refusal-rate`" in to_markdown(report)
    assert lint(load(path), max_refusal_rate=0.5).counts().get("refusal-rate") is None
    assert check_refusal_rate({"refusal_rate": 0.9}, None) == []
    assert main(["check", str(path), "--max-refusal-rate", "0.25", "--fail-on", "error"]) == 1
    assert main(["check", str(path), "--max-refusal-rate", "0.5", "--fail-on", "error"]) == 0


@pytest.mark.parametrize(
    "text",
    [
        "I can't help but wonder whether the proof holds.",
        "It's not appropriate to use a semicolon there; use a comma.",
        "I would advise against using global variables here.",
    ],
)
def test_ordinary_openings_are_not_refusals(text):
    assert refusal_phrase(text) is None


def test_refusal_after_a_heading_line():
    assert refusal_phrase("Title\nI cannot help with that.") == "I cannot help"
    assert refusal_phrase("It's not appropriate for me to share that.")


def test_char_script_halfwidth_kana_and_modifier_letters():
    assert char_script("ｱ") == "Kana"  # HALFWIDTH KATAKANA LETTER A
    assert dominant_script("ｱｲｳ漢字" * 5) == "Japanese"
    assert char_script("ʰ") is None  # MODIFIER LETTER SMALL H
    assert char_script("ᵃ") is None  # MODIFIER LETTER SMALL A
