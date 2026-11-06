# sftlint

A linter for chat, SFT and tool-calling datasets in JSONL: it checks turn structure, truncation, duplicate prompts, length outliers and PII or secret patterns, and writes a Markdown or JSON report with an exit code for CI.

## Why this is hard

A fine-tuning file can parse as valid JSON on every line and still teach the wrong thing. A system turn in the middle of a conversation, two user turns in a row, or an empty assistant turn is accepted by most loaders and then rendered by a chat template into a token sequence the model never sees at inference time. A final answer cut off by a generation limit trains the model to stop mid-sentence. A credential pasted into a prompt is memorised. None of these are schema errors, and they hide in files of hundreds of thousands of records written in four or five slightly different layouts (OpenAI `messages`, ShareGPT `from`/`value`, Alpaca `instruction`/`output`, prompt/completion pairs). Each check also has to be conservative: a rule that fires on a tenth of a clean dataset is switched off and never read again.

## Install

```
pip install git+https://github.com/vipul21435/sftlint
```

Python 3.10 or newer. No runtime dependencies outside the standard library.

## Quickstart

`examples/sample.jsonl` has nine small conversations, seven of them with a planted problem.

```
$ sftlint check examples/sample.jsonl
examples/sample.jsonl: 9 records, format messages, 7 with findings (3 errors, 4 warnings, 0 info)
  duplicate-prompt   warning  1
  empty-turn         error    1
  final-role         warning  1
  role-order         error    1
  system-position    error    1
  truncated-final    warning  2
$ echo $?
1
```

The Markdown report lists each finding with its line and turn:

```
$ sftlint check examples/sample.jsonl --fail-on never --markdown -
...
- line 3 turn 1: `truncated-final` unclosed code fence
- line 4 turn 1: `system-position` system turn is not the first turn
- line 5 turn 1: `role-order` two user turns in a row
- line 6 turn 1: `truncated-final` ends with ':'
- line 7 turn 1: `empty-turn` assistant turn content is empty
- line 8 turn 2: `final-role` last turn is user
- line 9: `duplicate-prompt` same prompt as line 2
```

Other commands and options:

```
sftlint rules                                  # every rule id, default severity, summary
sftlint detect data.jsonl                      # detected format and record counts
sftlint check data.jsonl --json report.json --markdown report.md
sftlint check data.jsonl --select role-order,empty-turn
sftlint check data.jsonl --ignore length-outlier --fail-on warning
sftlint check data.jsonl --format sharegpt --outlier-z 4 --max-examples 50
```

Exit codes: 0 when no finding is at or above `--fail-on` (default `error`; `never` always exits 0), 1 when one is, 2 on a usage or I/O error. From Python, see `examples/lint_sample.py`.

## How it works

**Formats.** Each record is normalised to a list of `(role, content, tool_calls, tool_call_id)` turns. `--format auto` (the default) takes a majority vote of per-record detection over the first 200 records; a record that does not match the file format is reported as `format-mismatch` instead of being guessed at. Supported layouts:

| format | shape | notes |
|---|---|---|
| `messages` | `{"messages": [{"role", "content"}, ...]}` (also under `conversation`, `conversations`, `chat`) | OpenAI chat format and the input of Hugging Face chat templates; `developer` maps to system, `function` to tool; list-of-parts content is joined; a top-level `tools` list is kept |
| `sharegpt` | `{"conversations": [{"from", "value"}, ...]}` | `human` to user, `gpt` to assistant; `function_call` turns become assistant tool calls and `observation` turns tool results, as in the LLaMA-Factory data format |
| `alpaca` | `{"instruction", "input", "output"}` | `instruction` and `input` are concatenated into the user turn, as LLaMA-Factory documents; sftlint separates them with a blank line |
| `prompt-completion` | `{"prompt", "completion"}` | one user and one assistant turn |

**Rules.** Severities are defaults; `--fail-on` decides which ones fail a CI job.

| rule | severity | flags | why |
|---|---|---|---|
| `json-parse` | error | a line that is not valid JSON | nothing downstream can read it |
| `format-mismatch` | error | a record that does not match the file's format | mixed layouts are usually a merge mistake |
| `role-unknown` | error | a role outside system, user, assistant, tool | the OpenAI fine-tuning data-prep notebook rejects unrecognised roles; templates drop or crash on them |
| `system-position` | error | a system turn after the first turn | templates place the system prompt first; a later one is rendered in a position the model is not trained for |
| `role-order` | error | a first turn that is not user, two user or two assistant turns in a row, a user turn right after a tool result, a tool turn not preceded by an assistant tool call | the expected grammar is `user (assistant-with-tool-calls tool+)* assistant`, repeated |
| `empty-turn` | error | a turn with no text and no tool calls | trains on an empty target, or an empty context |
| `duplicate-turn` | warning | a turn that repeats an earlier turn of the same role verbatim (whitespace collapsed) | often a copy error in multi-turn synthesis |
| `final-role` | warning | a last turn that is not assistant | the record has no target to learn from; the OpenAI data-prep notebook checks that every conversation has an assistant message |
| `truncated-final` | warning | a final assistant turn with an unclosed triple-backtick fence, a last character that cannot end a sentence (`, ; : ( [ {` and operators), or a last word that is an article, conjunction or preposition | the usual trace of a generation that hit its token limit |
| `duplicate-prompt` | warning | a record whose prompt (every turn before the final assistant turn, with roles, whitespace collapsed, case kept) equals an earlier record's | exact duplicates over-weight one prompt; near-duplicates are out of scope (see datasieve) |
| `length-outlier` | info | a prompt or response length that is a robust outlier for the file | catches empty-ish and runaway records without a fixed limit |
| `pii-email` | warning | an email address outside reserved domains and placeholder local parts | personal data in training text |
| `pii-card` | warning | 13 to 19 digits starting 2 to 6 that pass the Luhn checksum | payment card numbers |
| `secret-key` | error | a known credential prefix, a PEM private-key header, or a high-entropy value assigned to a key-like name | credentials in training text get memorised |

**Length outliers.** Prompt and response lengths are counted in characters and in whitespace-separated words. Scores use the modified z-score of Iglewicz and Hoaglin on `x = log(1 + chars)`:

```
M_i = 0.6745 (x_i - median(x)) / MAD,    MAD = median(|x_i - median(x)|)
```

A record is flagged when `|M_i| > 3.5` (the cut-off the NIST handbook quotes from Iglewicz and Hoaglin; change it with `--outlier-z`). The log makes the score symmetric for right-skewed lengths, and the median and MAD stop a few huge records from masking each other the way a mean and standard deviation would. The rule is skipped below 20 records. A test draws 20,000 standard normal values and checks that fewer than 0.2 % score above 3.5.

**PII and secrets.** Each pattern has a validation step:

- Email: domains reserved for documentation and testing by RFC 2606 and RFC 6761 (`example.com`, `.test`, `.invalid`, `localhost`) are skipped, as are local parts such as `your-email` or `recipient`.
- Card numbers: digits may be grouped by single spaces or hyphens; the first digit must be 2 to 6 (the card-network industry identifiers of ISO/IEC 7812, which also keeps ISBN-13 numbers out), the digits must not all be the same, and the Luhn checksum must hold: double every second digit from the right, subtract 9 from any result above 9, and require the total to be divisible by 10.
- Secrets: fixed prefixes (cloud access key ids, personal access tokens, chat-app bot tokens, PEM private-key headers) match directly. A value assigned to a name such as `api_key`, `token` or `password` must be at least 20 characters, mix at least two of lower case, upper case and digits, contain no placeholder word (`your`, `example`, `xxxx` ...), and have Shannon entropy `H = -sum p_c log2 p_c` of at least 3.5 bits per character.

Matches are masked in reports (first and last two characters kept), and each distinct match is reported once per conversation.

## Results on public data

Two slices of public datasets, both under the Apache 2.0 licence as stated on their dataset cards (checked on the pages linked below). The data is not committed; `scripts/fetch_data.py` downloads it into `./data`, which git ignores.

```
uv run python scripts/fetch_data.py --out data --rows 2000
uv run sftlint check data/smoltalk-magpie-ultra.jsonl --markdown data/smoltalk-magpie-ultra.report.md
uv run sftlint check data/hermes-func-calling-singleturn.jsonl --markdown data/hermes-func-calling-singleturn.report.md
```

**smoltalk, `smol-magpie-ultra` subset, first 2,000 train rows** ([dataset card](https://huggingface.co/datasets/HuggingFaceTB/smoltalk); the card states the Smol-Magpie-Ultra subset is Apache 2.0). Format detected: `messages`. 78 records with findings, 0 errors, exit code 0.

| rule | count |
|---|---:|
| `truncated-final` | 5 |
| `duplicate-turn` | 3 |
| `length-outlier` | 76 |
| every other rule | 0 |

All five `truncated-final` findings are unclosed code fences, and reading them shows each answer stops in the middle of code (for example a list literal ending in `4.7,`). The three `duplicate-turn` findings are repeated user turns such as "Would you continue with more ideas?". In this slice the median prompt (all turns before the last answer) is 4,436.5 characters and the median response 2,179.

**hermes-function-calling-v1, `func-calling-singleturn.json`, all 1,893 records** ([dataset card](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1), Apache 2.0). Format detected: `sharegpt`. 123 records with findings, 0 errors, exit code 0.

| rule | count |
|---|---:|
| `pii-email` | 62 |
| `pii-card` | 2 |
| `length-outlier` | 76 |
| every other rule | 0 |

The `pii-email` matches are addresses written into the synthetic user requests; the rule cannot tell a made-up address on a real domain from a real one, which is why it is a warning. Both `pii-card` matches are the widely published test card number starting 4111, again inside user requests. This file keeps tool calls as `<tool_call>` tags inside assistant text, which the current normaliser does not parse (see Limitations).

Linting the 2,000-record smoltalk slice (14 MB) took 1.9 s of wall time on an arm64 laptop (`/usr/bin/time -p sftlint check data/smoltalk-magpie-ultra.jsonl`).

## Limitations

Not implemented yet:

- Tool-call checks: arguments as valid JSON, arguments against the declared JSON Schema in `tools`, and tool results matched to call ids. Tool calls are parsed and kept, but no rule reads them yet, and `<tool_call>` tags inside assistant text (the Hermes layout) are not parsed.
- Refusal-phrase rate and language-mix rules.
- Token counts with a real tokenizer; lengths are characters and whitespace words.
- `--fix` (dropping or trimming only what a named rule flags).
- Formats other than the four above, and inputs other than JSONL (no Parquet, no JSON arrays).

Known behaviour:

- `truncated-final` is a heuristic. An answer that legitimately ends with a colon, or a code answer that closes its fence with four backticks, is flagged; an answer cut at a sentence boundary is not.
- `duplicate-prompt` is exact. Prompts that differ by one character are not duplicates here.
- `length-outlier` is relative to the file: in a file that is mostly broken, the broken records are the norm.
- `pii-*` and `secret-key` are pattern matches with validation, not a PII classifier. Names, phone numbers and postal addresses are not detected.

## Related tools

- [datasieve](https://github.com/vipul21435/datasieve): validate and deduplicate SFT and preference data (schema quarantine, exact and MinHash near-dedup, contamination checks). sftlint reports exact duplicate prompts but does not quarantine or remove records, and leaves near-dedup to datasieve.
- [prefpairs](https://github.com/vipul21435/prefpairs): store pairwise preference judgments and rank with Bradley-Terry and Elo.
- [trajlens](https://github.com/vipul21435/trajlens): read coding-agent trajectories, measure per-run behaviour and classify failures.

## References

- OpenAI Cookbook, "Data preparation and analysis for chat model fine-tuning": https://github.com/openai/openai-cookbook/blob/main/examples/Chat_finetuning_data_prep.ipynb
- Hugging Face Transformers, chat templating guide: https://huggingface.co/docs/transformers/main/en/chat_templating
- LLaMA-Factory data format (Alpaca and ShareGPT, `function_call` and `observation` tags): https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README.md
- NIST/SEMATECH e-Handbook of Statistical Methods, 1.3.5.17 Detection of Outliers (modified z-score): https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
- Luhn algorithm: https://en.wikipedia.org/wiki/Luhn_algorithm
- ISO/IEC 7812 (issuer identification numbers): https://en.wikipedia.org/wiki/ISO/IEC_7812
- RFC 2606, Reserved Top Level DNS Names: https://www.rfc-editor.org/rfc/rfc2606
- RFC 6761, Special-Use Domain Names: https://www.rfc-editor.org/rfc/rfc6761

## Licence

MIT. See [LICENSE](LICENSE).

Built by Vipul Raj Jha · https://vipul21435.github.io
