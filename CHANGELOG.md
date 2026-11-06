# Changelog

## Unreleased

### Added

- Results on two more public datasets, run after the rules were written: a 2,000-row slice of ultrachat_200k (multi-turn chat, MIT) and a 2,000-row slice of glaive-function-calling-v2 (tool calls with declared functions, Apache 2.0), with per-rule counts, examples by line, wall times and commands in the README.
- `scripts/fetch_data.py --only` to fetch a subset of the sources, and fetchers for the two new slices; the glaive rows are rewritten as `messages` records with a `tools` list by `glaive_to_messages`, which is tested.

### Fixed

- `truncated-final` reads a run of three or more backticks at the end of a line, after other text, as a fence that opens or closes a block; two complete ultrachat answers written that way were reported as unclosed code fences.
- `tool-args-schema` reports "matches more than one of oneOf" only when every branch uses supported keywords, since an ignored keyword such as `not` can let an extra branch match.

## 0.1.0 - 2026-11-06

### Added

- Format detection and normalisation for OpenAI / Hugging Face `messages`, ShareGPT, Alpaca, prompt/completion and text rendered by ChatML, Zephyr or Llama 3 chat templates, with a per-file majority vote.
- Tool calls read from `tool_calls` lists, ShareGPT `function_call` turns and `<tool_call>` tags; results from tool turns and `<tool_response>` tags; declared tools from `tools` or a `<tools>` block in the system prompt.
- Structural rules: `role-unknown`, `system-position`, `role-order`, `empty-turn`, `duplicate-turn`, `final-role`, `truncated-final` (including a missing end-of-turn token in rendered text).
- Tool rules: `tool-args-json`, `tool-unknown`, `tool-args-schema` (a dependency-free JSON Schema subset), `tool-result-match`.
- Content rules: `refusal` (phrase match, with the refusal rate in the report), the `refusal-rate` gate behind `--max-refusal-rate`, and `language-mix` (dominant Unicode script of the answer against the user turn).
- Dataset rules: `duplicate-prompt` (exact) and `length-outlier` (modified z-score on log length).
- PII and secret rules: `pii-email`, `pii-card` (Luhn), `pii-iban` (mod 97) and `secret-key` (prefixes and entropy), reading tool-call arguments as well as text.
- `sftlint check` with Markdown and JSON reports, `--select`, `--ignore`, `--fail-on` exit codes, `--min-records`, and `--tokenizer` (`pretokens`, or optional tiktoken).
- `--fix` with `--fix-rules`, which drops whole records or trailing turns and writes through a temporary file, so the output may be the input.
- `sftlint rules` and `sftlint detect` commands.
- Per-field character, word and token length statistics.
- Python API: `lint`, `load`, `Report`, `Dataset`, `to_json`, `to_markdown`, `apply_fixes`, `FixResult`.
- Scripts for results on public data: a fetch script (backs off on HTTP 429, writes each file atomically), a length chart script, a script by language, and a script that prints the Hermes tool-rule breakdowns. Measured results on public smoltalk, Hermes function-calling and Aya files are in the README.
- README with install, quickstart, how the checks work, results on public data, limitations and related tools.

### Known limitations

See the Limitations section of the README for what is not implemented (other input formats, language identification, a refusal classifier, near-duplicate detection) and for the heuristic behaviour of several rules.
