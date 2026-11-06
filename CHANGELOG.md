# Changelog

## 0.1.0 (unreleased)

- Format detection and normalisation for OpenAI / Hugging Face `messages`, ShareGPT, Alpaca, prompt/completion and text rendered by ChatML, Zephyr or Llama 3 chat templates, with per-file majority vote.
- Tool calls read from `tool_calls` lists, ShareGPT `function_call` turns and `<tool_call>` tags; results from tool turns and `<tool_response>` tags; declared tools from `tools` or a `<tools>` block in the system prompt.
- Structural rules: `role-unknown`, `system-position`, `role-order`, `empty-turn`, `duplicate-turn`, `final-role`, `truncated-final` (including a missing end-of-turn token in rendered text).
- Tool rules: `tool-args-json`, `tool-unknown`, `tool-args-schema` (a dependency-free JSON Schema subset), `tool-result-match`.
- Content rules: `refusal` (phrase match, with the refusal rate in the report), the `refusal-rate` gate behind `--max-refusal-rate`, and `language-mix` (dominant Unicode script of answer against user turn).
- Dataset rules: `duplicate-prompt` (exact) and `length-outlier` (modified z-score on log length).
- PII and secret rules: `pii-email`, `pii-card` (Luhn), `pii-iban` (mod 97), `secret-key` (prefixes and entropy), reading tool-call arguments as well as text.
- `sftlint check` with Markdown and JSON reports, `--select`, `--ignore`, `--fail-on` exit codes, `--fix` with `--fix-rules`, and `--tokenizer` (`pretokens` or optional tiktoken); `sftlint rules`; `sftlint detect`.
- Per-field character, word and token length statistics.
- Fetch script (backs off on HTTP 429, writes each file atomically), length chart script, a script that prints the Hermes tool-rule breakdowns, and measured results on public smoltalk, Hermes function-calling and Aya files.
- Package-level exports: `lint`, `load`, `Report`, `Dataset`, `to_json`, `to_markdown`, `apply_fixes`, `FixResult`.
- Fixes after the second review: `--fix` writes through a temporary file, so OUT may be the input; a UTF-8 byte-order mark is skipped; `<tool_response>` payload fields are no longer read as call ids; `secret-key` skips attribute paths and environment-variable names; `duplicate-prompt` includes declared tools and tool calls; `length-outlier` falls back to the mean absolute deviation when MAD is 0; `truncated-final` accepts a final capital `A`; empty tool results are not `empty-turn` findings; NaN and Infinity in tool arguments are invalid JSON; card numbers need digit groups of four (or 4-6-5 / 4-6-4); SSH remotes and documentation keys are skipped; narrower refusal phrases; half-width kana and modifier letters in the script detector; `--outlier-z` must be finite and above 0; internal errors and deep nesting give one-line messages; `--format` checks each record against that format's own shape; `lint` accepts a rule-id string and `load` rejects unknown formats.
