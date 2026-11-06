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
- Fetch script, length chart script, and measured results on public smoltalk and Hermes function-calling files.
