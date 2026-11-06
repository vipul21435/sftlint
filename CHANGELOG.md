# Changelog

## 0.1.0 (unreleased)

- Format detection and normalisation for OpenAI / Hugging Face `messages`, ShareGPT, Alpaca and prompt/completion JSONL, with per-file majority vote.
- Structural rules: `role-unknown`, `system-position`, `role-order`, `empty-turn`, `duplicate-turn`, `final-role`, `truncated-final`.
- Dataset rules: `duplicate-prompt` (exact) and `length-outlier` (modified z-score on log length).
- PII and secret rules: `pii-email`, `pii-card` (Luhn), `secret-key` (prefixes and entropy).
- `sftlint check` with Markdown and JSON reports, `--select`, `--ignore`, `--fail-on` exit codes; `sftlint rules`; `sftlint detect`.
- Per-field character and word length statistics.
- Fetch script and measured results on public smoltalk and Hermes function-calling slices.
