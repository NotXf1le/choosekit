# Changelog

## Unreleased

- Confirmed the SemIf benchmark's `--model` against the server's `/v1/models` before running, since llama.cpp answers with whatever is loaded however the request names the model. `--skip-model-check` keeps the old behaviour and records the run as unchecked.

## 0.5.0 - 2026-09-20

- Added label scoring through OpenRouter with `choosekit/openrouter`.

## 0.4.2

- Made the request cancellation signal available to custom prompt formatters.
- Rejected conflicting forced-token log probabilities in `llama.cpp` responses.
- Validated SemIf comparison inputs before computing metrics and created missing benchmark output directories.

## 0.4.1

- Corrected the SemIf benchmark attribution.

## 0.4.0

- Initial public release.
