# Changelog

## Unreleased

## 0.6.0 - 2026-09-22

- Added image inputs for vision-capable llama.cpp, Ollama, and OpenRouter models.
  llama.cpp image scoring uses `labels` mode.
- Added labels-only Ollama scoring with up to 20 choices through `choosekit/ollama`.
- Added an exact `/v1/models` check to the SemIf llama.cpp benchmark, with
  `--skip-model-check` for unverified runs.

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
