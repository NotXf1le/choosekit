# Changelog

## 0.9.1 - 2026-10-26

- Added optional `ScoringError.httpStatus` for HTTP failures from llama.cpp, Ollama, and OpenRouter.

## 0.9.0 - 2026-10-20

- Added the MCP `chooseMany` tool for independent named questions with shared context and optional images.
- Added optional `concurrency` to `chooseMany()` without changing its default parallelism. Queued questions keep a snapshot of their inputs and stop on failure or cancellation.

## 0.8.0 - 2026-10-10

- Added `chooseMany()` for independent questions with common context and optional images.

## 0.7.0 - 2026-09-28

- Reduced llama.cpp scoring requests with batched tokenization and overlapping vision checks.
- Removed a redundant image-prefix tokenization request.
- Choices absent from llama.cpp's first `top_logprobs` result now receive zero probability by default. Set `probeMissingLogprobs: true` to score them with additional requests.

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
