# TODO

## Deferred

- Tokenize the prompt and prompt-plus-candidate inputs with bounded parallelism in the `llama.cpp` adapter instead of awaiting every `/tokenize` request sequentially. Preserve input order and exact boundary-aware tokenization; make concurrency configurable and measure transport overhead separately from server processing. Consider a future native batch-tokenization endpoint returning one token array per input, because stock `/tokenize` does not provide that response shape.
- Add native `llama.cpp` support for requesting raw log probabilities for multiple arbitrary token IDs in one request (`logprob_token_ids`). Prefer zero-token generation (`n_predict: 0`) and calculate every requested token's log probability from the full-vocabulary softmax. Keep the current top-N plus exact forced-token fallback path for servers without the extension; an explicitly selected bulk mode must fail clearly when unsupported instead of silently changing protocols.
- Add an OpenRouter scorer behind the existing `Scorer` and `Chooser` contracts. Support label scoring only: request `logprobs` and up to 20 `top_logprobs` from a compatible provider, require every choice label to be present, and fail clearly instead of treating a missing label as zero probability. Do not require a local tokenizer or expose `minimal-prefix` for this backend.
- Add an optional local MCP package for Claude Code, Codex, and OpenCode. Expose `choose` as a read-only tool, allow either `llama.cpp` or OpenRouter as the scorer, and keep MCP dependencies out of the core `choosekit` package.
