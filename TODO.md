# TODO

## Deferred

- Consider removing `addSpecialTokens` from the `llama.cpp` options in a future release.
- Add native `llama.cpp` support for requesting raw log probabilities for multiple arbitrary token IDs in one request (`logprob_token_ids`). Prefer zero-token generation (`n_predict: 0`) and calculate every requested token's log probability from the full-vocabulary softmax. Keep the current top-N plus exact forced-token fallback path for servers without the extension; an explicitly selected bulk mode must fail clearly when unsupported instead of silently changing protocols.
