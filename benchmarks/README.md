# Benchmarks

`data/semif-authored144.jsonl` is an exact copy of SemIf's official [`benchmarks/data/authored144.jsonl`](https://github.com/TheoLeeCJ/SemIf/blob/b9cb32537e78be65f19abfcb1de8fc504b627d84/benchmarks/data/authored144.jsonl) at commit `b9cb32537e78be65f19abfcb1de8fc504b627d84`. Only the local filename differs.

The 144 examples were authored by the SemIf project. The dataset is distributed under SemIf's MIT license, reproduced in `data/SEMIF-LICENSE.txt`.

Build the package before running a benchmark:

```sh
npm ci
npm run build
```

Run the published llama.cpp adapter against a local server:

```sh
node benchmarks/run-semif.mjs \
  --mode labels \
  --base-url http://127.0.0.1:11434/ \
  --model qwen3.8-27b-text-64k \
  --output benchmarks/results/semif-qwen-labels.json
```

Before the first row, the script asks the server which models it serves (`GET /v1/models`) and stops
unless `--model` is one of them. llama.cpp answers with whatever is loaded however the request names
the model, so without that check a run labelled `qwen3.8-27b-text-64k` may have been answered by
something else entirely. `--skip-model-check` runs anyway and records `modelChecked: false` with a
null `resolvedModel` in the report's `runtime` block.

Run the Jev comparison with an OpenRouter API key:

```sh
OPENROUTER_API_KEY=... node benchmarks/run-semif-openrouter-jev.mjs \
  --model typesafe/jev-1.13 \
  --output benchmarks/results/semif-jev-1.13.json
```

Compare the complete distributions:

```sh
node benchmarks/compare-semif.mjs \
  --qwen benchmarks/results/semif-qwen-labels.json \
  --jev benchmarks/results/semif-jev-1.13.json \
  --output benchmarks/results/semif-comparison.json
```

Benchmark result files are ignored because they can contain environment-specific timing and provider metadata.
