# Benchmarks

## SuperGPQA

The chart uses a deterministic 1,000-question sample stratified by discipline and
difficulty. The random-choice baseline is the mean of `1 / number of choices` across
the sample. OpenRouter models are included only when they return `top_logprobs`.
The evaluation sample excludes a deterministic 100-question pilot used to select
working model and provider pairs.

### Prepare the dataset

```sh
hf download m-a-p/SuperGPQA SuperGPQA-all.jsonl \
  --repo-type dataset \
  --revision 4430d4458112c7d4497fdcf94d7cc223313d6acf \
  --local-dir benchmarks/.data/supergpqa/source
node benchmarks/prepare-supergpqa.mjs
npm run build
```

### Run the benchmark

```sh
OPENROUTER_API_KEY=... node benchmarks/run-supergpqa.mjs \
  --backend openrouter \
  --model ibm-granite/granite-4.0-h-micro \
  --provider cloudflare \
  --sample-method proportional \
  --sample-size 1000 \
  --output benchmarks/results/supergpqa-granite-4.0-h-micro-cloudflare.json
```

The chart uses these OpenRouter model and provider pairs:

| Model | Provider | Result |
|---|---|---|
| `ibm-granite/granite-4.0-h-micro` | `cloudflare` | `supergpqa-granite-4.0-h-micro-cloudflare.json` |
| `meta-llama/llama-3.1-8b-instruct` | `novita` | `supergpqa-llama-3.1-8b-novita.json` |
| `z-ai/glm-4.7-flash` | `cloudflare` | `supergpqa-glm-4.7-flash-cloudflare.json` |
| `z-ai/glm-5.2` | `cloudflare` | `supergpqa-glm-5.2-cloudflare.json` |
| `google/gemma-4-26b-a4b-it` | `dekallm` | `supergpqa-gemma-4-26b-dekallm.json` |
| `ibm-granite/granite-4.2-8b` | `coreweave` | `supergpqa-granite-4.2-8b-coreweave.json` |
| `deepseek/deepseek-v4.1-flash` | `wafer` | `supergpqa-deepseek-v4.1-flash-wafer.json` |
| `deepseek/deepseek-v4-pro-0813` | `cloudflare` | `supergpqa-deepseek-v4-pro-cloudflare.json` |
| `moonshotai/kimi-k3` | `morph` | `supergpqa-kimi-k3-morph.json` |

Run Jev on the same sample:

```sh
OPENROUTER_API_KEY=... node benchmarks/run-supergpqa.mjs \
  --backend jev \
  --sample-method proportional \
  --sample-size 1000 \
  --output benchmarks/results/supergpqa-jev-1.13.json
```

Run a local model through llama.cpp:

```sh
node benchmarks/run-supergpqa.mjs \
  --backend llama-cpp \
  --base-url http://127.0.0.1:8080 \
  --model qwen3.8-27b-text-64k \
  --sample-method proportional \
  --sample-size 1000 \
  --output benchmarks/results/supergpqa-qwen3.8-27b-local.json
```

The X axis is average cost per decision multiplied by seconds per decision. The local
model is shown as a horizontal accuracy line.

```sh
node benchmarks/generate-supergpqa-chart.mjs
```

## SemIf

`data/semif-authored144.jsonl` is SemIf's official
[`benchmarks/data/authored144.jsonl`](https://github.com/TheoLeeCJ/SemIf/blob/b9cb32537e78be65f19abfcb1de8fc504b627d84/benchmarks/data/authored144.jsonl)
at commit `b9cb32537e78be65f19abfcb1de8fc504b627d84`. The examples were authored by the
SemIf project and are distributed under its MIT license, reproduced in
`data/SEMIF-LICENSE.txt`.

```sh
npm ci
npm run build

node benchmarks/run-semif.mjs \
  --mode labels \
  --base-url http://127.0.0.1:11434/ \
  --model qwen3.8-27b-text-64k \
  --output benchmarks/results/semif-qwen-labels.json
```

Before the first row, the script asks the server which models it serves (`GET /v1/models`) and stops
unless `--model` is one of them. llama.cpp answers with whatever is loaded however the request names
the model, so without that check a run labelled `qwen3.8-27b-text-64k` may have been answered by
something else entirely. `--skip-model-check` runs anyway and records `modelChecked: false` in the
report's `runtime` block.

Run the Jev comparison with an OpenRouter API key:

```sh
OPENROUTER_API_KEY=... node benchmarks/run-semif-openrouter-jev.mjs \
  --model typesafe/jev-1.13 \
  --output benchmarks/results/semif-jev-1.13.json

node benchmarks/compare-semif.mjs \
  --qwen benchmarks/results/semif-qwen-labels.json \
  --jev benchmarks/results/semif-jev-1.13.json \
  --output benchmarks/results/semif-comparison.json
```
