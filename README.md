# choosekit

`choosekit` scores a finite set of choices with a language model you already run and returns a typed decision with a probability distribution.

```sh
npm install choosekit
```

## Why

Agents often need to choose from known options:

- approve or reject an action;
- route a message;
- select the next tool;
- classify evidence;
- choose one legal move.

`choosekit` uses the model's next-token probability mass to score declared alternatives directly, eliminating extra decoding and validation.

The project was inspired by [Jev and the System One model interface](https://typesafe.ai/blog/introducing-system-one-models-and-jev): application state in, typed probabilistic decisions out. Jev is a specialized hosted model. `choosekit` explores the same useful interface with a model you control. Application state stays on infrastructure you choose, and the decision path can use a model already running inside an existing deployment.

`choosekit` is an independent project with no affiliation to TypeSafe or Jev.

## llama.cpp

```ts
import { fromLlamaCpp } from "choosekit/llama-cpp";

const choose = fromLlamaCpp({
  baseURL: "http://127.0.0.1:8080/",
  mode: "labels",
});

const decision = await choose({
  context: "The deployment modifies production data and no backup exists.",
  question: "Should this action run without human approval?",
  choices: {
    yes: "The action is reversible, low-impact, and within scope.",
    no: "The action is destructive, irreversible, or broader than requested.",
  },
  signal: AbortSignal.timeout(30_000),
});

console.log(decision.choice);       // "no"
console.log(decision.distribution); // { yes: ..., no: ... }
```

Compatibility requires the native llama.cpp `/tokenize` and `/completion` endpoints with raw pre-sampling log probabilities and returned token IDs. An OpenAI-compatible `/v1` endpoint alone lacks these capabilities.

The library has no telemetry.

## Scoring modes

| Mode | Candidate representation | Use when |
|---|---|---|
| `labels` | `A`, `B`, `C`, ... | Default. The choice set has at most 26 entries. |
| `minimal-prefix` | Original JSON-quoted keys | The key names should influence the decision, or the set has more than 26 entries. |

`labels` shows each description under a short label and maps the selected label back to the caller's key after selection. It usually needs one distinguishing token per option.

`minimal-prefix` walks the token tree until every key is distinguishable. For keys such as `watermelon` and `watermelon juice`, the shared token path is handled once and scoring stops when the paths separate.

## Benchmark

The local adapter was compared with `typesafe/jev-1.13` on SemIf's official 144-row `authored144` benchmark, which covers evidence interpretation, rule application, and candidate selection. The local model was **Qwen 3.8 27B Q4_XL** served by llama.cpp on an **NVIDIA RTX 4090**. The Qwen run used the default A/B/C mode. The model was already loaded, and requests were sent one at a time to a llama.cpp server on the same machine.

| Metric | Qwen 3.8 27B Q4_XL + choosekit | Jev 1.13 |
|---|---:|---:|
| Accuracy | 96.53% (139/144) | 96.53% (139/144) |
| Average balanced accuracy across task families | 96.01% | 95.56% |
| Median latency (p50) | 239 ms | 368 ms |
| 95th percentile latency (p95) | 286 ms | 546 ms |
| Throughput | 4.02 decisions/s | 2.43 decisions/s |

These results are specific to this 144-case benchmark, and performance can differ on other decision workloads. Latency is end-to-end. The Qwen server ran on the same machine. Jev was accessed through a hosted API. The timings therefore include different transport overhead. The repository includes an exact copy of SemIf's [`authored144.jsonl`](https://github.com/TheoLeeCJ/SemIf/blob/b9cb32537e78be65f19abfcb1de8fc504b627d84/benchmarks/data/authored144.jsonl), its MIT license, and the [reproduction commands](benchmarks/README.md).

### Probability examples

Examples from the same benchmark:

[`eafc22c8c40df3932a8e`](benchmarks/data/semif-authored144.jsonl#L112) asks whether the crate is currently in storage. The protocol gives the inventory priority; the current inventory and desk-log entries are missing.

| Choice | Qwen probability | Jev probability |
|---|---:|---:|
| Supported | 1.893% | 1.000% |
| Contradicted | 0.354% | 0.000% |
| **Insufficient evidence (selected by both)** | **97.753%** | **99.000%** |

[`f46f392ef9e9e9df564b`](benchmarks/data/semif-authored144.jsonl#L76) asks whether permission to scan a notebook is still valid. The owner authorized scanning on Monday; the record omits all later permission changes.

| Choice | Qwen probability | Jev probability |
|---|---:|---:|
| Supported | 2.366% | 0.000% |
| **Insufficient evidence (selected by both)** | **97.605%** | **99.000%** |
| Contradicted | 0.029% | 1.000% |

The Qwen + llama.cpp probabilities shown here are [uncalibrated](https://proceedings.mlr.press/v70/guo17a.html). [Jev is trained for calibrated decisions](https://typesafe.ai/blog/introducing-system-one-models-and-jev). The distributions look similar in these examples. This benchmark measures accuracy, latency, and distribution similarity.

### Distribution comparison

All 144 cases and all 432 option probabilities were matched by case and choice ID.

Total variation distance (TVD) compares two complete probability distributions. A TVD of 0% means they are identical; 100% means each system assigns probability to entirely different choices. Pearson correlation measures whether individual probabilities rise and fall together, with 1.0 indicating perfect linear correlation.

| Comparison | Result |
|---|---:|
| Both systems selected the same choice | 94.44% (136/144) |
| Pearson correlation across all 432 probabilities | 0.948 |
| Median total variation distance | 2.52% |
| Mean total variation distance | 9.36% |
| Cases with TVD at or below 5% | 64.58% (93/144) |
| Cases with TVD at or below 10% | 77.08% (111/144) |
| Cases with TVD above 20% | 14.58% (21/144) |

Most distributions are similar. Some differ substantially: the systems select different choices in eight cases, and the largest TVD is 94.23%.

## Prompt formatting

`context` is the exact serialized prefix against which the decision is scored. The default formatter preserves that prefix, then appends the question, choice descriptions, and an answer marker. Conversation history remains entirely in the caller-provided context.

For chat models, pass the model's normal serialized chat prefix and use `formatPrompt` when the decision turn needs a particular template. Use the template required by your model; this example uses Qwen's chat markers:

```ts
const choose = fromLlamaCpp({
  baseURL: "http://127.0.0.1:8080/",
  mode: "minimal-prefix",
  formatPrompt: ({ context, instruction }) =>
    `${context}<|im_start|>user\n${instruction}<|im_end|>\n<|im_start|>assistant\n`,
});
```

The formatted prompt must start with `context` unchanged so an existing server-side prefix cache can still be reused.

## Custom scorer

Use `createChooser` with any backend that can return one conditional log-likelihood per candidate:

```ts
import { createChooser, type Scorer } from "choosekit";

const scorer: Scorer = async ({ prompt, candidates, signal }) => ({
  logprobs: await scoreCandidateSequences(prompt, candidates, signal),
});

const choose = createChooser(scorer);
```

Scores use natural logarithms and must be at most zero.

## Result

`choose()` resolves to:

```ts
{
  choice,          // selected caller key
  distribution,    // normalized probability for every supplied key
  scores,          // backend log-likelihood score for every key
  margin,          // largest probability minus the second largest
  entropy,         // Shannon entropy in nats
  boundaryTokens,  // prompt tokens rolled back at a tokenization boundary
  usage,           // backend work, when reported
}
```

The result and its nested records are immutable. Each call is stateless. The caller controls action execution, inference retries, and model selection.

## Requirements

- Node.js 20 or newer.
- No runtime dependencies, model downloads, installation hooks, or bundled inference servers.

[Apache-2.0](LICENSE). Copyright 2026 NotXf1le.
