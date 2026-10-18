# choosekit

`choosekit` scores a finite set of choices and returns a typed decision with a probability distribution. 
It accepts `text and images` through llama.cpp, Ollama, and OpenRouter.

## Use cases

### Text decisions

<img src="assets/use-cases/overview.gif" width="1200" alt="Search, summarize, and compare handlers scored for four requests.">

```ts
const relevance = await choose({
  context: "",
  question: 'Does this passage answer whether export works offline? '
    + 'Passage: "Export is available on every paid plan."',
  choices: {
    answers: "Directly answers the question",
    partial: "Related, but missing the answer",
    unrelated: "Not relevant to the question",
  },
});
```

### Image decisions

<img src="assets/use-cases/vision.gif" width="1200" alt="Loading, ready, and error probabilities for four synthetic search-interface screenshots.">

## Install

Library:

```sh
npm install choosekit
```

MCP server:

```sh
npm install --global choosekit-mcp
```

## Why

Agents often need to choose from known options:

- approve or reject an action;
- route a message;
- select the next tool;
- classify evidence;
- choose one legal move.

`choosekit` scores choices using the model's conditional log probabilities at the token branches that distinguish them.

The project was inspired by [Jev and the System One model interface](https://typesafe.ai/blog/introducing-system-one-models-and-jev): application state in, typed probabilistic decisions out. Jev is a specialized hosted model. `choosekit` brings the same typed decision interface to general-purpose language models. The llama.cpp and Ollama backends run on infrastructure you choose; OpenRouter provides hosted inference.

`choosekit` is an independent project with no affiliation to TypeSafe or Jev.

## MCP server

[`choosekit-mcp`](packages/choosekit-mcp/README.md) provides read-only `choose` and `chooseMany` MCP tools through llama.cpp, Ollama, or OpenRouter for Claude Code, Codex, and OpenCode. Select the backend and configure it with environment variables when starting the MCP server.

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

The llama.cpp backend requires its native `/tokenize` and `/completion` endpoints. `minimal-prefix` is available only with this backend.

Set `probeMissingLogprobs: true` to score low-probability choices outside llama.cpp's returned `top_logprobs`.

The library has no telemetry.

## Ollama

```ts
import { fromOllama } from "choosekit/ollama";

const choose = fromOllama({ model: "your-model" });
```

`model` is required. The adapter uses `http://127.0.0.1:11434/` by default and supports only `labels` mode with up to 20 choices. It requires Ollama 0.12.11 or newer.

## OpenRouter

```ts
import { fromOpenRouter } from "choosekit/openrouter";

const choose = fromOpenRouter({
  apiKey: process.env.OPENROUTER_API_KEY!,
  model: "your-model",
});
```

The OpenRouter backend supports models and providers that return first-token `top_logprobs`, with up to 20 choices. It sends the prompt to OpenRouter and requests reasoning to be disabled.

Returned probabilities are normalized across the supplied choices and are not calibrated correctness estimates.

OpenRouter may route the same model through different providers. Set `provider: "provider-slug"` to use only that provider and disable fallback.

## Image inputs

llama.cpp, Ollama, and OpenRouter can score choices from images when the selected model supports vision. Pass raw base64 data with its media type:

```ts
import { readFile } from "node:fs/promises";

const decision = await choose({
  context: "Inspect the attached screenshot.",
  question: "Which state is the interface in?",
  choices: {
    ready: "The interface is ready for input.",
    loading: "The interface is still loading.",
  },
  images: [{
    mediaType: "image/png",
    base64: (await readFile("screenshot.png")).toString("base64"),
  }],
});
```

Supported media types are PNG, JPEG, and WebP. llama.cpp image inputs currently support `labels` mode only.

The MCP server accepts image file paths through `imagePaths` in `labels` mode. Paths are resolved from the server's working directory by default; set `CHOOSEKIT_IMAGE_ROOT` to use another root. Every image must remain within that root and be a PNG, JPEG, or WebP file. With OpenRouter, the image contents are sent to the remote service.

## Multiple questions

Use `chooseMany()` to ask several independent questions about the same context:

```ts
import { chooseMany } from "choosekit";

const decisions = await chooseMany(choose, {
  context: "A payout failed three days in a row. Payroll is due today.",
  questions: {
    route: {
      question: "Which team should handle this?",
      choices: { billing: "Payments and payouts", technical: "Bugs and API errors" },
    },
    urgency: {
      question: "How urgent is this?",
      choices: { high: "Needs attention today", low: "Can wait" },
    },
  },
});

console.log(decisions.route.choice, decisions.urgency.choice);
```

Each question calls `choose()` concurrently and returns its own decision and, when available, token and request counts. Supplied images are sent with each call. If one call fails, the group rejects while calls already in progress continue.

## Scoring modes

| Mode | Candidate representation | Use when |
|---|---|---|
| `labels` | `A`, `B`, `C`, ... | Default. Up to 26 choices with llama.cpp or 20 with Ollama or OpenRouter. |
| `minimal-prefix` | Original JSON-quoted keys | llama.cpp only. Use when key names should influence the decision. |

Choices for which the backend returns no logprob receive zero probability.

In `labels` mode, choices are shown to the model as `A`, `B`, `C` instead of their original keys. For example, `refund: "Issue the refund"` is shown as `"A": "Issue the refund"`. Each description must therefore make the option clear. `choosekit` maps the selected label back to the original key.

`minimal-prefix` walks the token tree until every key is distinguishable. For keys such as `watermelon` and `watermelon juice`, the shared token path is handled once and scoring stops when the paths separate.

## Return value

`choose()` resolves to:

```ts
{
  choice,          // selected caller key
  distribution,    // normalized probability for every supplied key
  scores,          // backend log-probability score for every key
  margin,          // largest probability minus the second largest
  entropy,         // Shannon entropy in nats
  boundaryTokens,  // prompt tokens rolled back at a tokenization boundary
  usage,           // backend work, when reported
}
```

The result and its nested records are immutable. Each call is stateless. The caller controls action execution, inference retries, and model selection.

## Prompt formatting

`context` is copied unchanged to the start of the scoring prompt. The default formatter then appends the question, choice descriptions, and an answer marker.

Use `formatPrompt` only when you need custom prompt formatting. The result must preserve `context` as an unchanged prefix so an existing server-side prefix cache can still be reused.

## Custom scorer

Use `createChooser` with any backend that can return one comparable conditional log-probability score per candidate:

```ts
import { createChooser, type Scorer } from "choosekit";

const scorer: Scorer = async ({ prompt, candidates, signal }) => ({
  logprobs: await scoreCandidateSequences(prompt, candidates, signal),
});

const choose = createChooser(scorer);
```

Scores use natural logarithms and must be at most zero.

## Benchmarks

Model comparisons, methodology, and reproduction commands live in [benchmarks/README.md](benchmarks/README.md).

## Requirements

- Node.js 20 or newer.
- The `choosekit` package has no runtime dependencies, model downloads, installation hooks, or bundled inference servers.

[Apache-2.0](LICENSE). Copyright 2026 NotXf1le.
