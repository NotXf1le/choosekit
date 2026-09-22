# choosekit-mcp

`choosekit-mcp` exposes [choosekit](https://github.com/NotXf1le/choosekit) as a read-only MCP tool backed by llama.cpp, Ollama, or OpenRouter.

The server uses stdio. It returns a decision and probability distribution. Execution of the selected action remains with the MCP client.

## Configuration

Select and configure the backend when the MCP process starts. `CHOOSEKIT_BACKEND` defaults to `llama-cpp`.

| Backend | Environment variable | Default | Description |
|---|---|---|---|
| All | `CHOOSEKIT_BACKEND` | `llama-cpp` | `llama-cpp`, `ollama`, or `openrouter` |
| All | `CHOOSEKIT_MODEL` | Not set | Optional for llama.cpp; required for Ollama and OpenRouter |
| All | `CHOOSEKIT_MODE` | `labels` | `labels` for every backend; `minimal-prefix` is llama.cpp only |
| All | `CHOOSEKIT_IMAGE_ROOT` | Current working directory | Root directory for files supplied through `imagePaths` |
| llama.cpp | `CHOOSEKIT_BASE_URL` | Required | Base URL of the llama.cpp server, for example `http://127.0.0.1:8080` |
| Ollama | `CHOOSEKIT_BASE_URL` | `http://127.0.0.1:11434` | Base URL of the Ollama server |
| OpenRouter | `OPENROUTER_API_KEY` | Required | OpenRouter API key |
| OpenRouter | `OPENROUTER_PROVIDER` | Not set | Pins one provider and disables fallback |

Every tool call uses the configuration set when the MCP process starts. Backend settings are not accepted as tool arguments. The llama.cpp server must provide its native `/tokenize` and `/completion` endpoints. OpenRouter receives the context, question, and choice descriptions.

Ollama and OpenRouter support `labels` mode with up to 20 choices. llama.cpp supports up to 26 choices in `labels` mode and has no additional MCP choice limit in `minimal-prefix` mode.

### Claude Code

```sh
claude mcp add choosekit --env CHOOSEKIT_BASE_URL=http://127.0.0.1:8080 -- npx -y choosekit-mcp
```

### Codex

```sh
codex mcp add choosekit --env CHOOSEKIT_BASE_URL=http://127.0.0.1:8080 -- npx -y choosekit-mcp
```

### OpenCode

```sh
opencode mcp add choosekit --env CHOOSEKIT_BASE_URL=http://127.0.0.1:8080 -- npx -y choosekit-mcp
```

### Ollama

For example, with Codex:

```sh
codex mcp add choosekit --env CHOOSEKIT_BACKEND=ollama --env CHOOSEKIT_MODEL=your-model -- npx -y choosekit-mcp
```

### OpenRouter

For example, with Codex:

```sh
codex mcp add choosekit --env CHOOSEKIT_BACKEND=openrouter --env OPENROUTER_API_KEY=... --env CHOOSEKIT_MODEL=your-model -- npx -y choosekit-mcp
```

On native Windows, launch `npx` through `cmd` in any example above: replace `-- npx -y choosekit-mcp` with `-- cmd /c npx -y choosekit-mcp`.

## Tool

The server exposes one tool named `choose`:

```json
{
  "context": "The customer reports that a payout failed.",
  "question": "Which team should handle this?",
  "choices": {
    "billing": "Payments, payouts, invoices, and refunds",
    "technical": "Bugs, outages, integrations, and API errors",
    "sales": "Pricing, upgrades, and new accounts"
  }
}
```

To score images, add `imagePaths` in `labels` mode:

```json
{
  "context": "Inspect the attached screenshot.",
  "question": "Which state is the interface in?",
  "choices": {
    "ready": "The interface is ready for input.",
    "loading": "The interface is still loading."
  },
  "imagePaths": ["screenshots/status.png"]
}
```

Relative paths are resolved from `CHOOSEKIT_IMAGE_ROOT`, which defaults to the MCP process working directory. Every image path must remain within that root. PNG, JPEG, and WebP files are supported, and the selected model must support vision. With OpenRouter, the image contents are sent to the remote service.

With `labels`, choosekit maps the supplied choice keys to A/B/C labels for scoring, so each description must contain the option's full meaning. With `minimal-prefix`, it scores the shortest token prefixes that distinguish the original keys. When no supplied option may apply, add an explicit choice such as `insufficient_information`.

The result contains the selected key, the complete normalized distribution, raw scores, margin, entropy, token-boundary rollback, and backend usage when available. A score is `null` when the upstream backend did not return a log probability for that choice; its probability in the distribution is `0`. Probabilities represent relative preference among the supplied choices. Estimating correctness requires separate calibration.

## Requirements

- Node.js 20 or newer.
- For llama.cpp, a reachable server with a compatible model already loaded.
- For Ollama, version 0.12.11 or newer and a compatible model.
- For OpenRouter, an API key and a model that returns the required log probabilities.

[Apache-2.0](LICENSE). Copyright 2026 NotXf1le.
