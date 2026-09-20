# choosekit-mcp

`choosekit-mcp` exposes [choosekit](https://github.com/NotXf1le/choosekit) as a read-only MCP tool backed by a model already running in llama.cpp.

The server uses stdio. It returns a decision and probability distribution. Execution of the selected action remains with the MCP client.

## Configuration

Set the llama.cpp connection when the MCP process starts:

| Environment variable | Default | Description |
|---|---|---|
| `CHOOSEKIT_BASE_URL` | Required | Base URL of the llama.cpp server, for example `http://127.0.0.1:8080` |
| `CHOOSEKIT_MODEL` | Not set | Model alias sent with llama.cpp requests |
| `CHOOSEKIT_MODE` | `labels` | Candidate scoring mode: `labels` or `minimal-prefix` |

Every tool call uses the configuration set when the MCP process starts. The llama.cpp server must provide its native `/tokenize` and `/completion` endpoints.

### Claude Code

```sh
claude mcp add choosekit --env CHOOSEKIT_BASE_URL=http://127.0.0.1:8080 -- npx -y choosekit-mcp
```

On native Windows, launch `npx` through `cmd`:

```powershell
claude mcp add choosekit --env CHOOSEKIT_BASE_URL=http://127.0.0.1:8080 -- cmd /c npx -y choosekit-mcp
```

### Codex

```sh
codex mcp add choosekit --env CHOOSEKIT_BASE_URL=http://127.0.0.1:8080 -- npx -y choosekit-mcp
```

### OpenCode

```sh
opencode mcp add choosekit --env CHOOSEKIT_BASE_URL=http://127.0.0.1:8080 -- npx -y choosekit-mcp
```

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

With `labels`, choosekit maps the supplied choice keys to A/B/C labels for scoring, so each description must contain the option's full meaning. With `minimal-prefix`, it scores the shortest token prefixes that distinguish the original keys. When no supplied option may apply, add an explicit choice such as `insufficient_information`.

The result contains the selected key, the complete normalized distribution, raw scores, margin, entropy, token-boundary rollback, and backend usage when available. Probabilities represent relative preference among the supplied choices. Estimating correctness requires separate calibration.

## Requirements

- Node.js 20 or newer.
- A reachable llama.cpp server with a compatible model already loaded.

[Apache-2.0](LICENSE). Copyright 2026 NotXf1le.
