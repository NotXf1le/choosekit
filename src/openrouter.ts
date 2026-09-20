import { createFormattedChooser } from "./internal-chooser.js";
import type { Chooser, ChooserOptions, Scorer, Usage } from "./types.js";
import { isCount, isRecord, requireText, ScoringError } from "./validation.js";

const ENDPOINT = "https://openrouter.ai/api/v1/chat/completions";
const MAX_CANDIDATES = 20;
const CLAMPED_LOGPROB = -9999;

export interface OpenRouterOptions extends ChooserOptions {
  readonly apiKey: string;
  readonly model: string;
  readonly fetch?: typeof globalThis.fetch;
  /** Pin one OpenRouter provider and disable provider fallback. */
  readonly provider?: string;
}

async function post(fetchImpl: typeof globalThis.fetch, apiKey: string, body: unknown,
  signal?: AbortSignal): Promise<unknown> {
  signal?.throwIfAborted();
  let response: Response;
  try {
    response = await fetchImpl(ENDPOINT, {
      method: "POST",
      headers: {
        authorization: `Bearer ${apiKey}`,
        "content-type": "application/json",
      },
      body: JSON.stringify(body),
      ...(signal ? { signal } : {}),
    });
  } catch {
    signal?.throwIfAborted();
    throw new ScoringError("OpenRouter request failed.");
  }
  signal?.throwIfAborted();
  if (!response.ok) {
    throw new ScoringError(`OpenRouter returned HTTP ${response.status} for /chat/completions.`);
  }
  try {
    const value: unknown = await response.json();
    signal?.throwIfAborted();
    return value;
  } catch {
    signal?.throwIfAborted();
    throw new ScoringError("OpenRouter returned invalid JSON for /chat/completions.");
  }
}

function parseUsage(value: unknown): Usage | undefined {
  if (value === undefined || value === null) return undefined;
  if (!isRecord(value)
    || !isCount(value.prompt_tokens)
    || !isCount(value.completion_tokens)) {
    throw new ScoringError("OpenRouter returned invalid token usage.");
  }
  let cachedTokens: number | null = null;
  if (value.prompt_tokens_details !== undefined && value.prompt_tokens_details !== null) {
    if (!isRecord(value.prompt_tokens_details)) {
      throw new ScoringError("OpenRouter returned invalid cached-token usage.");
    }
    const cached = value.prompt_tokens_details.cached_tokens;
    if (cached !== undefined && cached !== null) {
      if (!isCount(cached) || cached > value.prompt_tokens) {
        throw new ScoringError("OpenRouter returned invalid cached-token usage.");
      }
      cachedTokens = cached;
    }
  }
  return Object.freeze({
    promptTokens: value.prompt_tokens,
    cachedTokens,
    completionTokens: value.completion_tokens,
    requests: 1,
  });
}

function hasRefusal(choice: Record<string, unknown>): boolean {
  if (choice.finish_reason === "content_filter") return true;
  const message = choice.message;
  return isRecord(message) && message.refusal !== undefined && message.refusal !== null;
}

function parseScores(value: unknown, candidates: readonly string[]): {
  readonly logprobs: readonly number[];
  readonly usage?: Usage;
} {
  if (!isRecord(value) || !Array.isArray(value.choices) || value.choices.length === 0) {
    throw new ScoringError("OpenRouter returned an invalid completion.");
  }
  const choice: unknown = value.choices[0];
  if (!isRecord(choice)) {
    throw new ScoringError("OpenRouter returned an invalid completion.");
  }
  if (hasRefusal(choice)) {
    throw new ScoringError("OpenRouter refused or filtered the scoring request.");
  }
  if (!isRecord(choice.logprobs)) {
    throw new ScoringError("OpenRouter did not return logprobs.");
  }
  if (Array.isArray(choice.logprobs.refusal) && choice.logprobs.refusal.length > 0) {
    throw new ScoringError("OpenRouter refused or filtered the scoring request.");
  }
  const content = choice.logprobs.content;
  if (!Array.isArray(content) || content.length !== 1 || !isRecord(content[0])) {
    throw new ScoringError("OpenRouter did not return exactly one scored token position.");
  }
  const top = content[0].top_logprobs;
  if (!Array.isArray(top) || top.length === 0) {
    throw new ScoringError("OpenRouter did not return top logprobs.");
  }

  const expected = new Set(candidates);
  const found = new Map<string, number>();
  for (const entry of top) {
    if (!isRecord(entry) || typeof entry.token !== "string") {
      throw new ScoringError("OpenRouter returned an invalid top-logprob entry.");
    }
    if (!expected.has(entry.token)) continue;
    if (found.has(entry.token)) {
      throw new ScoringError(`OpenRouter returned duplicate logprobs for label ${entry.token}.`);
    }
    if (typeof entry.logprob !== "number" || !Number.isFinite(entry.logprob)
      || entry.logprob > 0 || entry.logprob <= CLAMPED_LOGPROB) {
      throw new ScoringError(`OpenRouter returned an invalid or clamped logprob for label ${entry.token}.`);
    }
    if (entry.bytes !== undefined && entry.bytes !== null) {
      if (!Array.isArray(entry.bytes) || entry.bytes.length !== 1
        || entry.bytes[0] !== entry.token.charCodeAt(0)) {
        throw new ScoringError(`OpenRouter returned invalid bytes for label ${entry.token}.`);
      }
    }
    found.set(entry.token, entry.logprob);
  }

  if (found.size === 0) {
    throw new ScoringError("OpenRouter did not return logprobs for any choice label.");
  }
  const usage = parseUsage(value.usage);
  return {
    logprobs: Object.freeze(candidates.map((candidate) => found.get(candidate) ?? -Infinity)),
    ...(usage === undefined ? {} : { usage }),
  };
}

export function fromOpenRouter(options: OpenRouterOptions): Chooser {
  if (!isRecord(options)) throw new TypeError("options must be an object.");
  const { apiKey, model, provider, formatPrompt } = options;
  requireText(apiKey, "apiKey");
  requireText(model, "model");
  if (provider !== undefined) requireText(provider, "provider");
  if (options.fetch !== undefined && typeof options.fetch !== "function") {
    throw new TypeError("fetch must be a function.");
  }
  const fetchImpl = options.fetch ?? globalThis.fetch;
  if (typeof fetchImpl !== "function") throw new TypeError("A fetch implementation is required.");

  const score: Scorer = async ({ prompt, candidates, signal }) => {
    if (candidates.length > MAX_CANDIDATES) {
      throw new TypeError(`OpenRouter supports at most ${MAX_CANDIDATES} choices.`);
    }
    const response = await post(fetchImpl, apiKey, {
      model,
      messages: [{ role: "user", content: prompt }],
      max_tokens: 1,
      stream: false,
      temperature: 1,
      top_p: 1,
      logprobs: true,
      top_logprobs: MAX_CANDIDATES,
      reasoning_effort: "none",
      ...(provider === undefined ? {} : {
        provider: { only: [provider], allow_fallbacks: false },
      }),
    }, signal);
    return parseScores(response, candidates);
  };
  return createFormattedChooser(score, formatPrompt === undefined ? {} : { formatPrompt }, "labels");
}
