import { createFormattedChooser } from "./internal-chooser.js";
import type { Chooser, ChooserOptions, Scorer, Usage } from "./types.js";
import { isCount, isRecord, requireText, ScoringError } from "./validation.js";

const DEFAULT_BASE_URL = "http://127.0.0.1:11434";
const MAX_CANDIDATES = 20;

export interface OllamaOptions extends ChooserOptions {
  readonly model: string;
  readonly baseURL?: string;
  readonly fetch?: typeof globalThis.fetch;
}

function endpoint(value: unknown): string {
  const baseURL = value === undefined ? DEFAULT_BASE_URL : value;
  requireText(baseURL, "baseURL");
  const url = new URL(baseURL);
  if ((url.protocol !== "http:" && url.protocol !== "https:") || url.username || url.password
    || url.search || url.hash) {
    throw new TypeError("baseURL must be an HTTP(S) URL without credentials, query, or fragment.");
  }
  const path = url.pathname.replace(/\/api\/?$/, "").replace(/\/$/, "");
  url.pathname = `${path}/api/chat`;
  return url.href;
}

async function post(fetchImpl: typeof globalThis.fetch, url: string, body: unknown,
  signal?: AbortSignal): Promise<unknown> {
  signal?.throwIfAborted();
  let response: Response;
  try {
    response = await fetchImpl(url, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
      ...(signal ? { signal } : {}),
    });
  } catch (error) {
    signal?.throwIfAborted();
    throw error;
  }
  signal?.throwIfAborted();
  if (!response.ok) {
    throw new ScoringError(`Ollama returned HTTP ${response.status} for ${new URL(url).pathname}.`);
  }
  try {
    const value: unknown = await response.json();
    signal?.throwIfAborted();
    return value;
  } catch {
    signal?.throwIfAborted();
    throw new ScoringError(`Ollama returned invalid JSON for ${new URL(url).pathname}.`);
  }
}

function parseUsage(value: Record<string, unknown>): Usage {
  const promptTokens = value.prompt_eval_count;
  const completionTokens = value.eval_count;
  if (!isCount(promptTokens) || !isCount(completionTokens)) {
    throw new ScoringError("Ollama returned invalid token usage.");
  }
  const cached = value.prompt_eval_cached_count;
  if (cached !== undefined && (!isCount(cached) || cached > promptTokens)) {
    throw new ScoringError("Ollama returned invalid cached-token usage.");
  }
  return Object.freeze({
    promptTokens,
    cachedTokens: cached === undefined ? null : cached,
    completionTokens,
    requests: 1,
  });
}

function collectLabelScore(value: unknown, expected: ReadonlySet<string>,
  found: Map<string, number>): void {
  if (!isRecord(value) || typeof value.token !== "string") {
    throw new ScoringError("Ollama returned an invalid logprob entry.");
  }
  if (!expected.has(value.token)) return;
  if (typeof value.logprob !== "number" || !Number.isFinite(value.logprob)
    || value.logprob > 0) {
    throw new ScoringError(`Ollama returned an invalid logprob for label ${value.token}.`);
  }
  if (value.bytes !== undefined && value.bytes !== null) {
    if (!Array.isArray(value.bytes) || value.bytes.length !== 1
      || value.bytes[0] !== value.token.charCodeAt(0)) {
      throw new ScoringError(`Ollama returned invalid bytes for label ${value.token}.`);
    }
  }
  const existing = found.get(value.token);
  if (existing !== undefined && existing !== value.logprob) {
    throw new ScoringError(`Ollama returned conflicting logprobs for label ${value.token}.`);
  }
  found.set(value.token, value.logprob);
}

function parseScores(value: unknown, candidates: readonly string[]): {
  readonly logprobs: readonly number[];
  readonly usage: Usage;
} {
  if (!isRecord(value) || !Array.isArray(value.logprobs) || value.logprobs.length !== 1
    || !isRecord(value.logprobs[0])) {
    throw new ScoringError("Ollama did not return exactly one scored token position.");
  }
  const position = value.logprobs[0];
  if (!Array.isArray(position.top_logprobs) || position.top_logprobs.length === 0) {
    throw new ScoringError("Ollama did not return top logprobs.");
  }
  const expected = new Set(candidates);
  const found = new Map<string, number>();
  collectLabelScore(position, expected, found);
  for (const entry of position.top_logprobs) collectLabelScore(entry, expected, found);
  if (found.size === 0) {
    throw new ScoringError("Ollama did not return logprobs for any choice label.");
  }
  return {
    logprobs: Object.freeze(candidates.map((candidate) => found.get(candidate) ?? -Infinity)),
    usage: parseUsage(value),
  };
}

export function fromOllama(options: OllamaOptions): Chooser {
  if (!isRecord(options)) throw new TypeError("options must be an object.");
  const { model, formatPrompt } = options;
  requireText(model, "model");
  if (options.fetch !== undefined && typeof options.fetch !== "function") {
    throw new TypeError("fetch must be a function.");
  }
  const fetchImpl = options.fetch ?? globalThis.fetch;
  if (typeof fetchImpl !== "function") throw new TypeError("A fetch implementation is required.");
  const url = endpoint(options.baseURL);

  const score: Scorer = async ({ prompt, candidates, signal }) => {
    if (candidates.length > MAX_CANDIDATES) {
      throw new TypeError(`Ollama supports at most ${MAX_CANDIDATES} choices.`);
    }
    const response = await post(fetchImpl, url, {
      model,
      messages: [{ role: "user", content: prompt }],
      stream: false,
      think: false,
      logprobs: true,
      top_logprobs: MAX_CANDIDATES,
      options: { num_predict: 1 },
    }, signal);
    return parseScores(response, candidates);
  };
  return createFormattedChooser(score, formatPrompt === undefined ? {} : { formatPrompt }, "labels");
}
