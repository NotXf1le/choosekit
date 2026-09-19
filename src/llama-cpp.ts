import { createFormattedChooser } from "./internal-chooser.js";
import { candidateTree, tokenIds } from "./tokens.js";
import type { TokenNode } from "./tokens.js";
import type { Chooser, ChooserOptions, Scorer, Usage } from "./types.js";
import { isCount, isRecord, requireText, ScoringError } from "./validation.js";

export interface LlamaCppOptions extends ChooserOptions {
  readonly baseURL: string;
  /** A/B/C labels by default, or original keys scored to their shortest unique token prefix. */
  readonly mode?: "labels" | "minimal-prefix";
  /** Optional for a direct single-model server; use the router's alias when routing models. */
  readonly model?: string;
  readonly headers?: Readonly<Record<string, string>>;
  readonly fetch?: typeof globalThis.fetch;
  /** Must match the agent's tokenizer setting. Defaults to false for serialized context. */
  readonly addSpecialTokens?: boolean;
}

interface Endpoints {
  readonly tokenize: string;
  readonly completion: string;
}

interface Probe {
  readonly logprob: number;
  readonly topLogprobs: ReadonlyMap<number, number>;
  readonly promptTokens: number;
  readonly cachedTokens: number | null;
  readonly completionTokens: number;
}

interface Branch {
  readonly node: TokenNode;
  readonly indices: readonly number[];
  readonly suffix: readonly number[];
  readonly score: number;
}

function endpoints(baseURL: string): Endpoints {
  requireText(baseURL, "baseURL");
  const url = new URL(baseURL);
  if ((url.protocol !== "http:" && url.protocol !== "https:") || url.username || url.password
    || url.search || url.hash) {
    throw new TypeError("baseURL must be an HTTP(S) URL without credentials, query, or fragment.");
  }
  const path = url.pathname.replace(/\/v1\/?$/, "").replace(/\/$/, "");
  url.pathname = `${path}/tokenize`;
  const tokenize = url.href;
  url.pathname = `${path}/completion`;
  return { tokenize, completion: url.href };
}

function requestHeaders(value: unknown): Readonly<Record<string, string>> {
  if (value === undefined) return Object.freeze({ "content-type": "application/json" });
  if (!isRecord(value) || Object.getOwnPropertySymbols(value).length !== 0) {
    throw new TypeError("headers must be a record of string names and values.");
  }
  const entries = Object.entries(value).map(([name, headerValue]) => {
    requireText(name, "Header name");
    if (typeof headerValue !== "string") {
      throw new TypeError("headers must be a record of string names and values.");
    }
    return [name, headerValue] as const;
  }).filter(([name]) => name.toLowerCase() !== "content-type");
  return Object.freeze({ ...Object.fromEntries(entries), "content-type": "application/json" });
}

async function post(fetchImpl: typeof globalThis.fetch, url: string,
  headers: Readonly<Record<string, string>>, body: unknown, signal?: AbortSignal): Promise<unknown> {
  signal?.throwIfAborted();
  let response: Response;
  try {
    response = await fetchImpl(url, {
      method: "POST", headers, body: JSON.stringify(body), ...(signal ? { signal } : {}),
    });
  } catch (error) {
    signal?.throwIfAborted();
    throw error;
  }
  signal?.throwIfAborted();
  if (!response.ok) {
    throw new ScoringError(`llama.cpp returned HTTP ${response.status} for ${new URL(url).pathname}.`);
  }
  try {
    const value: unknown = await response.json();
    signal?.throwIfAborted();
    return value;
  } catch {
    signal?.throwIfAborted();
    throw new ScoringError(`llama.cpp returned invalid JSON for ${new URL(url).pathname}.`);
  }
}

function parseTokenization(value: unknown): number[] {
  if (!isRecord(value)) throw new ScoringError("llama.cpp returned an invalid tokenization.");
  return tokenIds(value.tokens);
}

function parseProbe(value: unknown, prompt: readonly number[], targetTokenId: number): Probe {
  if (!isRecord(value)) throw new ScoringError("llama.cpp returned an invalid completion.");
  if (value.truncated === true) {
    throw new ScoringError("llama.cpp truncated the token prefix while scoring a candidate.");
  }

  const output = tokenIds(value.tokens);
  if (output[0] !== targetTokenId) {
    throw new ScoringError("llama.cpp did not generate the forced target token.");
  }
  if (!Array.isArray(value.completion_probabilities) || value.completion_probabilities.length === 0) {
    throw new ScoringError("llama.cpp did not return completion probabilities.");
  }
  const first: unknown = value.completion_probabilities[0];
  if (!isRecord(first) || first.id !== targetTokenId) {
    throw new ScoringError("llama.cpp returned a probability for a different token.");
  }
  if (!Array.isArray(first.top_logprobs) || first.top_logprobs.length === 0) {
    throw new ScoringError("llama.cpp did not return the requested raw probability data.");
  }
  const logprob: unknown = first.logprob;
  if (typeof logprob !== "number" || !Number.isFinite(logprob) || logprob > 0 || logprob <= -1e30) {
    throw new ScoringError("The forced token logprob is missing, invalid, or clamped.");
  }
  const topLogprobs = new Map<number, number>();
  for (const entry of first.top_logprobs) {
    if (!isRecord(entry) || !isCount(entry.id)
      || typeof entry.logprob !== "number" || !Number.isFinite(entry.logprob)
      || entry.logprob > 0 || entry.logprob <= -1e30) {
      throw new ScoringError("llama.cpp returned an invalid raw top-logprob entry.");
    }
    if (topLogprobs.has(entry.id)) {
      throw new ScoringError("llama.cpp returned duplicate raw top-logprob token IDs.");
    }
    topLogprobs.set(entry.id, entry.logprob);
  }

  const settings = value.generation_settings;
  if (isRecord(settings)
    && (settings.post_sampling_probs === true || settings.backend_sampling === true)) {
    throw new ScoringError("llama.cpp applied incompatible post-sampling probability settings.");
  }

  const promptTokens: unknown = value.tokens_evaluated;
  if (!isCount(promptTokens) || promptTokens !== prompt.length) {
    throw new ScoringError("llama.cpp did not evaluate the supplied numeric token prefix as sent.");
  }
  let completionTokens = output.length;
  if (value.tokens_predicted !== undefined) {
    if (!isCount(value.tokens_predicted) || value.tokens_predicted < 1) {
      throw new ScoringError("llama.cpp returned invalid generated-token usage.");
    }
    completionTokens = value.tokens_predicted;
  }

  let cachedTokens: number | null = null;
  if (isRecord(value.timings) && value.timings.cache_n !== undefined) {
    if (!isCount(value.timings.cache_n) || value.timings.cache_n > promptTokens) {
      throw new ScoringError("llama.cpp returned an invalid prompt-cache count.");
    }
    cachedTokens = value.timings.cache_n;
  }
  return { logprob, topLogprobs, promptTokens, cachedTokens, completionTokens };
}

export function fromLlamaCpp(options: LlamaCppOptions): Chooser {
  if (!isRecord(options)) throw new TypeError("options must be an object.");
  const { baseURL, model, mode = "labels", addSpecialTokens = false, formatPrompt } = options;
  if (model !== undefined) requireText(model, "model");
  if (mode !== "labels" && mode !== "minimal-prefix") {
    throw new TypeError("mode must be labels or minimal-prefix.");
  }
  if (typeof addSpecialTokens !== "boolean") throw new TypeError("addSpecialTokens must be a boolean.");
  if (options.fetch !== undefined && typeof options.fetch !== "function") {
    throw new TypeError("fetch must be a function.");
  }
  const fetchImpl = options.fetch ?? globalThis.fetch;
  if (typeof fetchImpl !== "function") throw new TypeError("A fetch implementation is required.");
  const urls = endpoints(baseURL);
  const headers = requestHeaders(options.headers);

  const score: Scorer = async ({ prompt, candidates, signal }) => {
    signal?.throwIfAborted();
    const encoded: number[][] = [];
    for (const content of [prompt, ...candidates.map((candidate) => prompt + candidate)]) {
      const response = await post(fetchImpl, urls.tokenize, headers, {
        content, add_special: addSpecialTokens, ...(model === undefined ? {} : { model }),
      }, signal);
      signal?.throwIfAborted();
      encoded.push(parseTokenization(response));
    }

    const prefix = encoded[0]!;
    const { root: treeRoot, shared } = candidateTree(prefix, encoded.slice(1));
    const base = prefix.slice(0, shared);
    const scores: number[] = new Array(candidates.length);
    let promptTokens = 0;
    let cachedTokens: number | null = 0;
    let completionTokens = 0;
    let requests = encoded.length;

    let root = treeRoot;
    const rootSuffix: number[] = [];
    while (root.children.size === 1) {
      const first = root.children.entries().next().value as [number, TokenNode] | undefined;
      if (!first) throw new ScoringError("Candidate tree has an invalid shared prefix.");
      rootSuffix.push(first[0]);
      root = first[1];
    }

    const work: Branch[] = [{
      node: root, indices: candidates.map((_, index) => index), suffix: rootSuffix, score: 0,
    }];
    while (work.length > 0) {
      const branch = work.pop()!;
      if (branch.indices.length < 2 || branch.node.children.size === 0) {
        throw new ScoringError("Candidate tree did not reach a distinguishing branch.");
      }
      const groups = new Map<number, number[]>();
      const position = shared + branch.suffix.length;
      for (const index of branch.indices) {
        const tokenId = encoded[index + 1]![position]!;
        const group = groups.get(tokenId);
        if (group) group.push(index);
        else groups.set(tokenId, [index]);
      }
      const children = [...branch.node.children];
      const siblingLogprobs = new Map<number, number>();
      const numericPrefix = [...base, ...branch.suffix];
      for (const [targetTokenId] of children) {
        if (siblingLogprobs.has(targetTokenId)) continue;
        const collectSiblings = siblingLogprobs.size === 0 && children.length > 1;
        signal?.throwIfAborted();
        const response = await post(fetchImpl, urls.completion, headers, {
          prompt: numericPrefix,
          ...(model === undefined ? {} : { model }),
          n_predict: 1,
          n_probs: collectSiblings ? 64 : 1,
          post_sampling_probs: false,
          backend_sampling: false,
          samplers: ["top_k"],
          top_k: 1,
          mirostat: 0,
          logit_bias: [[targetTokenId, 1_000_000_000]],
          return_tokens: true,
          stream: false,
          cache_prompt: true,
        }, signal);
        signal?.throwIfAborted();
        const probe = parseProbe(response, numericPrefix, targetTokenId);
        promptTokens += probe.promptTokens;
        cachedTokens = cachedTokens === null || probe.cachedTokens === null
          ? null : cachedTokens + probe.cachedTokens;
        completionTokens += probe.completionTokens;
        requests++;

        siblingLogprobs.set(targetTokenId, probe.logprob);
        if (collectSiblings) {
          for (const [tokenId, logprob] of probe.topLogprobs) {
            if (branch.node.children.has(tokenId) && !siblingLogprobs.has(tokenId)) {
              siblingLogprobs.set(tokenId, logprob);
            }
          }
        }
      }

      const deferred: Branch[] = [];
      for (const [targetTokenId, node] of children) {
        const logprob = siblingLogprobs.get(targetTokenId);
        if (logprob === undefined) {
          throw new ScoringError("A candidate branch is missing its raw log probability.");
        }
        const value = branch.score + logprob;
        if (!Number.isFinite(value)) throw new ScoringError("Sequence log-likelihood overflowed.");
        const indices = groups.get(targetTokenId)!;
        if (indices.length === 1) {
          scores[indices[0]!] = value;
        } else {
          deferred.push({
            node,
            indices,
            suffix: [...branch.suffix, targetTokenId],
            score: value,
          });
        }
      }
      for (let i = deferred.length - 1; i >= 0; i--) work.push(deferred[i]!);
    }

    const usage: Usage = { promptTokens, cachedTokens, completionTokens, requests };
    return { logprobs: scores, boundaryTokens: prefix.length - shared, usage };
  };
  return createFormattedChooser(score, formatPrompt === undefined ? {} : { formatPrompt },
    mode === "labels" ? "labels" : "keys");
}
