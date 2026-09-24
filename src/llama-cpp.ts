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
  /** Controls add_special for llama.cpp tokenization. Defaults to false; image inputs require false. */
  readonly addSpecialTokens?: boolean;
  /** Probe choices missing from the first top-logprob list. Defaults to false. */
  readonly probeMissingLogprobs?: boolean;
}

interface Endpoints {
  readonly tokenize: string;
  readonly detokenize: string;
  readonly completion: string;
  readonly props: string;
}

interface ImageSupport {
  readonly marker: string;
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

class UnsupportedBatchTokenization extends Error {}
class BatchTokenizationTooLarge extends Error {}

const TOKENIZE_CONCURRENCY = 16;

function isBatchFormatRejection(message: string): boolean {
  return /(?:content|input).{0,40}(?:must|expected|requires?).{0,40}(?:string|array)|(?:expected|requires?).{0,40}string.{0,40}content|(?:unsupported|invalid).{0,40}(?:mixed|array)|(?:mixed|array).{0,40}(?:not supported)/i
    .test(message);
}

function endpoints(baseURL: string, model?: string): Endpoints {
  requireText(baseURL, "baseURL");
  const url = new URL(baseURL);
  if ((url.protocol !== "http:" && url.protocol !== "https:") || url.username || url.password
    || url.search || url.hash) {
    throw new TypeError("baseURL must be an HTTP(S) URL without credentials, query, or fragment.");
  }
  const path = url.pathname.replace(/\/v1\/?$/, "").replace(/\/$/, "");
  url.pathname = `${path}/tokenize`;
  const tokenize = url.href;
  url.pathname = `${path}/detokenize`;
  const detokenize = url.href;
  url.pathname = `${path}/completion`;
  const completion = url.href;
  url.pathname = `${path}/props`;
  if (model !== undefined) url.searchParams.set("model", model);
  return { tokenize, detokenize, completion, props: url.href };
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
  headers: Readonly<Record<string, string>>, body: unknown, signal?: AbortSignal,
  batchTokenize = false): Promise<unknown> {
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
    if (batchTokenize && response.status === 413) throw new BatchTokenizationTooLarge();
    if (batchTokenize && [400, 415, 422].includes(response.status)) {
      const message = await response.text();
      signal?.throwIfAborted();
      if (isBatchFormatRejection(message)) throw new UnsupportedBatchTokenization();
    }
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

async function get(fetchImpl: typeof globalThis.fetch, url: string,
  headers: Readonly<Record<string, string>>, signal?: AbortSignal): Promise<unknown> {
  signal?.throwIfAborted();
  let response: Response;
  try {
    response = await fetchImpl(url, { method: "GET", headers, ...(signal ? { signal } : {}) });
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

function parseBatchedTokenization(value: unknown, expectedParts: number): number[][] {
  if (!isRecord(value) || !Array.isArray(value.tokens)) {
    throw new ScoringError("llama.cpp returned an invalid tokenization.");
  }
  const parts: number[][] = [[]];
  let markers = 0;
  for (const id of value.tokens) {
    if (id === -1) {
      markers++;
      parts.push([]);
    } else if (isCount(id)) {
      parts[parts.length - 1]!.push(id);
    } else {
      throw new ScoringError("llama.cpp returned an invalid batched token ID.");
    }
  }
  if (markers === 0 && parts[0]!.length > 0) throw new UnsupportedBatchTokenization();
  if (markers !== expectedParts - 1 || parts.some((part) => part.length === 0)) {
    throw new ScoringError("llama.cpp returned invalid batched tokenization boundaries.");
  }
  return parts;
}

function parseDetokenization(value: unknown): string {
  if (!isRecord(value) || typeof value.content !== "string") {
    throw new ScoringError("llama.cpp returned an invalid detokenization.");
  }
  return value.content;
}

function parseImageSupport(value: unknown): ImageSupport {
  if (!isRecord(value) || !isRecord(value.modalities) || value.modalities.vision !== true) {
    throw new ScoringError("llama.cpp does not advertise vision support for this model.");
  }
  if (typeof value.media_marker !== "string" || value.media_marker.length === 0) {
    throw new ScoringError("llama.cpp did not return a multimodal media marker.");
  }
  return Object.freeze({ marker: value.media_marker });
}

function parseProbe(value: unknown, expectedPromptTokens: number | null,
  targetTokenId: number): Probe {
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
  const repeatedLogprob = topLogprobs.get(targetTokenId);
  if (repeatedLogprob !== undefined && repeatedLogprob !== logprob) {
    throw new ScoringError("llama.cpp returned conflicting logprobs for the forced token.");
  }

  const settings = value.generation_settings;
  if (isRecord(settings)
    && (settings.post_sampling_probs === true || settings.backend_sampling === true)) {
    throw new ScoringError("llama.cpp applied incompatible post-sampling probability settings.");
  }

  const promptTokens: unknown = value.tokens_evaluated;
  if (!isCount(promptTokens)) {
    throw new ScoringError("llama.cpp returned invalid prompt-token usage.");
  }
  if (expectedPromptTokens !== null && promptTokens !== expectedPromptTokens) {
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
  const { baseURL, model, mode = "labels", addSpecialTokens = false,
    probeMissingLogprobs = false, formatPrompt } = options;
  if (model !== undefined) requireText(model, "model");
  if (mode !== "labels" && mode !== "minimal-prefix") {
    throw new TypeError("mode must be labels or minimal-prefix.");
  }
  if (typeof addSpecialTokens !== "boolean") throw new TypeError("addSpecialTokens must be a boolean.");
  if (typeof probeMissingLogprobs !== "boolean") {
    throw new TypeError("probeMissingLogprobs must be a boolean.");
  }
  if (options.fetch !== undefined && typeof options.fetch !== "function") {
    throw new TypeError("fetch must be a function.");
  }
  const fetchImpl = options.fetch ?? globalThis.fetch;
  if (typeof fetchImpl !== "function") throw new TypeError("A fetch implementation is required.");
  const urls = endpoints(baseURL, model);
  const headers = requestHeaders(options.headers);
  let batchUnsupported = false;
  const score: Scorer = async ({ prompt, candidates, images, signal }) => {
    const hasImages = images !== undefined && images.length > 0;
    if (hasImages && mode !== "labels") {
      throw new TypeError("llama.cpp image inputs require labels mode.");
    }
    if (hasImages && addSpecialTokens) {
      throw new TypeError("llama.cpp image inputs require addSpecialTokens to be false.");
    }
    signal?.throwIfAborted();
    const contents = [prompt, ...candidates.map((candidate) => prompt + candidate)];
    let requests = hasImages ? 1 : 0;
    const tokenizeIndividually = async (): Promise<number[][]> => {
      const result: number[][] = new Array(contents.length);
      let next = 0;
      let failed = false;
      await Promise.all(Array.from({ length: Math.min(TOKENIZE_CONCURRENCY, contents.length) },
        async () => {
          try {
            while (!failed && next < contents.length) {
              const index = next++;
              signal?.throwIfAborted();
              requests++;
              const response = await post(fetchImpl, urls.tokenize, headers, {
                content: contents[index], add_special: addSpecialTokens,
                ...(model === undefined ? {} : { model }),
              }, signal);
              signal?.throwIfAborted();
              result[index] = parseTokenization(response);
            }
          } catch (error) {
            failed = true;
            throw error;
          }
        }));
      return result;
    };
    const tokenize = async (): Promise<number[][]> => {
      if (addSpecialTokens || batchUnsupported) return tokenizeIndividually();
      try {
        requests++;
        const content: (string | number)[] = [];
        for (const part of contents) {
          if (content.length > 0) content.push(-1);
          content.push(part);
        }
        const response = await post(fetchImpl, urls.tokenize, headers, {
          content, add_special: false, ...(model === undefined ? {} : { model }),
        }, signal, true);
        signal?.throwIfAborted();
        return parseBatchedTokenization(response, contents.length);
      } catch (error) {
        if (!(error instanceof UnsupportedBatchTokenization || error instanceof BatchTokenizationTooLarge)) {
          throw error;
        }
        if (error instanceof UnsupportedBatchTokenization) batchUnsupported = true;
        return tokenizeIndividually();
      }
    };
    const [imageSupport, encoded] = hasImages
      ? await Promise.all([
        get(fetchImpl, urls.props, headers, signal).then(parseImageSupport), tokenize(),
      ])
      : [undefined, await tokenize()];

    const prefix = encoded[0]!;
    const { root: treeRoot, shared } = candidateTree(prefix, encoded.slice(1));
    const base = prefix.slice(0, shared);
    const scores: number[] = new Array(candidates.length);
    let promptTokens = 0;
    let cachedTokens: number | null = 0;
    let completionTokens = 0;

    let root = treeRoot;
    const rootSuffix: number[] = [];
    while (root.children.size === 1) {
      const first = root.children.entries().next().value as [number, TokenNode] | undefined;
      if (!first) throw new ScoringError("Candidate tree has an invalid shared prefix.");
      rootSuffix.push(first[0]);
      root = first[1];
    }
    const materializeImagePrompt = async (numericPrefix: readonly number[]) => {
      const response = await post(fetchImpl, urls.detokenize, headers, {
        tokens: numericPrefix, ...(model === undefined ? {} : { model }),
      }, signal);
      requests++;
      const detokenized = parseDetokenization(response);
      if (detokenized.includes(imageSupport!.marker)) {
        throw new ScoringError("The formatted prompt contains llama.cpp's multimodal media marker.");
      }

      const value = Object.freeze({
        prompt_string: `${images!.map(() => imageSupport!.marker).join("\n")}\n${detokenized}`,
        multimodal_data: Object.freeze(images!.map(({ base64 }) => base64)),
      });
      return value;
    };

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
      const promptValue = hasImages ? await materializeImagePrompt(numericPrefix) : numericPrefix;
      for (const [targetTokenId] of children) {
        if (siblingLogprobs.has(targetTokenId)) continue;
        if (!probeMissingLogprobs && siblingLogprobs.size > 0) break;
        const collectSiblings = siblingLogprobs.size === 0 && children.length > 1;
        signal?.throwIfAborted();
        const response = await post(fetchImpl, urls.completion, headers, {
          prompt: promptValue,
          ...(model === undefined ? {} : { model }),
          n_predict: 1,
          n_probs: collectSiblings
            ? Math.min(256, Math.max(64, children.length * 16))
            : 1,
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
        const probe = parseProbe(response, hasImages ? null : numericPrefix.length, targetTokenId);
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
          if (probeMissingLogprobs) {
            throw new ScoringError("A candidate branch is missing its raw log probability.");
          }
          for (const index of groups.get(targetTokenId)!) scores[index] = -Infinity;
          continue;
        }
        const value = branch.score + logprob;
        if (!Number.isFinite(value)) {
          throw new ScoringError("Candidate log-probability score overflowed.");
        }
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
