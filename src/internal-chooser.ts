import type {
  Choices, ChoiceKey, ChoiceRequest, Chooser, ChooserOptions, Decision, Scorer, Usage,
} from "./types.js";
import { isCount, isLogprob, isRecord, requireText, ScoringError } from "./validation.js";

export type CandidateFormat = "keys" | "labels";

const LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

function snapshot(choices: unknown): [string, string][] {
  if (!isRecord(choices) || Object.getOwnPropertySymbols(choices).length !== 0) {
    throw new TypeError("choices must be a record of string keys and descriptions.");
  }
  const entries = Object.entries(choices);
  if (entries.length < 2) throw new TypeError("Provide at least two choices.");
  return entries.map(([key, description]) => {
    requireText(key, "Choice key");
    if (key !== key.trim()) throw new TypeError("Choice keys cannot have surrounding whitespace.");
    requireText(description, "Choice description");
    return [key, description];
  });
}

function instruction(question: string, entries: [string, string][],
  format: CandidateFormat): { content: string; candidates: readonly string[] } {
  if (format === "labels") {
    if (entries.length > LABELS.length) {
      throw new TypeError("The labels mode supports at most 26 choices.");
    }
    const labels = entries.map((_, index) => LABELS[index]!);
    const options = Object.fromEntries(entries.map(([, description], index) =>
      [labels[index]!, description]));
    return {
      content: [
        "Choose the best option for the question using the preceding context.",
        "Answer with exactly one uppercase option label and no other text.",
        "", `Question: ${JSON.stringify(question)}`,
        "", `Options: ${JSON.stringify(options, null, 2)}`,
      ].join("\n"),
      candidates: Object.freeze(labels),
    };
  }
  return {
    content: [
      "Choose the best option for the question using the preceding context.",
      "Answer with exactly one option key as a quoted JSON string, followed by a newline.",
      "Do not include an explanation.",
      "", `Question: ${JSON.stringify(question)}`,
      "", `Options: ${JSON.stringify(Object.fromEntries(entries), null, 2)}`,
    ].join("\n"),
    candidates: Object.freeze(entries.map(([key]) => `${JSON.stringify(key)}\n`)),
  };
}

function result<K extends string>(keys: readonly K[], scores: readonly number[],
  boundaryTokens: number, usage?: Usage): Decision<K> {
  let best = 0;
  for (let i = 1; i < scores.length; i++) {
    if (scores[i]! > scores[best]!) best = i;
  }
  const maximum = scores[best]!;
  if (maximum === -Infinity) throw new ScoringError("Every candidate has zero likelihood.");
  const weights = scores.map((score) => Math.exp(score - maximum));
  const total = weights.reduce((sum, weight) => sum + weight, 0);
  const probabilities = weights.map((weight) => weight / total);
  let second = 0;
  let entropy = 0;
  for (let i = 0; i < probabilities.length; i++) {
    const p = probabilities[i]!;
    if (i !== best && p > second) second = p;
    if (p > 0) entropy -= p * Math.log(p);
  }
  const record = (values: readonly number[]): Readonly<Record<K, number>> =>
    Object.freeze(Object.fromEntries(keys.map((key, i) => [key, values[i]!]))) as Record<K, number>;
  return Object.freeze({
    choice: keys[best]!,
    distribution: record(probabilities),
    scores: record(scores),
    margin: probabilities[best]! - second,
    entropy,
    boundaryTokens,
    ...(usage === undefined ? {} : { usage }),
  });
}

function validateUsage(value: unknown): Usage | undefined {
  if (value === undefined) return undefined;
  if (!isRecord(value)) throw new ScoringError("usage must be an object.");
  const { promptTokens, cachedTokens, completionTokens, requests } = value;
  if (!isCount(promptTokens) || !isCount(completionTokens) || !isCount(requests)) {
    throw new ScoringError("Usage counts must be nonnegative safe integers.");
  }
  if (cachedTokens !== null && (!isCount(cachedTokens) || cachedTokens > promptTokens)) {
    throw new ScoringError("cachedTokens must be null or an integer between 0 and promptTokens.");
  }
  return Object.freeze({ promptTokens, cachedTokens, completionTokens, requests });
}

export function createFormattedChooser(score: Scorer, options: ChooserOptions,
  candidateFormat: CandidateFormat): Chooser {
  if (typeof score !== "function") throw new TypeError("score must be a function.");
  const { formatPrompt } = options;
  if (formatPrompt !== undefined && typeof formatPrompt !== "function") {
    throw new TypeError("formatPrompt must be a function.");
  }
  return async <const C extends Choices>(request: ChoiceRequest<C>) => {
    const { context, question, choices, signal } = request;
    signal?.throwIfAborted();
    if (typeof context !== "string") throw new TypeError("context must be a string.");
    requireText(question, "question");
    const entries = snapshot(choices);
    const keys = entries.map(([key]) => key) as ChoiceKey<C>[];
    const prepared = instruction(question, entries, candidateFormat);
    const prompt = formatPrompt
      ? await formatPrompt(Object.freeze({ context, instruction: prepared.content }))
      : `${context}\n\n${prepared.content}\n\nAnswer: `;
    requireText(prompt, "Formatted prompt");
    if (!prompt.startsWith(context)) {
      throw new TypeError("formatPrompt must preserve context as an unchanged prefix.");
    }
    signal?.throwIfAborted();
    const scored = await score(Object.freeze({
      prompt, candidates: prepared.candidates, ...(signal ? { signal } : {}),
    }));
    signal?.throwIfAborted();
    if (!isRecord(scored) || !Array.isArray(scored.logprobs)
      || scored.logprobs.length !== keys.length) {
      throw new ScoringError("The scorer must return one logprob per candidate.");
    }
    const values: number[] = [];
    for (const value of scored.logprobs) {
      if (!isLogprob(value)) throw new ScoringError("Logprobs must be numbers <= 0, not NaN.");
      values.push(value);
    }
    const boundaryTokens = scored.boundaryTokens === undefined ? 0 : scored.boundaryTokens;
    if (!isCount(boundaryTokens)) {
      throw new ScoringError("boundaryTokens must be a nonnegative safe integer.");
    }
    return result(keys, values, boundaryTokens, validateUsage(scored.usage));
  };
}
