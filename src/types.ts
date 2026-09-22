export type Choices = Readonly<Record<string, string>>;
export type ChoiceKey<C extends Choices> = `${Extract<keyof C, string | number>}`;

export type ImageMediaType = "image/png" | "image/jpeg" | "image/webp";

export interface ImageInput {
  readonly mediaType: ImageMediaType;
  /** Raw base64 data without a data URL prefix. */
  readonly base64: string;
}

export interface ChoiceRequest<C extends Choices> {
  /** Existing context, copied unchanged to the beginning of the scoring prompt. */
  readonly context: string;
  readonly question: string;
  readonly choices: C;
  readonly images?: readonly ImageInput[];
  readonly signal?: AbortSignal;
}

export interface Usage {
  /** Counts repeated prefixes across inference requests, not unique context tokens. */
  readonly promptTokens: number;
  /** Server-reported cache reads. Null when any inference response omits this metric. */
  readonly cachedTokens: number | null;
  readonly completionTokens: number;
  /** All adapter HTTP calls, including tokenization; excludes SDK-level retries. */
  readonly requests: number;
}

export interface Decision<K extends string> {
  readonly choice: K;
  readonly distribution: Readonly<Record<K, number>>;
  readonly scores: Readonly<Record<K, number>>;
  /** Difference between the largest and second-largest probabilities. */
  readonly margin: number;
  /** Shannon entropy in nats, not a probability of correctness. */
  readonly entropy: number;
  /** Prompt tokens rolled back because tokenization crossed the answer boundary. */
  readonly boundaryTokens: number;
  readonly usage?: Usage;
}

export interface ScoreRequest {
  readonly prompt: string;
  readonly candidates: readonly string[];
  readonly images?: readonly ImageInput[];
  readonly signal?: AbortSignal;
}

export interface Scores {
  /** Comparable conditional log-probability scores, in candidate order, using natural logs. */
  readonly logprobs: readonly number[];
  readonly boundaryTokens?: number;
  readonly usage?: Usage;
}

export type Scorer = (request: ScoreRequest) => Scores | PromiseLike<Scores>;
export type Chooser = <const C extends Choices>(request: ChoiceRequest<C>) =>
  Promise<Decision<ChoiceKey<C>>>;

export interface PromptInput {
  readonly context: string;
  readonly instruction: string;
  readonly signal?: AbortSignal;
}

export interface ChooserOptions {
  /** Append model-specific chat turns. The result must start with context unchanged. */
  readonly formatPrompt?: (input: PromptInput) => string | PromiseLike<string>;
}
