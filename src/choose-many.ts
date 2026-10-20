import type {
  ChoiceKey, Chooser, Choices, Decision, ImageInput,
} from "./types.js";
import { isRecord } from "./validation.js";

type Questions = Readonly<Record<string, { readonly question: string; readonly choices: Choices }>>;

type Request<Q extends Questions> = {
  readonly context: string;
  readonly questions: Q;
  readonly images?: readonly ImageInput[];
  readonly signal?: AbortSignal;
  /** Maximum simultaneous choose calls in this group. Omit to start all questions concurrently. */
  readonly concurrency?: number;
};

type Results<Q extends Questions> = Readonly<{
  [K in keyof Q]: Decision<ChoiceKey<Q[K]["choices"]>>;
}>;

export async function chooseMany<const Q extends Questions>(
  choose: Chooser, request: Request<Q>,
): Promise<Results<Q>> {
  if (typeof choose !== "function") throw new TypeError("choose must be a function.");
  const { context, questions, images, signal, concurrency } = request;
  signal?.throwIfAborted();
  if (!isRecord(questions) || Object.getOwnPropertySymbols(questions).length !== 0) {
    throw new TypeError("questions must be an object of named questions.");
  }
  if (concurrency !== undefined && (!Number.isSafeInteger(concurrency) || concurrency < 1)) {
    throw new TypeError("concurrency must be a positive safe integer.");
  }
  const entries = Object.entries(questions).map(([name, { question, choices }]) => [
    name, question,
    isRecord(choices) && Object.getOwnPropertySymbols(choices).length === 0
      ? Object.freeze({ ...choices }) : choices,
  ] as const);
  if (entries.length === 0) throw new TypeError("Provide at least one question.");
  const sharedImages = Array.isArray(images)
    ? Object.freeze(images.map((image: ImageInput) => isRecord(image)
      ? Object.freeze({ mediaType: image.mediaType, base64: image.base64 }) : image))
    : images;

  const decisions: [string, Decision<string>][] = new Array(entries.length);
  let next = 0;
  let failed = false;
  await Promise.all(Array.from({ length: Math.min(concurrency ?? entries.length, entries.length) }, async () => {
    try {
      while (!failed && next < entries.length) {
        signal?.throwIfAborted();
        const index = next++;
        const [name, question, choices] = entries[index]!;
        const decision = await choose({
          context, question, choices,
          ...(sharedImages === undefined ? {} : { images: sharedImages }),
          ...(signal === undefined ? {} : { signal }),
        });
        signal?.throwIfAborted();
        decisions[index] = [name, decision];
      }
    } catch (error) {
      failed = true;
      throw error;
    }
  }));
  return Object.freeze(Object.fromEntries(decisions)) as Results<Q>;
}
