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
};

type Results<Q extends Questions> = Readonly<{
  [K in keyof Q]: Decision<ChoiceKey<Q[K]["choices"]>>;
}>;

export async function chooseMany<const Q extends Questions>(
  choose: Chooser, request: Request<Q>,
): Promise<Results<Q>> {
  if (typeof choose !== "function") throw new TypeError("choose must be a function.");
  const { context, questions, images, signal } = request;
  signal?.throwIfAborted();
  if (!isRecord(questions) || Object.getOwnPropertySymbols(questions).length !== 0) {
    throw new TypeError("questions must be an object of named questions.");
  }
  const entries = Object.entries(questions);
  if (entries.length === 0) throw new TypeError("Provide at least one question.");

  const decisions = await Promise.all(entries.map(async ([name, { question, choices }]) =>
    [name, await choose({
      context, question, choices,
      ...(images === undefined ? {} : { images }),
      ...(signal === undefined ? {} : { signal }),
    })] as const));
  return Object.freeze(Object.fromEntries(decisions)) as Results<Q>;
}
