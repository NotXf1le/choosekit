import { createChooser, type Scorer, type Decision } from "choosekit";
import { fromLlamaCpp } from "choosekit/llama-cpp";
import { fromOllama } from "choosekit/ollama";
import { fromOpenRouter } from "choosekit/openrouter";

const scorer: Scorer = async ({ candidates, signal }) => {
  signal?.throwIfAborted();
  return { logprobs: candidates.map(() => -1) };
};
const choose = createChooser(scorer);
const decision = await choose({
  context: "Changed", question: "Next?", choices: { edit: "Edit", test: "Test" },
});
const key: "edit" | "test" = decision.choice;
const score: number = decision.distribution[key];
const typed: Decision<"edit" | "test"> = decision;
void score; void typed;
// @ts-expect-error Unknown choices do not exist in the result.
decision.distribution.other;
// @ts-expect-error Results are read-only.
decision.distribution.test = 0;
// @ts-expect-error Choice descriptions must be strings.
choose({ context: "", question: "?", choices: { a: 1, b: "B" } });
// @ts-expect-error A generating model is not a scoring function.
createChooser(async () => "test");
// @ts-expect-error The library does not accept a model name in place of a scorer.
createChooser("model-name");

const numeric = await choose({ context: "", question: "?", choices: { 1: "One", 2: "Two" } });
const numericKey: "1" | "2" = numeric.choice;
void numericKey;

const local = fromLlamaCpp({ baseURL: "http://127.0.0.1:8080", mode: "labels" });
void local({ context: "", question: "?", choices: { yes: "Yes", no: "No" } });
const ollama = fromOllama({ model: "test-model" });
void ollama({ context: "", question: "?", choices: { yes: "Yes", no: "No" } });
const remote = fromOpenRouter({ apiKey: "test-key", model: "test/model" });
void remote({ context: "", question: "?", choices: { yes: "Yes", no: "No" } });

// @ts-expect-error The old implicit agent-state input is not part of this API.
choose({ state: "Changed", question: "Next?", choices: { yes: "Yes", no: "No" } });
const formatter = createChooser(scorer, {
  formatPrompt: ({ context, instruction, signal }) => {
    signal?.throwIfAborted();
    return `${context}\n${instruction}\nAnswer: `;
  },
});
void formatter;
const cached: number | null | undefined = decision.usage?.cachedTokens;
void cached;
// @ts-expect-error No calibrated correctness confidence is claimed.
decision.confidence;
