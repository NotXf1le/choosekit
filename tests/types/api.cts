import { chooseMany, createChooser, ScoringError, type Scorer } from "choosekit";
import { fromLlamaCpp } from "choosekit/llama-cpp";
import { fromOllama } from "choosekit/ollama";
import { fromOpenRouter } from "choosekit/openrouter";
const score: Scorer = ({ candidates }) => ({ logprobs: candidates.map(() => -1) });
const choose = createChooser(score);
void choose({ context: "", question: "Next?", choices: { test: "Test", done: "Done" } });
void fromLlamaCpp({ baseURL: "http://127.0.0.1:8080" });
void fromOllama({ model: "test-model" });
void fromOpenRouter({ apiKey: "test-key", model: "test/model" });
void chooseMany(choose, {
  concurrency: 2,
  context: "", questions: { route: { question: "Next?", choices: { test: "Test", done: "Done" } } },
}).then((decisions) => {
  const key: "test" | "done" = decisions.route.choice;
  void key;
});

const httpError = new ScoringError(undefined, { httpStatus: 401, cause: new Error("upstream") });
const httpStatus: number | undefined = httpError.httpStatus;
void httpStatus;
void new ScoringError();
void new ScoringError("legacy", { cause: new Error("cause") });
// @ts-expect-error HTTP status must be numeric.
new ScoringError("HTTP failed", { httpStatus: "401" });
// @ts-expect-error HTTP metadata is read-only.
httpError.httpStatus = 429;
