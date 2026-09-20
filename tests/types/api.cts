import { createChooser, type Scorer } from "choosekit";
import { fromLlamaCpp } from "choosekit/llama-cpp";
import { fromOpenRouter } from "choosekit/openrouter";
const score: Scorer = ({ candidates }) => ({ logprobs: candidates.map(() => -1) });
const choose = createChooser(score);
void choose({ context: "", question: "Next?", choices: { test: "Test", done: "Done" } });
void fromLlamaCpp({ baseURL: "http://127.0.0.1:8080" });
void fromOpenRouter({ apiKey: "test-key", model: "test/model" });
