import assert from "node:assert/strict";
import { test } from "node:test";
import { fromLlamaCpp } from "../dist/esm/llama-cpp.js";
import { ScoringError } from "../dist/esm/index.js";
import { encode } from "./helpers.mjs";

const choiceRequest = Object.freeze({
  context: "A production change is pending.",
  question: "What should happen next?",
  choices: Object.freeze({ wait: "Wait for approval.", deploy: "Deploy immediately." }),
});

function fixture({ tokenizer = encode, logprob = () => -1,
  topTokenIds = (target) => [target], transform } = {}) {
  const calls = [];
  const fetch = async (url, init) => {
    const path = new URL(url).pathname;
    const body = JSON.parse(init.body);
    calls.push({ path, body, headers: init.headers });
    if (path === "/tokenize") return json({ tokens: tokenizer(body.content, body.add_special) });
    if (path !== "/completion") return json({}, 404);

    const target = body.logit_bias[0][0];
    const value = {
      tokens: [target],
      completion_probabilities: [{
        id: target,
        logprob: logprob(target),
        top_logprobs: topTokenIds(target, body).map((id) => ({ id, logprob: logprob(id) })),
      }],
      generation_settings: { post_sampling_probs: false, backend_sampling: false },
      tokens_evaluated: body.prompt.length,
      tokens_predicted: 1,
      timings: { cache_n: 0 },
    };
    return json(transform ? transform(value, body) : value);
  };
  return { calls, fetch };
}

function json(value, status = 200) {
  return new Response(JSON.stringify(value), {
    status, headers: { "content-type": "application/json" },
  });
}

const completions = (f) => f.calls.filter((call) => call.path === "/completion");

test("maps label scores back to choice keys and reuses sibling logprobs", async () => {
  const a = encode("A")[0];
  const b = encode("B")[0];
  const scores = new Map([[a, -2], [b, -0.25]]);
  const f = fixture({
    logprob: (id) => scores.get(id) ?? -10,
    topTokenIds: () => [a, b],
  });
  const choose = fromLlamaCpp({
    baseURL: "http://localhost:8080/v1", model: "local-model", fetch: f.fetch,
  });

  const decision = await choose(choiceRequest);

  assert.equal(decision.choice, "deploy");
  assert.deepEqual(decision.scores, { wait: -2, deploy: -0.25 });
  assert.equal(decision.usage.requests, f.calls.length);
  assert.equal(completions(f).length, 1);
  assert.equal(completions(f)[0].body.model, "local-model");
});

test("scores candidates separately when top logprobs contain none of them", async () => {
  const scores = new Map([[encode("A")[0], -0.3], [encode("B")[0], -1.4]]);
  const f = fixture({
    logprob: (id) => scores.get(id) ?? -10,
    topTokenIds: () => [999],
  });

  const decision = await fromLlamaCpp({ baseURL: "http://localhost:8080", fetch: f.fetch })(choiceRequest);

  assert.equal(decision.choice, "wait");
  assert.deepEqual(decision.scores, { wait: -0.3, deploy: -1.4 });
  assert.equal(completions(f).length, 2);
  assert.equal(completions(f)[1].body.n_probs, 1);
  assert.equal(decision.usage.requests, f.calls.length);
});

test("minimal-prefix scores only the first token that distinguishes shared keys", async () => {
  const quote = encode('"')[0];
  const space = encode(" ")[0];
  const scores = new Map([[quote, -0.1], [space, -1.2]]);
  const f = fixture({
    logprob: (id) => scores.get(id) ?? -10,
    topTokenIds: () => [quote, space],
  });
  const choose = fromLlamaCpp({
    baseURL: "http://localhost:8080", mode: "minimal-prefix", fetch: f.fetch,
  });

  const decision = await choose({
    context: "Choose a product.",
    question: "Which product?",
    choices: { watermelon: "The fruit.", "watermelon juice": "A drink." },
  });

  assert.equal(decision.choice, "watermelon");
  assert.deepEqual(decision.scores, { watermelon: -0.1, "watermelon juice": -1.2 });
  assert.equal(completions(f).length, 1);
  assert.deepEqual(completions(f)[0].body.prompt.slice(-encode('"watermelon').length),
    encode('"watermelon'));
});

test("continues scoring a group that separates at a deeper branch", async () => {
  const a = encode("a")[0];
  const b = encode("b")[0];
  const scores = new Map([[a, -0.2], [b, -1]]);
  const f = fixture({
    logprob: (id) => scores.get(id) ?? -10,
    topTokenIds: () => [a, b],
  });
  const choose = fromLlamaCpp({
    baseURL: "http://localhost:8080", mode: "minimal-prefix", fetch: f.fetch,
  });

  const decision = await choose({
    context: "Choose a key.",
    question: "Which key?",
    choices: { a: "A", ba: "BA", bb: "BB" },
  });

  assert.deepEqual(decision.scores, { a: -0.2, ba: -1.2, bb: -2 });
  assert.equal(completions(f).length, 2);
  assert.equal(completions(f)[1].body.prompt.length,
    completions(f)[0].body.prompt.length + 1);
});

test("reports a tokenization-boundary rollback", async () => {
  const pairs = (text, special = false) => {
    const bytes = new TextEncoder().encode(text);
    const ids = special ? [0] : [];
    for (let i = 0; i < bytes.length; i += 2) {
      ids.push(i + 1 < bytes.length ? 257 + bytes[i] * 256 + bytes[i + 1] : bytes[i] + 1);
    }
    return ids;
  };
  const a = pairs("abcA").at(-1);
  const b = pairs("abcB").at(-1);
  const scores = new Map([[a, -0.1], [b, -1.1]]);
  const f = fixture({
    tokenizer: pairs,
    logprob: (id) => scores.get(id) ?? -10,
    topTokenIds: () => [a, b],
  });
  const choose = fromLlamaCpp({
    baseURL: "http://localhost:8080", fetch: f.fetch,
    formatPrompt: ({ context }) => context,
  });

  const decision = await choose({
    context: "abc", question: "Which?", choices: { first: "A", second: "B" },
  });

  assert.equal(decision.choice, "first");
  assert.equal(decision.boundaryTokens, 1);
  assert.equal(completions(f)[0].body.prompt.length, 1);
});

test("rejects a probability attached to a different token", async () => {
  const f = fixture({ transform: (value) => ({
    ...value,
    completion_probabilities: [{ ...value.completion_probabilities[0], id: 999 }],
  }) });
  const choose = fromLlamaCpp({ baseURL: "http://localhost:8080", fetch: f.fetch });

  await assert.rejects(choose(choiceRequest),
    (error) => error instanceof ScoringError && /different token/.test(error.message));
});

test("rejects conflicting logprobs for the forced token", async () => {
  const f = fixture({ transform: (value) => {
    const forced = value.completion_probabilities[0];
    return {
      ...value,
      completion_probabilities: [{
        ...forced,
        top_logprobs: forced.top_logprobs.map((entry) => entry.id === forced.id
          ? { ...entry, logprob: entry.logprob - 0.5 }
          : entry),
      }],
    };
  } });
  const choose = fromLlamaCpp({ baseURL: "http://localhost:8080", fetch: f.fetch });

  await assert.rejects(choose(choiceRequest),
    (error) => error instanceof ScoringError && /conflicting logprobs/.test(error.message));
});
