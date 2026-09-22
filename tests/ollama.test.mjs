import assert from "node:assert/strict";
import { test } from "node:test";
import { fromOllama } from "../dist/esm/ollama.js";
import { ScoringError } from "../dist/esm/index.js";

const request = Object.freeze({
  context: "A customer says their payout failed.",
  question: "Which team should handle this message?",
  choices: Object.freeze({
    sales: "Pricing, upgrades, and new accounts.",
    technical: "Bugs, outages, integrations, and API errors.",
    billing: "Payments, payouts, invoices, and refunds.",
  }),
});

const alternatives = Object.freeze([
  { token: "B", bytes: [66], logprob: -2.2 },
  { token: "C", bytes: [67], logprob: -0.1 },
  { token: "x", bytes: [120], logprob: -4 },
]);

function response(topLogprobs = alternatives, overrides = {}) {
  return {
    model: "test-model",
    message: { role: "assistant", content: "A" },
    done: true,
    done_reason: "length",
    prompt_eval_count: 120,
    prompt_eval_cached_count: 80,
    eval_count: 1,
    logprobs: [{
      token: "A",
      bytes: [65],
      logprob: -1.2,
      top_logprobs: topLogprobs,
    }],
    ...overrides,
  };
}

function json(value, status = 200) {
  return new Response(JSON.stringify(value), {
    status,
    headers: { "content-type": "application/json" },
  });
}

function fixture(value = response(), status = 200) {
  const calls = [];
  return {
    calls,
    fetch: async (url, init) => {
      calls.push({ url, init, body: JSON.parse(init.body) });
      return json(value, status);
    },
  };
}

function chooser(f, extra = {}) {
  return fromOllama({ model: "test-model", fetch: f.fetch, ...extra });
}

test("scores labels through the native chat endpoint", async () => {
  const f = fixture();
  const decision = await chooser(f)(request);

  assert.equal(decision.choice, "billing");
  assert.deepEqual(decision.scores, { sales: -1.2, technical: -2.2, billing: -0.1 });
  assert.deepEqual(decision.usage, {
    promptTokens: 120,
    cachedTokens: 80,
    completionTokens: 1,
    requests: 1,
  });

  assert.equal(f.calls.length, 1);
  const call = f.calls[0];
  assert.equal(call.url, "http://127.0.0.1:11434/api/chat");
  assert.equal(call.init.method, "POST");
  assert.equal(new Headers(call.init.headers).get("content-type"), "application/json");
  const { messages, ...body } = call.body;
  assert.deepEqual(body, {
    model: "test-model",
    stream: false,
    think: false,
    logprobs: true,
    top_logprobs: 20,
    options: { num_predict: 1 },
  });
  assert.deepEqual(messages.map(({ role }) => role), ["user"]);
  assert.ok(messages[0].content.startsWith(request.context));
});

test("normalizes an explicit API base URL", async () => {
  const f = fixture();
  await chooser(f, { baseURL: "https://example.test/ollama/api" })(request);
  assert.equal(f.calls[0].url, "https://example.test/ollama/api/chat");
});

test("sends raw base64 image data to Ollama", async () => {
  const f = fixture();
  await chooser(f)({
    ...request,
    images: [{ mediaType: "image/webp", base64: "d2VicA==" }],
  });

  assert.deepEqual(f.calls[0].body.messages[0].images, ["d2VicA=="]);
  assert.ok(f.calls[0].body.messages[0].content.startsWith(request.context));
});

test("rejects invalid configuration before sending a request", () => {
  const f = fixture();
  assert.throws(() => fromOllama({ model: "", fetch: f.fetch }), /model/i);
  assert.throws(() => chooser(f, { baseURL: "https://user:secret@example.test" }), /baseURL/i);
  assert.equal(f.calls.length, 0);
});

test("supports at most 20 choices", async () => {
  const labels = [..."ABCDEFGHIJKLMNOPQRST"];
  const choices = Object.fromEntries(labels.map((label) => [label, `Choice ${label}`]));
  const top = labels.slice(1).map((token, index) => ({
    token,
    bytes: [token.charCodeAt(0)],
    logprob: -(index + 2),
  }));
  const f = fixture(response(top, {
    logprobs: [{ token: "A", bytes: [65], logprob: -1, top_logprobs: top }],
  }));

  const decision = await chooser(f)({ ...request, choices });
  assert.equal(Object.keys(decision.distribution).length, 20);

  const tooMany = { ...choices, U: "Choice U" };
  await assert.rejects(chooser(f)({ ...request, choices: tooMany }), /at most 20/i);
  assert.equal(f.calls.length, 1);
});

test("assigns zero probability to omitted labels and rejects no label scores", async () => {
  const oneMissing = fixture(response(alternatives.filter(({ token }) => token !== "B"), {
    prompt_eval_cached_count: undefined,
  }));
  const decision = await chooser(oneMissing)(request);
  assert.equal(decision.scores.technical, -Infinity);
  assert.equal(decision.distribution.technical, 0);
  assert.equal(decision.usage.cachedTokens, null);

  const noLabels = fixture(response([
    { token: "x", bytes: [120], logprob: -0.1 },
  ], {
    logprobs: [{
      token: "y", bytes: [121], logprob: -0.2,
      top_logprobs: [{ token: "x", bytes: [120], logprob: -0.1 }],
    }],
  }));
  await assert.rejects(chooser(noLabels)(request),
    (error) => error instanceof ScoringError && /choice label/i.test(error.message));
});

test("passes AbortSignal to fetch and preserves its reason", async () => {
  const controller = new AbortController();
  const reason = new Error("cancelled by caller");
  const f = fixture();
  f.fetch = async (_url, init) => {
    assert.equal(init.signal, controller.signal);
    controller.abort(reason);
    throw reason;
  };

  await assert.rejects(chooser(f)({ ...request, signal: controller.signal }),
    (error) => error === reason);
});

const malformed = [
  ["null response", null, /scored token/i],
  ["missing logprobs", response(undefined, { logprobs: undefined }), /scored token/i],
  ["multiple scored positions", response(undefined, {
    logprobs: [response().logprobs[0], response().logprobs[0]],
  }), /one|position|logprobs/i],
  ["missing top logprobs", response(undefined, {
    logprobs: [{ token: "A", bytes: [65], logprob: -1.2 }],
  }), /top.*logprobs/i],
  ["conflicting selected token", response([
    ...alternatives, { token: "A", bytes: [65], logprob: -1.3 },
  ]), /conflict|duplicate/i],
  ["invalid logprob", response(alternatives.map((entry) =>
    entry.token === "C" ? { ...entry, logprob: 0.1 } : entry)), /logprob/i],
  ["mismatched bytes", response(alternatives.map((entry) =>
    entry.token === "C" ? { ...entry, bytes: [99] } : entry)), /bytes/i],
  ["invalid cached usage", response(undefined, { prompt_eval_cached_count: 121 }), /cache|usage/i],
];

for (const [name, value, pattern] of malformed) {
  test(`rejects ${name}`, async () => {
    const f = fixture(value);
    await assert.rejects(chooser(f)(request),
      (error) => error instanceof ScoringError && pattern.test(error.message));
  });
}

test("reports HTTP and JSON failures without exposing the prompt", async () => {
  const failed = fixture({ error: request.context }, 500);
  await assert.rejects(chooser(failed)(request), (error) => {
    assert.match(error.message, /500/);
    assert.doesNotMatch(error.message, new RegExp(request.context));
    return true;
  });

  const invalidJson = fixture();
  invalidJson.fetch = async () => new Response("{", { status: 200 });
  await assert.rejects(chooser(invalidJson)(request),
    (error) => error instanceof ScoringError && /invalid JSON/i.test(error.message));
});
