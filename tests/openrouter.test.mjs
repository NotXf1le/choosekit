import assert from "node:assert/strict";
import { test } from "node:test";
import { fromOpenRouter } from "../dist/esm/openrouter.js";
import { ScoringError } from "../dist/esm/index.js";

const request = Object.freeze({
  context: "A customer reports that the same invoice was charged twice.",
  question: "Which team should handle this message?",
  choices: Object.freeze({
    sales: "Pricing, upgrades, and new accounts.",
    technical: "Bugs, outages, and integrations.",
    billing: "Payments, invoices, and refunds.",
    security: "Account compromise and suspicious activity.",
    support: "General product questions.",
  }),
});

const unorderedTopLogprobs = Object.freeze([
  { token: "A", bytes: [65], logprob: -1.2338635921 },
  { token: "C", bytes: [67], logprob: -1.1088635921 },
  { token: "E", bytes: [69], logprob: -1.7338635921 },
  { token: "B", bytes: [66], logprob: -2.2338635921 },
  { token: "D", bytes: [68], logprob: -2.3588635921 },
  { token: "c", bytes: [99], logprob: -8 },
]);

function scoredPosition(topLogprobs = unorderedTopLogprobs) {
  return {
    token: "A", bytes: [65], logprob: -1.2338635921, top_logprobs: topLogprobs,
  };
}

function response(topLogprobs = unorderedTopLogprobs, overrides = {}) {
  return {
    choices: [{
      finish_reason: "length",
      message: { role: "assistant", content: "A" },
      logprobs: { content: [scoredPosition(topLogprobs)] },
      ...overrides,
    }],
  };
}

function json(value, status = 200) {
  return new Response(JSON.stringify(value), {
    status, headers: { "content-type": "application/json" },
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
  return fromOpenRouter({
    apiKey: "test-secret", model: "test/model", fetch: f.fetch, ...extra,
  });
}

test("uses first-position label logprobs instead of the sampled token", async () => {
  const value = response();
  value.usage = {
    prompt_tokens: 180,
    completion_tokens: 1,
    prompt_tokens_details: { cached_tokens: 128 },
  };
  const f = fixture(value);
  const decision = await chooser(f)(request);

  assert.equal(decision.choice, "billing");
  assert.deepEqual(decision.scores, {
    sales: -1.2338635921,
    technical: -2.2338635921,
    billing: -1.1088635921,
    security: -2.3588635921,
    support: -1.7338635921,
  });
  for (const [key, percent] of Object.entries({
    sales: 29.14, technical: 10.72, billing: 33.02, security: 9.46, support: 17.67,
  })) {
    assert.ok(Math.abs(decision.distribution[key] * 100 - percent) < 0.01, key);
  }
  assert.deepEqual(decision.usage, {
    promptTokens: 180, cachedTokens: 128, completionTokens: 1, requests: 1,
  });

  assert.equal(f.calls.length, 1);
  const call = f.calls[0];
  assert.equal(call.url, "https://openrouter.ai/api/v1/chat/completions");
  assert.equal(call.init.method, "POST");
  const headers = new Headers(call.init.headers);
  assert.equal(headers.get("authorization"), "Bearer test-secret");
  assert.equal(headers.get("content-type"), "application/json");
  const { messages, ...body } = call.body;
  assert.deepEqual(body, {
    model: "test/model",
    max_tokens: 1,
    stream: false,
    temperature: 1,
    top_p: 1,
    logprobs: true,
    top_logprobs: 20,
    reasoning_effort: "none",
  });
  assert.equal(messages.length, 1);
  assert.equal(messages[0].role, "user");
  assert.ok(messages[0].content.startsWith(request.context));
  assert.ok(messages[0].content.includes(JSON.stringify(request.question)));
  for (const [index, description] of Object.values(request.choices).entries()) {
    const label = String.fromCharCode(65 + index);
    assert.ok(messages[0].content.includes(`"${label}": "${description}"`));
  }
  assert.ok(messages[0].content.endsWith("Answer: "));
});

test("pins an optional provider without fallback", async () => {
  const f = fixture();
  await chooser(f, { provider: "reka" })(request);
  assert.deepEqual(f.calls[0].body.provider, {
    only: ["reka"], allow_fallbacks: false,
  });
});

test("assigns zero probability to labels omitted from top logprobs", async () => {
  const f = fixture(response(unorderedTopLogprobs.filter(({ token }) => token !== "C")));
  const decision = await chooser(f)(request);

  assert.equal(decision.choice, "sales");
  assert.equal(decision.scores.billing, -Infinity);
  assert.equal(decision.distribution.billing, 0);
  assert.ok(Math.abs(Object.values(decision.distribution)
    .reduce((sum, probability) => sum + probability, 0) - 1) < 1e-12);
});

const malformed = [
  ["null logprobs", response(undefined, { logprobs: null }), /logprobs/i],
  ["no scored position", response(undefined, { logprobs: { content: [] } }), /exactly one/i],
  ["multiple scored positions", response(undefined, {
    logprobs: { content: [scoredPosition(), scoredPosition()] },
  }), /exactly one/i],
  ["no choice labels", response([{ token: "x", bytes: [120], logprob: -0.1 }]),
    /any choice label/i],
  ["duplicate required label", response([...unorderedTopLogprobs, unorderedTopLogprobs[0]]), /duplicate.*A|A.*duplicate/i],
  ["invalid logprob", response(unorderedTopLogprobs.map((entry) =>
    entry.token === "C" ? { ...entry, logprob: 0.1 } : entry)), /logprob/i],
  ["clamped logprob", response(unorderedTopLogprobs.map((entry) =>
    entry.token === "C" ? { ...entry, logprob: -9999 } : entry)), /clamped/i],
  ["mismatched bytes", response(unorderedTopLogprobs.map((entry) =>
    entry.token === "C" ? { ...entry, bytes: [99] } : entry)), /bytes/i],
  ["content filtering", response(unorderedTopLogprobs, { finish_reason: "content_filter" }),
    /refused|filtered/i],
  ["message refusal", response(undefined, {
    message: { role: "assistant", content: null, refusal: "Cannot answer." },
  }), /refused|filtered/i],
  ["logprob refusal", response(undefined, {
    logprobs: { content: [scoredPosition()], refusal: [{ token: "refusal" }] },
  }), /refused|filtered/i],
];

for (const [name, value, pattern] of malformed) {
  test(`rejects ${name}`, async () => {
    const f = fixture(value);
    await assert.rejects(chooser(f)(request),
      (error) => error instanceof ScoringError && pattern.test(error.message));
  });
}

test("rejects more than 20 choices before sending a request", async () => {
  const f = fixture();
  const choices = Object.fromEntries(Array.from({ length: 21 }, (_, index) =>
    [`choice_${index}`, `Choice ${index}`]));
  await assert.rejects(chooser(f)({ ...request, choices }), /at most 20/i);
  assert.equal(f.calls.length, 0);
});

test("accepts exactly 20 choices", async () => {
  const labels = [..."ABCDEFGHIJKLMNOPQRST"];
  const choices = Object.fromEntries(labels.map((label) => [label, `Choice ${label}`]));
  const top = labels.map((token, index) => ({
    token, bytes: [token.charCodeAt(0)], logprob: -(index + 1),
  }));
  const f = fixture(response(top));

  const decision = await chooser(f)({ ...request, choices });

  assert.equal(decision.choice, "A");
  assert.equal(Object.keys(decision.distribution).length, 20);
  assert.equal(f.calls.length, 1);
});

test("passes AbortSignal to fetch", async () => {
  const controller = new AbortController();
  const f = fixture();
  f.fetch = async (_url, init) => {
    assert.equal(init.signal, controller.signal);
    controller.abort();
    throw controller.signal.reason;
  };
  await assert.rejects(chooser(f)({ ...request, signal: controller.signal }),
    (error) => error === controller.signal.reason);
});

test("does not include the API key or prompt in HTTP errors", async () => {
  const f = fixture({ error: { message: `bad request: test-secret ${request.context}` } }, 400);
  await assert.rejects(chooser(f)(request), (error) => {
    assert.match(error.message, /400/);
    assert.doesNotMatch(error.message, /test-secret|charged twice/);
    return true;
  });
  assert.equal(f.calls.length, 1);
});

test("does not retry invalid JSON responses", async () => {
  let calls = 0;
  const choose = fromOpenRouter({
    apiKey: "test-secret",
    model: "test/model",
    fetch: async () => {
      calls++;
      return new Response("{", { status: 200 });
    },
  });

  await assert.rejects(choose(request), /invalid JSON/i);
  assert.equal(calls, 1);
});
