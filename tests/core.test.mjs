import assert from "node:assert/strict";
import { test } from "node:test";
import { createChooser, ScoringError } from "../dist/esm/index.js";
import { random, request } from "./helpers.mjs";

const fixed = (logprobs, extra = {}) => createChooser(() => ({ logprobs, ...extra }));

test("returns the selected choice and complete normalized distribution", async () => {
  const decision = await fixed([Math.log(0.1), Math.log(0.8), Math.log(0.1)])(request);
  assert.equal(decision.choice, "test");
  assert.ok(Math.abs(decision.distribution.test - 0.8) < 1e-14);
  assert.ok(Math.abs(decision.margin - 0.7) < 1e-14);
  assert.ok(Math.abs(decision.entropy - (-(0.2 * Math.log(0.1) + 0.8 * Math.log(0.8)))) < 1e-14);
  assert.equal(decision.boundaryTokens, 0);
  assert.equal("confidence" in decision, false);
  assert.equal("calibrated" in decision, false);
});

test("ties use Object.entries order, not alphabetical order", async () => {
  const d = await fixed([-2, -2, -2])(request);
  assert.equal(d.choice, "edit");
  assert.equal(d.margin, 0);
  assert.equal(d.distribution.edit, 1 / 3);
  assert.equal(d.entropy, Math.log(3));
});

test("stable softmax handles large negative values and impossible choices", async () => {
  const d = await fixed([-1e300, -1e300, -Infinity])(request);
  assert.deepEqual(d.distribution, { edit: 0.5, test: 0.5, done: 0 });
  assert.equal(d.entropy, Math.log(2));
});

test("never returns a uniform distribution for all-impossible candidates", async () => {
  await assert.rejects(fixed([-Infinity, -Infinity, -Infinity])(request), ScoringError);
});

test("underflow is allowed and does not create NaN entropy", async () => {
  const d = await fixed([0, -10000, -Infinity])(request);
  assert.deepEqual(d.distribution, { edit: 1, test: 0, done: 0 });
  assert.equal(d.entropy, 0);
  assert.equal(d.margin, 1);
});

test("scores one complete escaped and terminated key per candidate", async () => {
  const choices = Object.fromEntries([
    ["read", "Read."], ["read_more", "Read more."], ["цитата\"\\\n🙂", "Unicode."],
  ]);
  let called = 0;
  const choose = createChooser((input) => {
    called++;
    assert.deepEqual(input.candidates, Object.keys(choices).map((k) => JSON.stringify(k) + "\n"));
    assert.ok(input.prompt.includes(JSON.stringify(choices, null, 2)));
    assert.ok(input.prompt.startsWith(request.context));
    assert.ok(input.prompt.endsWith("Answer: "));
    return { logprobs: [-3, -2, -1] };
  });
  const d = await choose({ ...request, choices });
  assert.equal(d.choice, Object.keys(choices)[2]);
  assert.equal(called, 1);
});

test("snapshots image inputs before an asynchronous boundary", async () => {
  let release;
  const gate = new Promise((resolve) => { release = resolve; });
  const image = { mediaType: "image/png", base64: "aW1hZ2U=" };
  const images = [image];
  const choose = createChooser(({ images }) => {
    assert.deepEqual(images, [{ mediaType: "image/png", base64: "aW1hZ2U=" }]);
    return { logprobs: [-1, -2, -3] };
  }, {
    formatPrompt: async ({ context, instruction }) => { await gate; return context + instruction; },
  });
  const pending = choose({ ...request, images });
  image.base64 = "Y2hhbmdlZA==";
  images.push({ mediaType: "image/jpeg", base64: "YWRkZWQ=" });
  release();
  await pending;
});

test("empty context is supported", async () => {
  const d = await fixed([-1, -2, -3])({ ...request, context: "" });
  assert.equal(d.choice, "edit");
});

test("result and its nested records cannot be changed", async () => {
  const d = await fixed([-3, -1, -2])(request);
  for (const item of [d, d.distribution, d.scores]) assert.ok(Object.isFrozen(item));
  assert.throws(() => { d.choice = "done"; }, TypeError);
  assert.throws(() => { d.distribution.test = 0; }, TypeError);
  assert.throws(() => { d.scores.test = 0; }, TypeError);
});

test("dangerous object property names remain ordinary choice keys", async () => {
  const choices = Object.fromEntries([
    ["__proto__", "First"], ["constructor", "Second"], ["hasOwnProperty", "Third"],
  ]);
  const d = await fixed([-1, -2, -3])({ ...request, choices });
  assert.equal(d.choice, "__proto__");
  assert.equal(Object.getPrototypeOf(d.distribution), Object.prototype);
  assert.equal(typeof d.distribution.__proto__, "number");
  assert.equal(Object.hasOwn(d.distribution, "constructor"), true);
  assert.equal({}.polluted, undefined);
});

test("null-prototype records and integer keys are supported", async () => {
  const choices = Object.assign(Object.create(null), { 10: "Ten", 2: "Two" });
  const d = await fixed([-1, -2])({ ...request, choices });
  assert.equal(d.choice, "2");
});

test("ignores inherited choices", async () => {
  const choices = Object.assign(Object.create({ inherited: "Do not include me." }), { a: "A", b: "B" });
  assert.deepEqual(Object.keys((await fixed([-1, -2])({ ...request, choices })).scores), ["a", "b"]);
});

for (const [name, change] of [
  ["missing context", { context: undefined }], ["object context", { context: {} }],
  ["empty question", { question: " " }], ["numeric question", { question: 1 }],
  ["array choices", { choices: ["a", "b"] }], ["null choices", { choices: null }],
  ["no choices", { choices: {} }], ["one choice", { choices: { a: "A" } }],
  ["blank key", { choices: { "": "A", b: "B" } }],
  ["spaced key", { choices: { " a": "A", b: "B" } }],
  ["blank description", { choices: { a: " ", b: "B" } }],
  ["numeric description", { choices: { a: 1, b: "B" } }],
  ["symbol key", { choices: { a: "A", b: "B", [Symbol("c")]: "C" } }],
  ["non-array images", { images: {} }],
  ["unsupported image type", { images: [{ mediaType: "image/bmp", base64: "YQ==" }] }],
  ["empty image data", { images: [{ mediaType: "image/png", base64: "" }] }],
]) {
  test(`rejects ${name} before invoking the scorer`, async () => {
    let called = false;
    const choose = createChooser(() => { called = true; return { logprobs: [] }; });
    await assert.rejects(choose({ ...request, ...change }), TypeError);
    assert.equal(called, false);
  });
}

for (const bad of [NaN, Infinity, 0.01, "-1", null, undefined, true]) {
  test(`rejects invalid logprob ${String(bad)}`, async () => {
    await assert.rejects(fixed([-1, bad, -2])(request), ScoringError);
  });
}

for (const scored of [null, undefined, [], {}, { logprobs: [-1] },
  { logprobs: [-1, -2, -3, -4] }, { logprobs: new Float64Array([-1, -2, -3]) },
  { logprobs: Array(3) }]) {
  test(`rejects malformed score batch ${JSON.stringify(scored)}`, async () => {
    await assert.rejects(createChooser(() => scored)(request), ScoringError);
  });
}

for (const value of [-1, 0.5, Infinity, NaN, "1", null, Number.MAX_SAFE_INTEGER + 1]) {
  test(`rejects invalid boundaryTokens ${String(value)}`, async () => {
    await assert.rejects(fixed([-1, -2, -3], { boundaryTokens: value })(request), ScoringError);
  });
}

test("reports a tokenizer boundary rollback", async () => {
  assert.equal((await fixed([-1, -2, -3], { boundaryTokens: 2 })(request)).boundaryTokens, 2);
});

test("does not retry or replace scorer exceptions", async () => {
  const original = new Error("Connection failed");
  let calls = 0;
  const choose = createChooser(() => { calls++; throw original; });
  await assert.rejects(choose(request), (error) => error === original);
  assert.equal(calls, 1);
});

test("supports an async scorer and asynchronous chat formatting", async () => {
  const choose = createChooser(async ({ prompt }) => {
    assert.ok(prompt.startsWith(request.context + "<user>"));
    assert.ok(prompt.endsWith("<assistant>"));
    assert.ok(!prompt.endsWith("Answer: "));
    return { logprobs: [-1, -2, -3] };
  }, { formatPrompt: async ({ context, instruction }) => `${context}<user>${instruction}<assistant>` });
  assert.equal((await choose(request)).choice, "edit");
});

test("rejects invalid configuration and formatter output", async () => {
  assert.throws(() => createChooser(null), TypeError);
  assert.throws(() => createChooser(() => ({}), { formatPrompt: 2 }), TypeError);
  for (const value of ["", " ", 2, null]) {
    await assert.rejects(createChooser(() => ({ logprobs: [-1, -2, -3] }),
      { formatPrompt: () => value })(request), TypeError);
  }
});

test("checks cancellation before expensive work", async () => {
  const controller = new AbortController();
  const reason = new Error("Cancelled");
  controller.abort(reason);
  let called = false;
  await assert.rejects(createChooser(() => { called = true; })({ ...request, signal: controller.signal }),
    (error) => error === reason);
  assert.equal(called, false);
});

test("checks cancellation after formatting and before scoring", async () => {
  const controller = new AbortController();
  let called = false;
  const choose = createChooser(() => { called = true; }, {
    formatPrompt: ({ context, instruction, signal }) => {
      assert.equal(signal, controller.signal);
      controller.abort();
      return context + instruction;
    },
  });
  await assert.rejects(choose({ ...request, signal: controller.signal }), { name: "AbortError" });
  assert.equal(called, false);
});

test("passes the same AbortSignal and rejects a late result after cancellation", async () => {
  const controller = new AbortController();
  const choose = createChooser(({ signal }) => {
    assert.equal(signal, controller.signal);
    controller.abort();
    return { logprobs: [-1, -2, -3] };
  });
  await assert.rejects(choose({ ...request, signal: controller.signal }), { name: "AbortError" });
});

test("snapshots input choices before an asynchronous boundary", async () => {
  let release;
  const gate = new Promise((resolve) => { release = resolve; });
  const choices = { a: "Original A", b: "Original B" };
  const choose = createChooser(({ candidates, prompt }) => {
    assert.deepEqual(candidates, ['"a"\n', '"b"\n']);
    assert.ok(prompt.includes("Original A"));
    return { logprobs: [-1, -2] };
  }, { formatPrompt: async ({ context, instruction }) => { await gate; return context + instruction; } });
  const pending = choose({ ...request, choices });
  delete choices.a;
  choices.b = "Changed";
  choices.c = "Added";
  release();
  assert.equal((await pending).choice, "a");
});

test("concurrent decisions do not share mutable context", async () => {
  const choose = createChooser(async ({ candidates }) => {
    await new Promise((resolve) => setImmediate(resolve));
    return { logprobs: candidates.map((_, i) => -i - 1) };
  });
  const outputs = await Promise.all(Array.from({ length: 50 }, (_, i) =>
    choose({ ...request, choices: { [`${i}:a`]: "A", [`${i}:b`]: "B" } })));
  outputs.forEach((output, i) => assert.equal(output.choice, `${i}:a`));
});

test("1000 seeded distributions match an independent normalization and preserve shifts", async () => {
  const next = random();
  for (let run = 0; run < 1000; run++) {
    const n = [2, 4, 8, 16][run % 4];
    const choices = Object.fromEntries(Array.from({ length: n }, (_, i) => [`k${i}`, `Option ${i}`]));
    const weights = Array.from({ length: n }, () => next() + 0.001);
    const sum = weights.reduce((a, b) => a + b, 0);
    const logs = weights.map((p) => Math.log(p / 2));
    const plain = await fixed(logs)({ ...request, choices });
    const shifted = await fixed(logs.map((s) => s - 10000))({ ...request, choices });
    let actualSum = 0;
    for (let i = 0; i < n; i++) {
      const key = `k${i}`;
      assert.ok(Math.abs(plain.distribution[key] - weights[i] / sum) < 1e-14);
      assert.ok(Math.abs(plain.distribution[key] - shifted.distribution[key]) < 1e-12);
      actualSum += plain.distribution[key];
    }
    assert.ok(Math.abs(actualSum - 1) < 1e-14);
    assert.equal(plain.choice, shifted.choice);
    assert.ok(plain.entropy >= 0 && plain.entropy <= Math.log(n) + 1e-14);
  }
});

test("formatting cannot silently replace, escape, or prepend to the current context", async () => {
  for (const formatPrompt of [
    ({ context, instruction }) => `prefix${context}${instruction}`,
    ({ context, instruction }) => JSON.stringify(context) + instruction,
    ({ instruction }) => instruction,
  ]) {
    let called = false;
    const choose = createChooser(() => { called = true; return { logprobs: [-1, -2, -3] }; }, { formatPrompt });
    await assert.rejects(choose(request), /unchanged prefix/);
    assert.equal(called, false);
  }
});

test("Unicode and raw whitespace remain unchanged at the start of the prompt", async () => {
  const context = ' \n\t"quoted"\\unescaped / 🙂\r\n';
  const choose = createChooser(({ prompt }) => {
    assert.equal(prompt.slice(0, context.length), context);
    return { logprobs: [-1, -2, -3] };
  });
  await choose({ ...request, context });
});

test("usage is copied and frozen without inventing cache metrics", async () => {
  const usage = { promptTokens: 123, cachedTokens: null, completionTokens: 4, requests: 6 };
  const d = await fixed([-1, -2, -3], { usage })(request);
  assert.deepEqual(d.usage, usage);
  assert.ok(Object.isFrozen(d.usage));
  usage.promptTokens = 999;
  assert.equal(d.usage.promptTokens, 123);
  assert.equal(d.usage.cachedTokens, null);
});

for (const usage of [null, [], {}, { promptTokens: -1, cachedTokens: 0, completionTokens: 0, requests: 0 },
  { promptTokens: 1, cachedTokens: 2, completionTokens: 1, requests: 1 },
  { promptTokens: 1, cachedTokens: undefined, completionTokens: 1, requests: 1 },
  { promptTokens: 1, cachedTokens: 0, completionTokens: 1.5, requests: 1 },
  { promptTokens: 1, cachedTokens: 0, completionTokens: 1, requests: Infinity },
  { promptTokens: 1, cachedTokens: -1, completionTokens: 1, requests: 1 },
]) {
  test(`rejects inconsistent custom scorer usage ${JSON.stringify(usage)}`, async () => {
    await assert.rejects(fixed([-1, -2, -3], { usage })(request), ScoringError);
  });
}
