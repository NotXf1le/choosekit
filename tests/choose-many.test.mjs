import assert from "node:assert/strict";
import { test } from "node:test";
import { chooseMany, createChooser } from "../dist/esm/index.js";

test("returns separate decisions for named questions with shared inputs", async () => {
  const controller = new AbortController();
  const images = [{ mediaType: "image/png", base64: "aW1hZ2U=" }];
  const seen = [];
  const choose = createChooser(({ prompt, candidates, images: receivedImages, signal }) => {
    seen.push({ prompt, receivedImages, signal });
    return {
      logprobs: candidates.map((_, index) => -index - 1),
      usage: {
        promptTokens: prompt.includes("Which team?") ? 10 : 20,
        cachedTokens: 2, completionTokens: 1, requests: 1,
      },
    };
  });
  const decisions = await chooseMany(choose, {
    context: "A payout failed.", images, signal: controller.signal,
    questions: {
      route: {
        question: "Which team?",
        choices: { billing: "Payments and refunds", technical: "Bugs and API errors" },
      },
      urgency: {
        question: "How urgent?",
        choices: { high: "Needs attention today", low: "Can wait" },
      },
    },
  });

  assert.deepEqual(Object.keys(decisions), ["route", "urgency"]);
  assert.equal(decisions.route.choice, "billing");
  assert.equal(decisions.urgency.choice, "high");
  assert.deepEqual(Object.keys(decisions.route.distribution), ["billing", "technical"]);
  assert.deepEqual(Object.keys(decisions.urgency.distribution), ["high", "low"]);
  assert.deepEqual(decisions.route.usage, {
    promptTokens: 10, cachedTokens: 2, completionTokens: 1, requests: 1,
  });
  assert.deepEqual(decisions.urgency.usage, {
    promptTokens: 20, cachedTokens: 2, completionTokens: 1, requests: 1,
  });
  assert.ok(Object.isFrozen(decisions));
  assert.equal(seen.length, 2);
  for (const call of seen) {
    assert.ok(call.prompt.startsWith("A payout failed."));
    assert.deepEqual(call.receivedImages, images);
    assert.equal(call.signal, controller.signal);
  }
});

test("starts questions concurrently and rejects when one fails", async () => {
  const failure = new Error("Backend failed");
  const started = [];
  let finishOther;
  let reportCompletion;
  const completed = new Promise((resolve) => { reportCompletion = resolve; });
  const choose = async ({ question }) => {
    started.push(question);
    if (question === "First?") throw failure;
    await new Promise((resolve) => { finishOther = resolve; });
    reportCompletion();
    return { choice: "yes" };
  };
  const pending = chooseMany(choose, {
    context: "Shared.",
    questions: {
      first: { question: "First?", choices: { yes: "Yes", no: "No" } },
      second: { question: "Second?", choices: { yes: "Yes", no: "No" } },
    },
  });
  await assert.rejects(pending, (error) => error === failure);
  assert.deepEqual(started, ["First?", "Second?"]);
  finishOther();
  await completed;
});

test("rejects an empty question set or an aborted signal before starting work", async () => {
  let called = false;
  const choose = async () => { called = true; throw new Error("Unexpected call"); };
  await assert.rejects(chooseMany(choose, { context: "", questions: {} }), TypeError);
  const signal = AbortSignal.abort();
  await assert.rejects(chooseMany(choose, {
    context: "", signal,
    questions: { one: { question: "One?", choices: { yes: "Yes", no: "No" } } },
  }), (error) => error === signal.reason);
  assert.equal(called, false);
});

test("passes a shared signal to every in-flight question", async () => {
  const controller = new AbortController();
  const reason = new Error("Cancelled");
  const received = [];
  const choose = ({ signal }) => new Promise((_, reject) => {
    received.push(signal);
    signal.addEventListener("abort", () => reject(signal.reason), { once: true });
  });
  const pending = chooseMany(choose, {
    context: "", signal: controller.signal,
    questions: {
      first: { question: "First?", choices: { yes: "Yes", no: "No" } },
      second: { question: "Second?", choices: { yes: "Yes", no: "No" } },
    },
  });
  assert.deepEqual(received, [controller.signal, controller.signal]);
  controller.abort(reason);
  await assert.rejects(pending, (error) => error === reason);
});

function questions(count) {
  return Object.fromEntries(Array.from({ length: count }, (_, i) => [
    `q${i}`, { question: `Question ${i}?`, choices: { yes: "Yes", no: "No" } },
  ]));
}

for (const concurrency of [1, 3]) {
  test(`snapshots queued questions, choices and images (concurrency: ${concurrency})`, async () => {
    const input = questions(5);
    const images = [{ mediaType: "image/png", base64: "aW1hZ2U=" }];
    const seen = [];
    let release;
    const first = new Promise((resolve) => { release = resolve; });
    const choose = createChooser(async (request) => {
      seen.push(request);
      if (seen.length === 1) await first;
      return { logprobs: [-0.1, -2] };
    });
    const pending = chooseMany(choose, {
      context: "Shared.", questions: input, images, concurrency,
    });

    input.q4.question = "Changed question?";
    input.q4.choices.yes = "Changed description";
    input.q4.choices.extra = "New choice";
    images[0].base64 = "Y2hhbmdlZA==";
    images.push({ mediaType: "image/jpeg", base64: "bmV3" });
    release();

    const result = await pending;
    assert.deepEqual(Object.keys(result.q4.distribution), ["yes", "no"]);
    assert.equal(seen.length, 5);
    const queued = seen.find(({ prompt }) => prompt.includes("Question 4?"));
    assert.ok(queued);
    assert.ok(queued.prompt.includes('"yes": "Yes"'));
    for (const request of seen) {
      assert.deepEqual(request.images, [{ mediaType: "image/png", base64: "aW1hZ2U=" }]);
    }
  });
}

test("snapshots image fields without depending on property enumeration", async () => {
  const expected = { mediaType: "image/png", base64: "aW1hZ2U=" };
  const images = [
    new class {
      get mediaType() { return expected.mediaType; }
      get base64() { return expected.base64; }
    }(),
    Object.defineProperties({}, {
      mediaType: { value: expected.mediaType },
      base64: { value: expected.base64 },
    }),
    Object.defineProperty({ ...expected }, "metadata", {
      enumerable: true,
      get() { assert.fail("Unrelated image properties must not be read."); },
    }),
  ];
  for (const image of images) {
    const seen = [];
    const choose = createChooser((request) => {
      seen.push(request.images);
      return { logprobs: [-0.1, -2] };
    });
    await chooseMany(choose, { context: "", questions: questions(2), images: [image], concurrency: 1 });
    assert.deepEqual(seen, [[expected], [expected]]);
  }
});

test("snapshotting preserves chooser validation for malformed inputs", async () => {
  let calls = 0;
  const choose = createChooser(() => { calls++; return { logprobs: [-0.1, -2] }; });
  const symbolChoices = Object.defineProperty({ yes: "Yes", no: "No" }, Symbol("hidden"), { value: "Invalid" });
  const arrayImage = Object.assign([], { mediaType: "image/png", base64: "aW1hZ2U=" });
  for (const choices of [["Yes", "No"], symbolChoices]) {
    await assert.rejects(chooseMany(choose, {
      context: "", questions: { one: { question: "One?", choices } },
    }), /choices must be a record/);
  }
  for (const images of [{ 0: arrayImage }, [arrayImage]]) {
    await assert.rejects(chooseMany(choose, {
      context: "", questions: questions(1), images,
    }), TypeError);
  }
  assert.equal(calls, 0);
});

for (const [count, concurrency] of [[1000, 3], [5, 1], [2, 3], [5, undefined]]) {
  test(`bounds ${count} questions and preserves input order (concurrency: ${concurrency ?? "default"})`, async () => {
    const input = questions(count);
    const active = new Map();
    let peak = 0;
    let calls = 0;
    const choose = ({ question }) => new Promise((resolve) => {
      calls++;
      active.set(question, () => {
        active.delete(question);
        resolve({ choice: question });
      });
      peak = Math.max(peak, active.size);
    });
    const pending = chooseMany(choose, {
      context: "", questions: input,
      ...(concurrency === undefined ? {} : { concurrency }),
    });
    const limit = Math.min(concurrency ?? count, count);
    // The initial window starts immediately, without serializing small groups.
    assert.equal(calls, limit);
    while (active.size > 0) {
      // Complete each window backwards to detect results stored in completion order.
      for (const release of [...active.values()].reverse()) release();
      await new Promise(setImmediate);
      assert.ok(active.size <= limit);
    }
    const result = await pending;
    assert.equal(peak, limit);
    assert.equal(calls, count);
    assert.deepEqual(Object.keys(result), Object.keys(input));
    for (const [name, value] of Object.entries(input)) {
      assert.equal(result[name].choice, value.question);
    }
    assert.ok(Object.isFrozen(result));
  });
}

test("rejects invalid concurrency before invoking the chooser", async () => {
  let calls = 0;
  const choose = async () => { calls++; };
  for (const concurrency of [0, -1, 1.5, NaN, Infinity, Number.MAX_SAFE_INTEGER + 1, "2", null]) {
    await assert.rejects(chooseMany(choose, {
      context: "", questions: questions(2), concurrency,
    }), /concurrency must be a positive safe integer/);
  }
  assert.equal(calls, 0);
});

for (const failure of [new Error("Backend failed"), undefined]) {
  test(`stops queued work on rejection and observes late sibling errors (${failure})`, async () => {
    const started = [];
    const rejectCalls = [];
    const choose = ({ question }) => new Promise((_, reject) => {
      started.push(question);
      rejectCalls.push(reject);
    });
    const pending = chooseMany(choose, { context: "", questions: questions(4), concurrency: 3 });
    assert.equal(started.length, 3);
    rejectCalls[1](failure);
    await assert.rejects(pending, (error) => error === failure);
    for (const reject of rejectCalls) reject(new Error("Late sibling failure"));
    await new Promise(setImmediate);
    assert.equal(started.length, 3);
  });
}

test("stops queued work even when the chooser throws synchronously", async () => {
  const failure = new Error("Synchronous failure");
  let calls = 0;
  await assert.rejects(chooseMany(() => { calls++; throw failure; }, {
    context: "", questions: questions(4), concurrency: 3,
  }), (error) => error === failure);
  assert.equal(calls, 1);
});

for (const honorsAbort of [true, false]) {
  test(`cancellation prevents queued work (chooser honors abort: ${honorsAbort})`, async () => {
    const controller = new AbortController();
    const reason = new Error("Cancelled");
    const received = [];
    const releases = [];
    const choose = ({ signal }) => new Promise((resolve, reject) => {
      received.push(signal);
      releases.push(() => resolve({ choice: "yes" }));
      if (honorsAbort) signal.addEventListener("abort", () => reject(signal.reason), { once: true });
    });
    const pending = chooseMany(choose, {
      context: "", questions: questions(4), signal: controller.signal, concurrency: 3,
    });
    assert.deepEqual(received, Array(3).fill(controller.signal));
    controller.abort(reason);
    for (const release of releases) release();
    await assert.rejects(pending, (error) => error === reason);
    await new Promise(setImmediate);
    assert.equal(received.length, 3);
  });
}

test("refills a free slot without waiting for the rest of its window", async () => {
  const releases = [];
  const pending = chooseMany(() => new Promise((resolve) => {
    releases.push(() => resolve({ choice: "yes" }));
  }), { context: "", questions: questions(3), concurrency: 2 });
  assert.equal(releases.length, 2);
  releases[1]();
  await new Promise(setImmediate);
  assert.equal(releases.length, 3);
  releases[0]();
  releases[2]();
  await pending;
});
