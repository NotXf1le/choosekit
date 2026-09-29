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
