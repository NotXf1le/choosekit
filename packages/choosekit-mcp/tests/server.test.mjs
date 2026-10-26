import assert from "node:assert/strict";
import { test } from "node:test";
import { InMemoryTransport } from "@modelcontextprotocol/server";
import { createChooser, ScoringError } from "choosekit";
import { ImageLoadError } from "../dist/images.js";
import { buildServer } from "../dist/server.js";

const request = Object.freeze({
  context: "A payout failed.",
  question: "Which team should handle this?",
  choices: Object.freeze({
    billing: "Payments, payouts, invoices, and refunds",
    technical: "Bugs, outages, integrations, and API errors",
  }),
});

async function connect(chooser, options) {
  const server = buildServer(chooser, options);
  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
  const pending = new Map();
  clientTransport.onmessage = (message) => {
    if (!("id" in message)) return;
    const waiter = pending.get(message.id);
    if (waiter) {
      pending.delete(message.id);
      waiter.resolve(message);
    }
  };
  clientTransport.onerror = (error) => {
    for (const waiter of pending.values()) waiter.reject(error);
    pending.clear();
  };
  await server.connect(serverTransport);
  await clientTransport.start();
  let nextId = 1;
  const startRpc = async (method, params) => {
    const id = nextId++;
    const response = new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        pending.delete(id);
        reject(new Error(`Timed out waiting for ${method}.`));
      }, 5_000);
      pending.set(id, {
        resolve: (value) => {
          clearTimeout(timeout);
          resolve(value);
        },
        reject: (error) => {
          clearTimeout(timeout);
          reject(error);
        },
      });
    });
    await clientTransport.send({ jsonrpc: "2.0", id, method, ...(params ? { params } : {}) });
    return { id, response };
  };
  const rpc = async (method, params) => {
    const call = await startRpc(method, params);
    return call.response;
  };
  const initialized = await rpc("initialize", {
    protocolVersion: "2025-06-18",
    capabilities: {},
    clientInfo: { name: "choosekit-mcp-tests", version: "1.0.0" },
  });
  assert.equal(initialized.result.protocolVersion, "2025-06-18");
  await clientTransport.send({ jsonrpc: "2.0", method: "notifications/initialized" });
  return {
    rpc,
    startRpc,
    discard(id) {
      const waiter = pending.get(id);
      if (!waiter) return;
      pending.delete(id);
      waiter.reject(new Error("Request cancelled by the test client."));
    },
    notify: (method, params) => clientTransport.send({ jsonrpc: "2.0", method, params }),
    close: async () => {
      await clientTransport.close();
      await server.close();
    },
  };
}

async function callChoose(connection, arguments_) {
  return connection.rpc("tools/call", { name: "choose", arguments: arguments_ });
}

const manyRequest = {
  context: request.context,
  questions: {
    route: { question: request.question, choices: request.choices },
    constructor: { question: "How urgent is this?", choices: { low: "Can wait", high: "Needs immediate attention" } },
  },
};

test("chooseMany starts independent questions together and loads shared images once", async (t) => {
  const calls = [];
  const images = [{ mediaType: "image/png", base64: "aW1hZ2U=" }];
  let loads = 0;
  let release;
  const gate = new Promise((resolve) => { release = resolve; });
  const chooser = async (input) => {
    calls.push(input);
    if (calls.length === 2) release();
    await gate;
    const keys = Object.keys(input.choices);
    return {
      choice: keys[0],
      distribution: Object.fromEntries(keys.map((key, i) => [key, i === 0 ? 1 : 0])),
      scores: Object.fromEntries(keys.map((key, i) => [key, i === 0 ? -0.1 : -Infinity])),
      margin: 1,
      entropy: 0,
      boundaryTokens: 0,
      usage: { promptTokens: input.question.length, cachedTokens: null, completionTokens: 1, requests: 1 },
    };
  };
  const connection = await connect(chooser, {
    backend: "ollama",
    imageLoader: async (paths) => {
      loads++;
      assert.deepEqual(paths, ["example.png"]);
      return images;
    },
  });
  t.after(() => connection.close());
  const response = await connection.rpc("tools/call", {
    name: "chooseMany", arguments: { ...manyRequest, imagePaths: ["example.png"] },
  });
  assert.notEqual(response.result.isError, true);
  const result = response.result.structuredContent;
  assert.deepEqual(Object.keys(result), ["route", "constructor"]);
  assert.equal(result.route.choice, "billing");
  assert.equal(result.constructor.choice, "low");
  assert.equal(result.route.scores.technical, null);
  assert.equal(result.route.usage.promptTokens, request.question.length);
  assert.equal(result.constructor.usage.promptTokens, manyRequest.questions.constructor.question.length);
  assert.deepEqual(JSON.parse(response.result.content[0].text), result);
  assert.equal(loads, 1);
  for (const call of calls) {
    assert.equal(call.context, manyRequest.context);
    assert.deepEqual(call.images, images);
    assert.equal(call.signal, calls[0].signal);
  }
});

test("chooseMany respects the requested concurrency", async (t) => {
  for (const concurrency of [1, 2]) {
    await t.test(`concurrency ${concurrency}`, async (t) => {
      let active = 0;
      let peak = 0;
      let calls = 0;
      const chooser = createChooser(async () => {
        calls++;
        peak = Math.max(peak, ++active);
        await new Promise(setImmediate);
        active--;
        return { logprobs: [-0.1, -2], boundaryTokens: 0 };
      });
      const connection = await connect(chooser);
      t.after(() => connection.close());
      const questions = { ...manyRequest.questions, extra: manyRequest.questions.route };
      const response = await connection.rpc("tools/call", {
        name: "chooseMany", arguments: { ...manyRequest, questions, concurrency },
      });
      assert.notEqual(response.result.isError, true);
      assert.deepEqual(Object.keys(response.result.structuredContent), Object.keys(questions));
      assert.equal(calls, 3);
      assert.equal(peak, concurrency);
    });
  }
});

test("chooseMany rejects invalid groups before calling the chooser", async (t) => {
  let calls = 0;
  const connection = await connect(async () => { calls++; }, { backend: "openrouter" });
  t.after(() => connection.close());
  const tooMany = Object.fromEntries(Array.from({ length: 21 }, (_, i) => [`choice${i}`, `Description ${i}`]));
  for (const arguments_ of [
    { ...manyRequest, questions: {} },
    { ...manyRequest, model: "test-model" },
    ...[0, -1, 1.5, "2", null, Number.MAX_SAFE_INTEGER + 1]
      .map((concurrency) => ({ ...manyRequest, concurrency })),
    { ...manyRequest, questions: { route: { ...manyRequest.questions.route, question: " " } } },
    { ...manyRequest, questions: { route: { ...manyRequest.questions.route, model: "test-model" } } },
    { ...manyRequest, questions: { route: { question: request.question, choices: tooMany } } },
    { ...manyRequest, questions: JSON.parse('{"__proto__":{"question":"Which?","choices":{"a":"First","b":"Second"}}}') },
  ]) {
    const response = await connection.rpc("tools/call", { name: "chooseMany", arguments: arguments_ });
    assert.equal(response.result.isError, true);
    assert.equal(calls, 0);
  }
});

test("chooseMany returns a sanitized group error when a question fails", async (t) => {
  const connection = await connect(async () => { throw new ScoringError("secret upstream details"); });
  t.after(() => connection.close());
  const response = await connection.rpc("tools/call", { name: "chooseMany", arguments: manyRequest });
  assert.equal(response.result.isError, true);
  assert.equal(response.result.structuredContent, undefined);
  assert.equal(response.result.content[0].text, "The model could not score the supplied choices.");
});

test("chooseMany passes MCP cancellation to every question", async (t) => {
  const signals = [];
  let started;
  const hasStarted = new Promise((resolve) => { started = resolve; });
  const connection = await connect(({ signal }) => new Promise((resolve, reject) => {
    signals.push(signal);
    if (signals.length === 2) started();
    signal.addEventListener("abort", () => reject(signal.reason), { once: true });
  }));
  t.after(() => connection.close());
  const call = await connection.startRpc("tools/call", { name: "chooseMany", arguments: manyRequest });
  await hasStarted;
  await connection.notify("notifications/cancelled", { requestId: call.id, reason: "test" });
  assert.ok(signals.every((signal) => signal.aborted));
  connection.discard(call.id);
  await assert.rejects(call.response, /cancelled by the test client/);
});

test("exposes a strict, read-only llama.cpp choose tool", async (t) => {
  const connection = await connect(async () => assert.fail("chooser should not run"));
  t.after(() => connection.close());

  const listed = await connection.rpc("tools/list");
  const [tool] = listed.result.tools;
  assert.equal(tool.name, "choose");
  assert.deepEqual(tool.annotations, { readOnlyHint: true, openWorldHint: false });
  assert.equal(tool.inputSchema.additionalProperties, false);
  assert.equal(tool.outputSchema.additionalProperties, false);

  const invalid = await callChoose(connection, { ...request, backend: "not allowed" });
  assert.equal(invalid.result.isError, true);
  assert.match(invalid.result.content[0].text, /backend|unrecognized|additional/i);
});

test("marks the OpenRouter tool as open-world", async (t) => {
  const connection = await connect(async () => assert.fail("chooser should not run"), {
    backend: "openrouter",
  });
  t.after(() => connection.close());

  const listed = await connection.rpc("tools/list");
  assert.deepEqual(listed.result.tools[0].annotations, {
    readOnlyHint: true,
    openWorldHint: true,
  });
});

test("exposes Ollama as a closed-world labels backend with at most 20 choices", async (t) => {
  const connection = await connect(async () => assert.fail("chooser should not run"), {
    backend: "ollama",
  });
  t.after(() => connection.close());

  const listed = await connection.rpc("tools/list");
  const [tool] = listed.result.tools;
  assert.deepEqual(tool.annotations, { readOnlyHint: true, openWorldHint: false });
  assert.equal(tool.inputSchema.properties.choices.maxProperties, 20);
});

test("passes loaded images to the chooser in labels mode", async (t) => {
  const image = Object.freeze({ mediaType: "image/png", base64: "aW1hZ2U=" });
  const seenPaths = [];
  let receivedImages;
  const imageLoader = async (paths) => {
    seenPaths.push(...paths);
    return [image];
  };
  const chooser = async ({ choices, images }) => {
    receivedImages = images;
    return {
      choice: "billing",
      distribution: { billing: 1, technical: 0 },
      scores: { billing: -0.1, technical: -2 },
      margin: 1,
      entropy: 0,
      boundaryTokens: 1,
    };
  };
  const connection = await connect(chooser, { backend: "ollama", imageLoader });
  t.after(() => connection.close());

  const listed = await connection.rpc("tools/list");
  assert.ok("imagePaths" in listed.result.tools[0].inputSchema.properties);
  const response = await callChoose(connection, { ...request, imagePaths: ["screen.png"] });

  assert.notEqual(response.result.isError, true);
  assert.deepEqual(seenPaths, ["screen.png"]);
  assert.deepEqual(receivedImages, [image]);
});

test("reports image loading failures without exposing local paths", async (t) => {
  const secretPath = "C:\\Users\\example\\private.png";
  const connection = await connect(async () => assert.fail("chooser should not run"), {
    imageLoader: async () => {
      throw new ImageLoadError(`could not read ${secretPath}`);
    },
  });
  t.after(() => connection.close());

  const response = await callChoose(connection, { ...request, imagePaths: ["missing.png"] });

  assert.equal(response.result.isError, true);
  assert.equal(response.result.content[0].text,
    "The image could not be loaded. Check imagePaths and CHOOSEKIT_IMAGE_ROOT.");
  assert.doesNotMatch(JSON.stringify(response), /private\.png/);
});

test("does not expose imagePaths without an image loader or in minimal-prefix mode", async (t) => {
  for (const options of [{}, { mode: "minimal-prefix", imageLoader: async () => [] }]) {
    await t.test(JSON.stringify(options), async (t) => {
      const connection = await connect(async () => assert.fail("chooser should not run"), options);
      t.after(() => connection.close());
      const listed = await connection.rpc("tools/list");
      assert.ok(!("imagePaths" in listed.result.tools[0].inputSchema.properties));
      const response = await callChoose(connection, { ...request, imagePaths: ["screen.png"] });
      assert.equal(response.result.isError, true);
    });
  }
});

test("OpenRouter accepts 20 choices and preserves the selected choice key", async (t) => {
  const choices = Object.fromEntries(Array.from({ length: 20 }, (_, index) =>
    [`choice_${index}`, `Choice ${index}`]));
  let receivedChoices;
  const chooser = async ({ choices: supplied }) => {
    receivedChoices = supplied;
    return {
      choice: "choice_19",
      distribution: Object.fromEntries(Object.keys(supplied).map((key) =>
        [key, key === "choice_19" ? 1 : 0])),
      scores: Object.fromEntries(Object.keys(supplied).map((key) => [key, -1])),
      margin: 1,
      entropy: 0,
      boundaryTokens: 1,
    };
  };
  const connection = await connect(chooser, { backend: "openrouter" });
  t.after(() => connection.close());

  const response = await callChoose(connection, { ...request, choices });

  assert.equal(response.result.structuredContent.choice, "choice_19");
  assert.deepEqual(receivedChoices, choices);
});

test("OpenRouter rejects 21 choices before calling the chooser", async (t) => {
  let called = false;
  const connection = await connect(async () => { called = true; }, { backend: "openrouter" });
  t.after(() => connection.close());
  const choices = Object.fromEntries(Array.from({ length: 21 }, (_, index) =>
    [`choice_${index}`, `Choice ${index}`]));

  const response = await callChoose(connection, { ...request, choices });

  assert.equal(response.result.isError, true);
  assert.equal(called, false);
});

test("returns the same decision as the direct chooser in text and structured content", async (t) => {
  const chooser = createChooser(() => ({ logprobs: [-2, -0.5], boundaryTokens: 1 }));
  const expected = await chooser(request);
  const connection = await connect(chooser);
  t.after(() => connection.close());

  const response = await callChoose(connection, request);
  assert.deepEqual(response.result.structuredContent, expected);
  assert.deepEqual(JSON.parse(response.result.content[0].text), expected);
});

test("serializes unavailable scores as null", async (t) => {
  const chooser = async () => ({
    choice: "billing",
    distribution: { billing: 1, technical: 0 },
    scores: { billing: -0.2, technical: -Infinity },
    margin: 1,
    entropy: 0,
    boundaryTokens: 1,
  });
  const connection = await connect(chooser, { backend: "openrouter" });
  t.after(() => connection.close());

  const response = await callChoose(connection, request);
  const expected = {
    choice: "billing",
    distribution: { billing: 1, technical: 0 },
    scores: { billing: -0.2, technical: null },
    margin: 1,
    entropy: 0,
    boundaryTokens: 1,
  };

  assert.notEqual(response.result.isError, true);
  assert.deepEqual(response.result.structuredContent, expected);
  assert.deepEqual(JSON.parse(response.result.content[0].text), expected);
});

test("does not expose scorer or implementation errors", async (t) => {
  const cases = [
    [async () => { throw new ScoringError("secret upstream response"); },
      "The model could not score the supplied choices."],
    [async () => { throw new Error("secret implementation detail"); },
      "The choice request failed unexpectedly."],
  ];

  for (const [chooser, expected] of cases) {
    await t.test(expected, async (t) => {
      const connection = await connect(chooser);
      t.after(() => connection.close());
      const response = await callChoose(connection, request);
      assert.equal(response.result.isError, true);
      assert.equal(response.result.content[0].text, expected);
      assert.doesNotMatch(JSON.stringify(response), /secret/);
    });
  }
});

test("passes MCP cancellation to the chooser", async (t) => {
  let receivedSignal;
  let started;
  const hasStarted = new Promise((resolve) => { started = resolve; });
  const chooser = ({ signal }) => new Promise((resolve, reject) => {
    receivedSignal = signal;
    started();
    signal.addEventListener("abort", () => reject(signal.reason), { once: true });
  });
  const connection = await connect(chooser);
  t.after(() => connection.close());

  const call = await connection.startRpc("tools/call", { name: "choose", arguments: request });
  await hasStarted;
  await connection.notify("notifications/cancelled", { requestId: call.id, reason: "test" });

  assert.equal(receivedSignal.aborted, true);
  connection.discard(call.id);
  await assert.rejects(call.response, /cancelled by the test client/);
});

test("rejects __proto__ instead of silently dropping it", async (t) => {
  let called = false;
  const chooser = async () => {
    called = true;
    assert.fail("chooser should not run");
  };
  const connection = await connect(chooser);
  t.after(() => connection.close());
  const arguments_ = {
    context: "Choose safely.",
    question: "Which key?",
    choices: JSON.parse('{"__proto__":"Prototype option","first":"First option","second":"Second option"}'),
  };

  const response = await callChoose(connection, arguments_);

  assert.equal(response.result.isError, true);
  assert.equal(called, false);
});

test("preserves constructor as an ordinary choice key", async (t) => {
  let receivedChoices;
  const chooser = async ({ choices }) => {
    receivedChoices = choices;
    return {
      choice: "constructor",
      distribution: { constructor: 0.6, other: 0.4 },
      scores: { constructor: -0.2, other: -0.6 },
      margin: 0.2,
      entropy: 0.67,
      boundaryTokens: 1,
    };
  };
  const connection = await connect(chooser);
  t.after(() => connection.close());
  const arguments_ = {
    context: "Choose safely.",
    question: "Which key?",
    choices: JSON.parse('{"constructor":"Constructor option","other":"Other option"}'),
  };

  const response = await callChoose(connection, arguments_);

  assert.equal(response.result.structuredContent.choice, "constructor");
  assert.deepEqual(receivedChoices, arguments_.choices);
});

for (const [tool, arguments_] of [["choose", request], ["chooseMany", manyRequest]]) {
  test(`${tool} reports only validated HTTP metadata and recovers after failure`, async (t) => {
    const choose = createChooser(() => ({ logprobs: [-0.1, -2], boundaryTokens: 1 }));
    let failure;
    const connection = await connect(async (input) => {
      if (failure) throw failure;
      return choose(input);
    });
    t.after(() => connection.close());

    const cases = [
      ...[100, 200, 302, 401, 403, 429, 503, 599].map((httpStatus) => [
        new ScoringError("secret upstream details", {
          httpStatus, cause: new Error("secret cause"),
        }),
        `The scoring backend returned HTTP ${httpStatus}.`,
      ]),
      ...[undefined, null, "401", 99, 600, 401.5, NaN, Infinity, -Infinity,
        { toString() { throw new Error("must not coerce"); } }].map((httpStatus) => [
        new ScoringError("secret upstream details", { httpStatus }),
        "The model could not score the supplied choices.",
      ]),
      [Object.assign(new Error("secret implementation detail"), { httpStatus: 401 }),
        "The choice request failed unexpectedly."],
    ];

    for (const [error, text] of cases) {
      failure = error;
      const response = await connection.rpc("tools/call", { name: tool, arguments: arguments_ });
      assert.equal(response.error, undefined);
      assert.equal(response.result.isError, true);
      assert.deepEqual(response.result.content, [{ type: "text", text }]);
      assert.equal(response.result.structuredContent, undefined);
      assert.doesNotMatch(JSON.stringify(response), /secret/);
    }

    failure = undefined;
    const response = await connection.rpc("tools/call", { name: tool, arguments: arguments_ });
    assert.equal(response.error, undefined);
    assert.notEqual(response.result.isError, true);
  });
}
