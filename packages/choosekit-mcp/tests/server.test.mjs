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
