import assert from "node:assert/strict";
import { test } from "node:test";
import { InMemoryTransport } from "@modelcontextprotocol/server";
import { createChooser, ScoringError } from "choosekit";
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

test("exposes a strict, read-only choose tool", async (t) => {
  const connection = await connect(async () => assert.fail("chooser should not run"));
  t.after(() => connection.close());

  const listed = await connection.rpc("tools/list");
  const [tool] = listed.result.tools;
  assert.equal(tool.name, "choose");
  assert.deepEqual(tool.annotations, { readOnlyHint: true, openWorldHint: false });
  assert.equal(tool.inputSchema.additionalProperties, false);
  assert.equal(tool.outputSchema.additionalProperties, false);

  const invalid = await callChoose(connection, { ...request, endpoint: "https://example.com" });
  assert.equal(invalid.result.isError, true);
  assert.match(invalid.result.content[0].text, /endpoint|unrecognized|additional/i);
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

test("does not expose scorer or implementation errors", async (t) => {
  const cases = [
    [async () => { throw new ScoringError("secret upstream response"); },
      "llama.cpp could not score the supplied choices."],
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
  const arguments_ = JSON.parse(JSON.stringify({
    context: "Choose safely.",
    question: "Which key?",
    choices: JSON.parse('{"__proto__":"Prototype option","first":"First option","second":"Second option"}'),
  }));

  const response = await callChoose(connection, arguments_);

  assert.equal(response.result.isError, true);
  assert.equal(called, false);
  assert.equal(Object.getPrototypeOf({}).polluted, undefined);
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
