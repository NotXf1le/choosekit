import assert from "node:assert/strict";
import { spawn, spawnSync } from "node:child_process";
import { createServer } from "node:http";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

const packageRoot = new URL("../", import.meta.url);
const cli = fileURLToPath(new URL("../dist/cli.js", import.meta.url));

function cliEnvironment(overrides = {}) {
  return {
    ...process.env,
    CHOOSEKIT_BASE_URL: "http://127.0.0.1:1",
    CHOOSEKIT_MODE: "labels",
    ...overrides,
  };
}

function startCli(env = cliEnvironment()) {
  const child = spawn(process.execPath, [cli], {
    cwd: packageRoot,
    env,
    stdio: ["pipe", "pipe", "pipe"],
  });
  const lines = [];
  const pending = new Map();
  let buffer = "";
  child.stdout.setEncoding("utf8");
  child.stdout.on("data", (chunk) => {
    buffer += chunk;
    for (;;) {
      const newline = buffer.indexOf("\n");
      if (newline < 0) break;
      const line = buffer.slice(0, newline).trim();
      buffer = buffer.slice(newline + 1);
      if (!line) continue;
      lines.push(line);
      const message = JSON.parse(line);
      if ("id" in message) {
        const waiter = pending.get(message.id);
        if (waiter) {
          pending.delete(message.id);
          waiter.resolve(message);
        }
      }
    }
  });
  child.on("error", (error) => {
    for (const waiter of pending.values()) waiter.reject(error);
    pending.clear();
  });
  child.on("exit", (code, signal) => {
    const error = new Error(`choosekit-mcp exited before responding (${code ?? signal}).`);
    for (const waiter of pending.values()) waiter.reject(error);
    pending.clear();
  });
  let nextId = 1;
  const rpc = (method, params) => {
    const id = nextId++;
    const response = new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        pending.delete(id);
        reject(new Error(`Timed out waiting for ${method}.`));
      }, 10_000);
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
    child.stdin.write(`${JSON.stringify({
      jsonrpc: "2.0", id, method, ...(params ? { params } : {}),
    })}\n`);
    return response;
  };
  return {
    child,
    lines,
    rpc,
    notify(method, params) {
      child.stdin.write(`${JSON.stringify({ jsonrpc: "2.0", method, params })}\n`);
    },
    close() {
      child.stdin.end();
      child.kill();
    },
  };
}

async function initialize(process_) {
  const response = await process_.rpc("initialize", {
    protocolVersion: "2025-06-18",
    capabilities: {},
    clientInfo: { name: "choosekit-mcp-tests", version: "1.0.0" },
  });
  assert.equal(response.result.protocolVersion, "2025-06-18");
  process_.notify("notifications/initialized");
}

test("reports invalid environment configuration without writing to stdout", () => {
  const cases = [
    ["missing base URL", { CHOOSEKIT_BASE_URL: "" }],
    ["invalid base URL", { CHOOSEKIT_BASE_URL: "not a URL" }],
    ["invalid mode", { CHOOSEKIT_MODE: "keys" }],
  ];
  for (const [name, override] of cases) {
    const result = spawnSync(process.execPath, [cli], {
      cwd: packageRoot,
      env: cliEnvironment(override),
      encoding: "utf8",
    });
    assert.equal(result.status, 1, name);
    assert.equal(result.stdout, "", name);
    assert.match(result.stderr, /^choosekit-mcp:/, name);
  }
});

test("lists tools without contacting llama.cpp and writes only MCP messages to stdout", async (t) => {
  const process_ = startCli();
  t.after(() => process_.close());
  await initialize(process_);

  const response = await process_.rpc("tools/list");

  assert.deepEqual(response.result.tools.map((tool) => tool.name), ["choose"]);
  assert.ok(process_.lines.length >= 2);
  for (const line of process_.lines) assert.doesNotThrow(() => JSON.parse(line));
});

test("serves a minimal-prefix choice using the configured llama.cpp endpoint", async (t) => {
  const requests = [];
  const llama = createServer(async (request, response) => {
    let body = "";
    for await (const chunk of request) body += chunk;
    const parsed = JSON.parse(body);
    requests.push({ url: request.url, body: parsed });
    response.setHeader("content-type", "application/json");
    if (request.url === "/tokenize") {
      const tokens = [...new TextEncoder().encode(parsed.content)].map((byte) => byte + 1);
      response.end(JSON.stringify({ tokens }));
      return;
    }
    const target = parsed.logit_bias[0][0];
    const score = target === "b".charCodeAt(0) + 1 ? -0.2 : -1.2;
    response.end(JSON.stringify({
      tokens: [target],
      completion_probabilities: [{
        id: target,
        logprob: score,
        top_logprobs: [{ id: target, logprob: score }],
      }],
      generation_settings: { post_sampling_probs: false, backend_sampling: false },
      tokens_evaluated: parsed.prompt.length,
      tokens_predicted: 1,
      timings: { cache_n: 0 },
    }));
  });
  await new Promise((resolve) => llama.listen(0, "127.0.0.1", resolve));
  t.after(() => new Promise((resolve) => llama.close(resolve)));
  const { port } = llama.address();
  const process_ = startCli(cliEnvironment({
    CHOOSEKIT_BASE_URL: `http://127.0.0.1:${port}`,
    CHOOSEKIT_MODEL: "fixture-model",
    CHOOSEKIT_MODE: "minimal-prefix",
  }));
  t.after(() => process_.close());
  await initialize(process_);

  const response = await process_.rpc("tools/call", {
    name: "choose",
    arguments: {
      context: "A payout failed.",
      question: "Which team should handle this?",
      choices: {
        billing: "Payments, payouts, invoices, and refunds",
        technical: "Bugs, outages, integrations, and API errors",
      },
    },
  });

  assert.equal(response.result.structuredContent.choice, "billing");
  assert.deepEqual(JSON.parse(response.result.content[0].text), response.result.structuredContent);
  assert.ok(requests.some((request) => request.url === "/tokenize"));
  const scoredTokenIds = requests
    .filter((request) => request.url === "/completion")
    .map((request) => Number(request.body.logit_bias[0][0]));
  assert.ok(scoredTokenIds.includes("b".charCodeAt(0) + 1));
  assert.ok(scoredTokenIds.includes("t".charCodeAt(0) + 1));
  assert.ok(!scoredTokenIds.includes("A".charCodeAt(0) + 1));
  assert.ok(requests.some((request) => request.body.model === "fixture-model"));
  for (const line of process_.lines) assert.doesNotThrow(() => JSON.parse(line));
});
