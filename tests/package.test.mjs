import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { test } from "node:test";
import { request } from "./helpers.mjs";

test("ESM and CommonJS public exports implement the same API", async () => {
  const esm = await import("choosekit");
  const cjs = createRequire(import.meta.url)("choosekit");
  assert.deepEqual(Object.keys(esm).sort(), Object.keys(cjs).sort());
  const a = await esm.createChooser(() => ({ logprobs: [-3, -1, -2] }))(request);
  const b = await cjs.createChooser(() => ({ logprobs: [-3, -1, -2] }))(request);
  assert.deepEqual(a, b);
  assert.equal(typeof (await import("choosekit/llama-cpp")).fromLlamaCpp, "function");
  assert.equal(typeof createRequire(import.meta.url)("choosekit/llama-cpp").fromLlamaCpp, "function");
  assert.equal(typeof (await import("choosekit/ollama")).fromOllama, "function");
  assert.equal(typeof createRequire(import.meta.url)("choosekit/ollama").fromOllama, "function");
  assert.equal(typeof (await import("choosekit/openrouter")).fromOpenRouter, "function");
  assert.equal(typeof createRequire(import.meta.url)("choosekit/openrouter").fromOpenRouter, "function");
});

test("has no runtime dependencies, install hooks or executable", () => {
  const pkg = JSON.parse(readFileSync(new URL("../package.json", import.meta.url)));
  for (const key of ["dependencies", "peerDependencies", "optionalDependencies", "bin"]) {
    assert.equal(pkg[key], undefined);
  }
  for (const key of ["preinstall", "install", "postinstall", "prepare"]) {
    assert.equal(pkg.scripts[key], undefined);
  }
  assert.equal(pkg.sideEffects, false);
});

test("importing public entrypoints does not call fetch or log anything", () => {
  const code = `
    globalThis.fetch = () => { throw new Error("Unexpected network call"); };
    await import("choosekit");
    await import("choosekit/llama-cpp");
    await import("choosekit/ollama");
    await import("choosekit/openrouter");
  `;
  const result = spawnSync(process.execPath, ["--input-type=module", "-e", code],
    { cwd: new URL("../", import.meta.url), encoding: "utf8" });
  assert.equal(result.status, 0, result.stderr);
  assert.equal(result.stdout, "");
  assert.equal(result.stderr, "");
});
