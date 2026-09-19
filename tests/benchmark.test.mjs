import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { spawnSync } from "node:child_process";
import { after, test } from "node:test";
import { pathToFileURL } from "node:url";

const root = resolve(import.meta.dirname, "..");
const input = join(root, "benchmarks", "data", "semif-authored144.jsonl");
const preload = pathToFileURL(join(root, "tests", "fixtures", "benchmark-fetch.mjs")).href;
const temporaryDirectories = [];

after(() => {
  for (const directory of temporaryDirectories) rmSync(directory, { recursive: true, force: true });
});

function temporaryPath(name) {
  const directory = mkdtempSync(join(tmpdir(), "choosekit-benchmark-"));
  temporaryDirectories.push(directory);
  return join(directory, "nested", name);
}

function run(script, args, env = {}) {
  return spawnSync(process.execPath, [
    "--import", preload,
    join(root, "benchmarks", script),
    ...args,
  ], {
    cwd: root,
    encoding: "utf8",
    env: { ...process.env, ...env },
  });
}

function writeJson(path, value) {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, JSON.stringify(value));
}

const qwenRows = [
  { id: "one", optionIds: ["a", "b"], gold: "a", predicted: "a", distribution: { a: 0.8, b: 0.2 } },
  { id: "two", optionIds: ["a", "b"], gold: "b", predicted: "b", distribution: { a: 0.3, b: 0.7 } },
];
const jevRows = [
  { id: "one", optionIds: ["a", "b"], gold: "a", predicted: "a", probabilities: { a: 0.7, b: 0.3 } },
  { id: "two", optionIds: ["a", "b"], gold: "b", predicted: "b", probabilities: { a: 0.4, b: 0.6 } },
];

test("llama.cpp benchmark creates its output directory before inference", () => {
  const output = temporaryPath("qwen.json");
  const result = run("run-semif.mjs", [
    "--input", input, "--output", output, "--limit", "1",
  ], { CHOOSEKIT_EXPECT_OUTPUT_PARENT: dirname(output) });

  assert.equal(result.status, 0, result.stderr);
  const report = JSON.parse(readFileSync(output, "utf8"));
  assert.equal(report.results.length, 1);
  assert.equal(report.summary.errors, 0);
});

test("OpenRouter benchmark creates its output directory before inference", () => {
  const output = temporaryPath("jev.json");
  const result = run("run-semif-openrouter-jev.mjs", [
    "--input", input, "--output", output, "--limit", "1",
  ], {
    CHOOSEKIT_EXPECT_OUTPUT_PARENT: dirname(output),
    OPENROUTER_API_KEY: "test-key-that-is-long-enough",
  });

  assert.equal(result.status, 0, result.stderr);
  const report = JSON.parse(readFileSync(output, "utf8"));
  assert.equal(report.results.length, 1);
  assert.equal(report.summary.errors, 0);
});

test("comparator creates a nested output and preserves its public result shape", () => {
  const qwen = temporaryPath("qwen.json");
  const jev = temporaryPath("jev.json");
  const output = temporaryPath("comparison.json");
  writeJson(qwen, { results: qwenRows });
  writeJson(jev, { results: jevRows });

  const result = run("compare-semif.mjs", ["--qwen", qwen, "--jev", jev, "--output", output]);

  assert.equal(result.status, 0, result.stderr);
  const comparison = JSON.parse(readFileSync(output, "utf8"));
  assert.equal(comparison.cases, 2);
  assert.equal(comparison.optionProbabilities, 4);
  assert.equal(comparison.sameTopChoice, 2);
});

const invalidComparisons = [
  ["duplicate Qwen IDs", [...qwenRows, qwenRows[0]], jevRows],
  ["duplicate Jev IDs", qwenRows, [...jevRows, jevRows[0]]],
  ["different case sets", qwenRows, [jevRows[0], { ...jevRows[1], id: "other" }]],
  ["different option order", qwenRows, [jevRows[0], { ...jevRows[1], optionIds: ["b", "a"] }]],
  ["different gold labels", qwenRows, [jevRows[0], { ...jevRows[1], gold: "a" }]],
  ["missing probability keys", qwenRows, [jevRows[0], { ...jevRows[1], probabilities: { a: 1 } }]],
  ["extra probability keys", qwenRows, [jevRows[0], { ...jevRows[1], probabilities: { a: 0.3, b: 0.6, c: 0.1 } }]],
  ["negative probabilities", [{ ...qwenRows[0], distribution: { a: 1.1, b: -0.1 } }, qwenRows[1]], jevRows],
  ["probabilities above one", qwenRows, [jevRows[0], { ...jevRows[1], probabilities: { a: -0.1, b: 1.1 } }]],
  ["unnormalized probabilities", [{ ...qwenRows[0], distribution: { a: 0.5, b: 0.2 } }, qwenRows[1]], jevRows],
];

for (const [name, qwenRowsValue, jevRowsValue] of invalidComparisons) {
  test(`comparator rejects ${name}`, () => {
    const qwen = temporaryPath("qwen.json");
    const jev = temporaryPath("jev.json");
    const output = temporaryPath("comparison.json");
    writeJson(qwen, { results: qwenRowsValue });
    writeJson(jev, { results: jevRowsValue });

    const result = run("compare-semif.mjs", ["--qwen", qwen, "--jev", jev, "--output", output]);

    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /(?:Error|TypeError)/);
    assert.throws(() => readFileSync(output));
  });
}

test("comparator refuses to overwrite an existing output", () => {
  const qwen = temporaryPath("qwen.json");
  const jev = temporaryPath("jev.json");
  const output = temporaryPath("comparison.json");
  writeJson(qwen, { results: qwenRows });
  writeJson(jev, { results: jevRows });
  writeJson(output, { sentinel: true });

  const result = run("compare-semif.mjs", ["--qwen", qwen, "--jev", jev, "--output", output]);

  assert.notEqual(result.status, 0);
  assert.deepEqual(JSON.parse(readFileSync(output, "utf8")), { sentinel: true });
});
