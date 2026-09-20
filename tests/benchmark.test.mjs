import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { spawnSync } from "node:child_process";
import { after, test } from "node:test";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
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
  assert.equal(report.summary.errors, 1);
  assert.match(report.results[0].error, /Intentional benchmark test failure/);
  assert.equal(report.runtime.modelChecked, true);
  assert.equal(report.runtime.resolvedModel, "qwen3.8-27b-text-64k");
});

test("llama.cpp benchmark refuses a model the server does not serve", () => {
  const output = temporaryPath("qwen.json");
  const result = run("run-semif.mjs", [
    "--input", input, "--output", output, "--limit", "1",
  ], {
    CHOOSEKIT_EXPECT_OUTPUT_PARENT: dirname(output),
    CHOOSEKIT_SERVED_MODELS: "/gguf/LFM2.5-1.2B-Instruct-Q8_0.gguf",
  });

  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /serves "\/gguf\/LFM2\.5-1\.2B-Instruct-Q8_0\.gguf", not "qwen3\.8-27b-text-64k"/);
  assert.throws(() => readFileSync(output));
});

test("llama.cpp benchmark explains an unreachable model list", () => {
  const output = temporaryPath("qwen.json");
  const result = run("run-semif.mjs", [
    "--input", input, "--output", output, "--limit", "1",
  ], { CHOOSEKIT_EXPECT_OUTPUT_PARENT: dirname(output), CHOOSEKIT_MODELS_STATUS: "404" });

  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /Could not list the models at .*\/v1\/models: HTTP 404/);
  assert.match(result.stderr, /--skip-model-check/);
});

test("llama.cpp benchmark records an unchecked run as unchecked", () => {
  const output = temporaryPath("qwen.json");
  const result = run("run-semif.mjs", [
    "--input", input, "--output", output, "--limit", "1", "--skip-model-check",
  ], {
    CHOOSEKIT_EXPECT_OUTPUT_PARENT: dirname(output),
    CHOOSEKIT_SERVED_MODELS: "/gguf/LFM2.5-1.2B-Instruct-Q8_0.gguf",
  });

  assert.equal(result.status, 0, result.stderr);
  const report = JSON.parse(readFileSync(output, "utf8"));
  assert.equal(report.runtime.modelChecked, false);
  assert.equal(report.runtime.resolvedModel, null);
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
  assert.equal(report.summary.errors, 1);
  assert.match(report.results[0].error, /Intentional benchmark test failure/);
});

test("comparator creates a nested output for valid reports", () => {
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
  ["duplicate Qwen IDs", [...qwenRows, qwenRows[0]], jevRows, /duplicate case ID/],
  ["duplicate Jev IDs", qwenRows, [...jevRows, jevRows[0]], /duplicate case ID/],
  ["different case sets", qwenRows, [jevRows[0], { ...jevRows[1], id: "other" }], /different case sets/],
  ["different option order", qwenRows, [jevRows[0], { ...jevRows[1], optionIds: ["b", "a"] }], /different option IDs/],
  ["different gold labels", qwenRows, [jevRows[0], { ...jevRows[1], gold: "a" }], /different gold labels/],
  ["mismatched probability keys", qwenRows, [jevRows[0], { ...jevRows[1], probabilities: { a: 0.9, c: 0.1 } }], /probability keys do not match/],
  ["negative probabilities", [{ ...qwenRows[0], distribution: { a: -0.1, b: 1 } }, qwenRows[1]], jevRows, /invalid probability for a/],
  ["probabilities above one", qwenRows, [jevRows[0], { ...jevRows[1], probabilities: { a: 0, b: 1.1 } }], /invalid probability for b/],
  ["unnormalized probabilities", [{ ...qwenRows[0], distribution: { a: 0.5, b: 0.2 } }, qwenRows[1]], jevRows, /probabilities sum to/],
  ["predicted labels outside the options", qwenRows, [jevRows[0], { ...jevRows[1], predicted: "c" }], /predicted label outside its options/],
  ["predicted labels below the maximum", qwenRows, [jevRows[0], { ...jevRows[1], predicted: "a" }], /predicted label is not a top choice/],
  ["rows with an empty error", [{ ...qwenRows[0], error: "" }, qwenRows[1]], jevRows, /row one failed:/],
];

for (const [name, qwenRowsValue, jevRowsValue, expectedError] of invalidComparisons) {
  test(`comparator rejects ${name}`, () => {
    const qwen = temporaryPath("qwen.json");
    const jev = temporaryPath("jev.json");
    const output = temporaryPath("comparison.json");
    writeJson(qwen, { results: qwenRowsValue });
    writeJson(jev, { results: jevRowsValue });

    const result = run("compare-semif.mjs", ["--qwen", qwen, "--jev", jev, "--output", output]);

    assert.notEqual(result.status, 0);
    assert.match(result.stderr, expectedError);
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
