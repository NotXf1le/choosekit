import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { performance } from "node:perf_hooks";
import { setTimeout as delay } from "node:timers/promises";
import { fromLlamaCpp } from "../dist/esm/llama-cpp.js";
import { fromOpenRouter } from "../dist/esm/openrouter.js";
import {
  SUPERGPQA_REVISION,
  SUPERGPQA_COMPATIBILITY_SAMPLE_SEED,
  SUPERGPQA_EVALUATION_SAMPLE_SEED,
  SUPERGPQA_PREPARED_SHA256,
  SUPERGPQA_ROWS,
  sampleSuperGpqaCompatibilityRows,
  sampleSuperGpqaEvaluationRows,
} from "./prepare-supergpqa.mjs";

const MAX_429_ATTEMPTS = 4;

function option(name, fallback) {
  const index = process.argv.indexOf(name);
  if (index === -1) return fallback;
  const value = process.argv[index + 1];
  if (value === undefined || value.startsWith("--")) throw new TypeError(`${name} requires a value.`);
  return value;
}

function flag(name) {
  return process.argv.includes(name);
}

function requireText(value, name) {
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new TypeError(`${name} must be a non-empty string.`);
  }
  return value;
}

function readCost(value) {
  const cost = value?.cost;
  return typeof cost === "number" && Number.isFinite(cost) && cost >= 0 ? cost : null;
}

function validateRow(row, index) {
  const optionIds = row?.choices !== null && typeof row?.choices === "object" && !Array.isArray(row.choices)
    ? Object.keys(row.choices) : [];
  if (row === null || typeof row !== "object" || Array.isArray(row)
    || typeof row.id !== "string" || row.id.length === 0 || row.context !== ""
    || typeof row.question !== "string" || row.question.length === 0
    || optionIds.length < 2 || optionIds.length > 20
    || typeof row.gold !== "string" || !optionIds.includes(row.gold)
    || typeof row.metadata?.discipline !== "string" || typeof row.metadata?.difficulty !== "string") {
    throw new TypeError(`SuperGPQA row ${row?.id ?? index + 1} is invalid.`);
  }
}

function summarize(results, elapsedSeconds) {
  const successful = results.filter((row) => row.error === undefined);
  const correct = successful.filter((row) => row.correct).length;
  const costComplete = results.every((row) => typeof row.costUsd === "number");
  return {
    rowsAttempted: results.length,
    rowsSuccessful: successful.length,
    errors: results.length - successful.length,
    endToEndAccuracy: results.length === 0 ? null : correct / results.length,
    wallSeconds: elapsedSeconds,
    decisionsPerSecond: elapsedSeconds === 0 ? null : results.length / elapsedSeconds,
    costUsd: costComplete ? results.reduce((total, row) => total + row.costUsd, 0) : null,
    costComplete,
    retries: results.reduce((total, row) => total + (row.retryCount ?? 0), 0),
  };
}

const backend = option("--backend", undefined);
if (backend !== "choosekit" && backend !== "llama-cpp" && backend !== "jev") {
  throw new TypeError("--backend must be choosekit, llama-cpp, or jev.");
}
const inputPath = option("--input", "benchmarks/.data/supergpqa/supergpqa.jsonl");
const sampleSizeArgument = option("--sample-size", "100");
const sampleSize = Number(sampleSizeArgument);
const sampleMethod = option("--sample-method", "balanced");
if (sampleMethod !== "balanced" && sampleMethod !== "proportional") {
  throw new TypeError("--sample-method must be balanced or proportional.");
}
const model = backend === "jev"
  ? option("--model", "typesafe/jev-1.13")
  : requireText(option("--model", process.env.OPENROUTER_MODEL), "--model");
const provider = backend === "choosekit"
  ? requireText(option("--provider", process.env.OPENROUTER_PROVIDER), "--provider")
  : undefined;
const baseURL = backend === "llama-cpp"
  ? requireText(option("--base-url", process.env.LLAMA_CPP_BASE_URL), "--base-url")
  : undefined;
const recordedProvider = provider ?? (backend === "llama-cpp" ? "local" : "openrouter");
const modelSlug = model.replace(/[^a-zA-Z0-9._-]+/g, "-");
const providerSlug = recordedProvider.replace(/[^a-zA-Z0-9._-]+/g, "-");
const outputPath = option("--output",
  `benchmarks/results/supergpqa-${backend}-${providerSlug}-${modelSlug}-${sampleMethod}-sample${sampleSize}.json`);
const resume = flag("--resume");
const apiKey = process.env.OPENROUTER_API_KEY;

if (backend !== "llama-cpp" && (typeof apiKey !== "string" || apiKey.length < 20)) {
  throw new Error("OPENROUTER_API_KEY is missing or invalid.");
}
if (existsSync(outputPath) && !resume) {
  throw new Error(`Refusing to overwrite existing output: ${outputPath}`);
}
if (!existsSync(outputPath) && resume) {
  throw new Error(`Cannot resume missing output: ${outputPath}`);
}
if (!Number.isSafeInteger(sampleSize) || sampleSize < 1) {
  throw new TypeError("--sample-size must be a positive integer.");
}

const preparedDataset = readFileSync(inputPath);
const preparedSha256 = createHash("sha256").update(preparedDataset).digest("hex");
if (preparedSha256 !== SUPERGPQA_PREPARED_SHA256) {
  throw new Error(`Unexpected prepared SuperGPQA SHA-256: ${preparedSha256}. Expected ${SUPERGPQA_PREPARED_SHA256}.`);
}
const preparedText = preparedDataset.toString("utf8").trim();
const preparedRows = preparedText.length === 0
  ? []
  : preparedText.split(/\r?\n/).map((line) => JSON.parse(line));
if (preparedRows.length !== SUPERGPQA_ROWS) {
  throw new Error(`Expected ${SUPERGPQA_ROWS} SuperGPQA rows, received ${preparedRows.length}.`);
}
const rowIds = new Set();
for (const [index, row] of preparedRows.entries()) {
  validateRow(row, index);
  if (rowIds.has(row.id)) throw new Error(`SuperGPQA contains duplicate row ID ${row.id}.`);
  rowIds.add(row.id);
}
const selectedRows = sampleMethod === "proportional"
  ? sampleSuperGpqaEvaluationRows(preparedRows, sampleSize)
  : sampleSuperGpqaCompatibilityRows(preparedRows, sampleSize);
mkdirSync(dirname(outputPath), { recursive: true });

const packageVersion = JSON.parse(
  readFileSync(new URL("../package.json", import.meta.url), "utf8"),
).version;
const sampleMetadata = sampleMethod === "proportional"
  ? {
      method: "proportional-discipline-difficulty",
      seed: SUPERGPQA_EVALUATION_SAMPLE_SEED,
      excludedCompatibilitySampleSize: 100,
    }
  : {
      method: "balanced-discipline-difficulty",
      seed: SUPERGPQA_COMPATIBILITY_SAMPLE_SEED,
    };
const benchmarkName = sampleMethod === "proportional"
  ? "SuperGPQA proportional discipline/difficulty sample"
  : "SuperGPQA balanced discipline/difficulty sample";
const datasetMetadata = {
  source: "https://huggingface.co/datasets/m-a-p/SuperGPQA",
  revision: SUPERGPQA_REVISION,
  sha256: preparedSha256,
  totalRows: preparedRows.length,
  selectedRows: selectedRows.length,
  sample: sampleMetadata,
};
const runtimeMetadata = {
  backend,
  model,
  provider: recordedProvider,
  ...(baseURL === undefined ? {} : { baseURL }),
  packageVersion,
};

let responseMetadata = { costUsd: null, retryCount: 0 };
const fetchWithMetadata = async (...args) => {
  let accumulatedCost = 0;
  let sawCost = false;
  for (let attempt = 1; attempt <= MAX_429_ATTEMPTS; attempt++) {
    const response = await globalThis.fetch(...args);
    const body = await response.text();
    let value;
    try { value = JSON.parse(body); } catch { value = undefined; }
    const attemptCost = readCost(value?.usage);
    if (attemptCost !== null) {
      accumulatedCost += attemptCost;
      sawCost = true;
    }
    responseMetadata = {
      costUsd: sawCost ? accumulatedCost : null,
      retryCount: attempt - 1,
      ...(typeof value?.model === "string" ? { resolvedModel: value.model } : {}),
      ...(typeof value?.provider === "string" ? { resolvedProvider: value.provider } : {}),
    };
    if (response.status === 429 && attempt < MAX_429_ATTEMPTS) {
      const retryAfter = Number(response.headers.get("retry-after"));
      await delay(Number.isFinite(retryAfter) ? retryAfter * 1000 : 1000, undefined,
        { signal: args[1]?.signal });
      continue;
    }
    return new Response(body, { status: response.status, statusText: response.statusText, headers: response.headers });
  }
  throw new Error("OpenRouter 429 retry limit reached.");
};

const choose = backend === "choosekit"
  ? fromOpenRouter({ apiKey, model, provider, fetch: fetchWithMetadata })
  : backend === "llama-cpp"
    ? fromLlamaCpp({ baseURL, model, mode: "labels" })
    : undefined;

async function decide(row) {
  if (choose !== undefined) {
    const decision = await choose({
      context: row.context,
      question: row.question,
      choices: row.choices,
      signal: AbortSignal.timeout(120_000),
    });
    return {
      predicted: decision.choice,
      scores: decision.scores,
      ...(backend === "llama-cpp" ? { costUsd: 0, retryCount: 0 } : responseMetadata),
    };
  }
  let response;
  let body;
  let accumulatedCost = 0;
  let sawCost = false;
  for (let attempt = 1; attempt <= MAX_429_ATTEMPTS; attempt++) {
    response = await fetch("https://openrouter.ai/api/alpha/decisions", {
      method: "POST",
      headers: {
        authorization: `Bearer ${apiKey}`,
        "content-type": "application/json",
        "x-openrouter-title": "choosekit SuperGPQA benchmark",
      },
      body: JSON.stringify({
        model,
        state: row.question,
        questions: {
          decision: {
            type: "choice",
            instructions: "Which answer choice is correct?",
            criteria: row.choices,
          },
        },
      }),
      signal: AbortSignal.timeout(120_000),
    });
    const responseText = await response.text();
    try { body = JSON.parse(responseText); } catch { body = undefined; }
    const attemptCost = readCost(body?.usage);
    if (attemptCost !== null) {
      accumulatedCost += attemptCost;
      sawCost = true;
    }
    responseMetadata = {
      costUsd: sawCost ? accumulatedCost : null,
      retryCount: attempt - 1,
    };
    if (response.status !== 429 || attempt === MAX_429_ATTEMPTS) break;
    const retryAfter = Number(response.headers.get("retry-after"));
    await delay(Number.isFinite(retryAfter) ? retryAfter * 1000 : 1000);
  }
  if (!response.ok) throw new Error(`OpenRouter returned HTTP ${response.status}.`);
  const answer = body?.answers?.decision;
  if (answer === null || typeof answer !== "object" || !Object.hasOwn(row.choices, answer.choice)) {
    throw new Error("Jev returned an invalid answer.");
  }
  return { predicted: answer.choice, ...responseMetadata };
}

let results = [];
let startedAt = new Date().toISOString();
let elapsedBeforeSession = 0;

function assertResumeMetadata(report) {
  const expected = {
    benchmark: benchmarkName,
    dataset: datasetMetadata,
    runtime: runtimeMetadata,
  };
  for (const key of Object.keys(expected)) {
    if (JSON.stringify(report[key]) !== JSON.stringify(expected[key])) {
      throw new Error(`Cannot resume: ${key} metadata does not match the current run.`);
    }
  }
  if (typeof report.startedAt !== "string" || !Number.isFinite(Date.parse(report.startedAt))) {
    throw new Error("Cannot resume: checkpoint has an invalid startedAt.");
  }
  if (!Array.isArray(report.results) || report.results.length > selectedRows.length
    || report.results.some((result, index) => result?.id !== selectedRows[index].id)) {
    throw new Error("Cannot resume: checkpoint result IDs are not a prefix of the selected rows.");
  }
}

if (resume) {
  const checkpoint = JSON.parse(readFileSync(outputPath, "utf8"));
  assertResumeMetadata(checkpoint);
  results = checkpoint.results;
  startedAt = checkpoint.startedAt;
  if (!Number.isFinite(checkpoint.summary?.wallSeconds) || checkpoint.summary.wallSeconds < 0) {
    throw new Error("Cannot resume: checkpoint has invalid elapsed time.");
  }
  elapsedBeforeSession = checkpoint.summary.wallSeconds;
}

const sessionStartedAt = performance.now();

function writeReport() {
  const elapsedSeconds = elapsedBeforeSession + (performance.now() - sessionStartedAt) / 1000;
  const report = {
    benchmark: benchmarkName,
    startedAt,
    dataset: datasetMetadata,
    runtime: runtimeMetadata,
    summary: summarize(results, elapsedSeconds),
    results,
  };
  const temporaryOutput = `${outputPath}.tmp`;
  writeFileSync(temporaryOutput, `${JSON.stringify(report, null, 2)}\n`);
  renameSync(temporaryOutput, outputPath);
  return report;
}

for (let index = 0; index < selectedRows.length; index++) {
  if (index < results.length && results[index].error === undefined) continue;
  const row = selectedRows[index];
  const decisionStartedAt = performance.now();
  responseMetadata = backend === "llama-cpp"
    ? { costUsd: 0, retryCount: 0 }
    : { costUsd: null, retryCount: 0 };
  try {
    const decision = await decide(row);
    const result = {
      id: row.id,
      gold: row.gold,
      predicted: decision.predicted,
      correct: decision.predicted === row.gold,
      latencyMs: performance.now() - decisionStartedAt,
      discipline: row.metadata.discipline,
      difficulty: row.metadata.difficulty,
      ...decision,
    };
    results[index] = result;
    console.log(`[${index + 1}/${selectedRows.length}] ${row.id} ${result.correct ? "correct" : "wrong"}`
      + ` ${result.latencyMs.toFixed(1)}ms`);
  } catch (error) {
    results[index] = {
      id: row.id,
      gold: row.gold,
      latencyMs: performance.now() - decisionStartedAt,
      discipline: row.metadata.discipline,
      difficulty: row.metadata.difficulty,
      ...responseMetadata,
      error: error instanceof Error ? `${error.name}: ${error.message}` : String(error),
    };
    console.error(
      `[${index + 1}/${selectedRows.length}] ${row.id} ERROR ${results[index].error}`,
    );
  }
  writeReport();
}

const report = writeReport();
console.log(JSON.stringify(report.summary, null, 2));
