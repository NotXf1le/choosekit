import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { performance } from "node:perf_hooks";
import { fromLlamaCpp } from "../dist/esm/llama-cpp.js";

function option(name, fallback) {
  const index = process.argv.indexOf(name);
  return index === -1 ? fallback : process.argv[index + 1];
}

function percentile(values, fraction) {
  if (values.length === 0) return null;
  const ordered = [...values].sort((a, b) => a - b);
  return ordered[Math.floor((ordered.length - 1) * fraction)];
}

function accuracy(rows) {
  return rows.length === 0 ? null : rows.filter((row) => row.correct).length / rows.length;
}

function balancedAccuracy(rows) {
  const classes = [...new Set(rows.map((row) => row.gold))];
  if (classes.length === 0) return null;
  return classes.reduce((sum, id) => {
    const members = rows.filter((row) => row.gold === id);
    return sum + members.filter((row) => row.predicted === id).length / members.length;
  }, 0) / classes.length;
}

function grouped(rows, field) {
  const groups = new Map();
  for (const row of rows) {
    const name = row[field];
    const members = groups.get(name);
    if (members) members.push(row);
    else groups.set(name, [row]);
  }
  return Object.fromEntries([...groups].map(([name, members]) => [name, {
    rows: members.length,
    accuracy: accuracy(members),
    balancedAccuracy: balancedAccuracy(members),
  }]));
}

function summarize(results, startedAt) {
  const successful = results.filter((row) => row.error === undefined);
  const families = grouped(successful, "family");
  const familyBalanced = Object.values(families).map((value) => value.balancedAccuracy)
    .filter((value) => value !== null);
  const latencies = successful.map((row) => row.latencyMs);
  const goldProbabilities = successful.map((row) => row.distribution[row.gold]);
  const nll = goldProbabilities.reduce((sum, value) => sum - Math.log(Math.max(value, Number.MIN_VALUE)), 0);
  const brier = successful.reduce((sum, row) => sum + row.optionIds.reduce((inner, id) => {
    const target = id === row.gold ? 1 : 0;
    return inner + (row.distribution[id] - target) ** 2;
  }, 0), 0);
  const usage = successful.reduce((total, row) => ({
    promptTokens: total.promptTokens + row.usage.promptTokens,
    cachedTokens: total.cachedTokens === null || row.usage.cachedTokens === null
      ? null : total.cachedTokens + row.usage.cachedTokens,
    completionTokens: total.completionTokens + row.usage.completionTokens,
    requests: total.requests + row.usage.requests,
  }), { promptTokens: 0, cachedTokens: 0, completionTokens: 0, requests: 0 });
  const elapsedSeconds = (performance.now() - startedAt) / 1000;
  return {
    rowsAttempted: results.length,
    rowsSuccessful: successful.length,
    errors: results.length - successful.length,
    accuracy: accuracy(successful),
    meanFamilyBalancedAccuracy: familyBalanced.length === 0
      ? null : familyBalanced.reduce((sum, value) => sum + value, 0) / familyBalanced.length,
    meanNll: successful.length === 0 ? null : nll / successful.length,
    meanMulticlassBrier: successful.length === 0 ? null : brier / successful.length,
    meanWinningProbability: successful.length === 0 ? null : successful.reduce(
      (sum, row) => sum + row.distribution[row.predicted], 0,
    ) / successful.length,
    latencyMs: {
      p50: percentile(latencies, 0.5),
      p95: percentile(latencies, 0.95),
      mean: latencies.length === 0
        ? null : latencies.reduce((sum, value) => sum + value, 0) / latencies.length,
    },
    wallSeconds: elapsedSeconds,
    decisionsPerSecond: elapsedSeconds === 0 ? null : successful.length / elapsedSeconds,
    usage,
    byFamily: families,
    byPartition: grouped(successful, "partition"),
  };
}

/** llama.cpp serves /v1/models beside /completion; mirror the adapter's base-URL handling. */
function modelsURL(baseURL) {
  const url = new URL(baseURL);
  if ((url.protocol !== "http:" && url.protocol !== "https:") || url.username || url.password
    || url.search || url.hash) {
    throw new TypeError("baseURL must be an HTTP(S) URL without credentials, query, or fragment.");
  }
  const path = url.pathname.replace(/\/v1\/?$/, "").replace(/\/$/, "");
  url.pathname = `${path}/v1/models`;
  return url.href;
}

async function servedModelIds(url) {
  const response = await fetch(url, { signal: AbortSignal.timeout(30_000) });
  if (!response.ok) throw new Error(`HTTP ${response.status} ${response.statusText}`);
  const body = await response.json();
  const entries = Array.isArray(body?.data) ? body.data : [];
  const ids = entries.map((entry) => entry?.id).filter((id) => typeof id === "string" && id.length > 0);
  if (ids.length === 0) throw new Error("the response listed no models");
  return ids;
}

/**
 * llama.cpp answers with whatever is loaded however the request names the
 * model, so an unchecked --model puts a name in the results that may never
 * have run. Confirm it against the server before spending an hour on rows.
 */
async function checkModel(baseURL, model) {
  const url = modelsURL(baseURL);
  let ids;
  try {
    ids = await servedModelIds(url);
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    throw new Error(`Could not list the models at ${url}: ${detail}.`
      + " Start the server first, or pass --skip-model-check to record the run as unverified.");
  }
  if (!ids.includes(model)) {
    throw new Error(`The server at ${url} serves ${ids.map((id) => `"${id}"`).join(", ")},`
      + ` not "${model}". llama.cpp answers with whatever is loaded however the request names the`
      + " model, so these results would carry a model name that never ran."
      + " Pass --model with one of the ids above, or --skip-model-check to record the run as unverified.");
  }
}

const input = option("--input", "benchmarks/data/semif-authored144.jsonl");
const mode = option("--mode", "labels");
if (mode !== "labels" && mode !== "minimal-prefix") {
  throw new TypeError("--mode must be labels or minimal-prefix.");
}
const output = option("--output", `benchmarks/results/semif-qwen3.8-27b-${mode}.json`);
const baseURL = option("--base-url", process.env.LLAMA_CPP_BASE_URL
  ?? "http://127.0.0.1:11434/");
const model = option("--model", process.env.LLAMA_CPP_MODEL ?? "qwen3.8-27b-text-64k");
const skipModelCheck = process.argv.includes("--skip-model-check");
const rawLimit = option("--limit", undefined);
const limit = rawLimit === undefined ? undefined : Number.parseInt(rawLimit, 10);

if (existsSync(output)) throw new Error(`Refusing to overwrite existing output: ${output}`);
if (limit !== undefined && (!Number.isSafeInteger(limit) || limit < 1)) {
  throw new TypeError("--limit must be a positive integer.");
}

const source = readFileSync(input);
const allRows = source.toString("utf8").trim().split(/\r?\n/).map((line) => JSON.parse(line));
if (allRows.length !== 144) throw new Error(`Expected 144 SemIf rows, received ${allRows.length}.`);
const rows = limit === undefined ? allRows : allRows.slice(0, limit);
mkdirSync(dirname(output), { recursive: true });
if (skipModelCheck) console.warn("--skip-model-check: the recorded model is the one requested, unconfirmed.");
else {
  await checkModel(baseURL, model);
  console.log(`Model ID confirmed in the server catalog: ${model}`);
}
const choose = fromLlamaCpp({ baseURL, model, mode });
const results = [];
const startedAt = performance.now();

for (let index = 0; index < rows.length; index++) {
  const row = rows[index];
  const gold = row.options[row.label]?.id;
  if (typeof gold !== "string") throw new Error(`Row ${row.id} has an invalid label.`);
  const choices = Object.fromEntries(row.options.map((entry) => [entry.id, entry.description]));
  const state = typeof row.state === "string" ? row.state : JSON.stringify(row.state);
  const mark = performance.now();
  try {
    const decision = await choose({
      context: state,
      question: row.question,
      choices,
      signal: AbortSignal.timeout(120_000),
    });
    const result = {
      id: row.id,
      family: row.family,
      partition: row.provenance?.partition ?? "unknown",
      variant: row.provenance?.variant ?? "unknown",
      optionIds: row.options.map((entry) => entry.id),
      gold,
      predicted: decision.choice,
      correct: decision.choice === gold,
      distribution: decision.distribution,
      scores: decision.scores,
      latencyMs: performance.now() - mark,
      boundaryTokens: decision.boundaryTokens,
      usage: decision.usage,
    };
    if (result.usage === undefined) throw new Error("Adapter did not report usage.");
    results.push(result);
    console.log(`[${index + 1}/${rows.length}] ${row.id} ${result.correct ? "correct" : "wrong"}`
      + ` ${result.latencyMs.toFixed(1)}ms`);
  } catch (error) {
    results.push({
      id: row.id,
      family: row.family,
      partition: row.provenance?.partition ?? "unknown",
      variant: row.provenance?.variant ?? "unknown",
      optionIds: row.options.map((entry) => entry.id),
      gold,
      latencyMs: performance.now() - mark,
      error: error instanceof Error ? `${error.name}: ${error.message}` : String(error),
    });
    console.error(`[${index + 1}/${rows.length}] ${row.id} ERROR ${results.at(-1).error}`);
  }

  const checkpoint = {
    benchmark: `SemIf authored144 through choosekit/llama-cpp (${mode})`,
    dataset: {
      source: "https://github.com/TheoLeeCJ/SemIf",
      revision: "b9cb32537e78be65f19abfcb1de8fc504b627d84",
      path: input,
      sha256: createHash("sha256").update(source).digest("hex"),
      totalRows: allRows.length,
      selectedRows: rows.length,
    },
    runtime: {
      baseURL,
      model,
      modelChecked: !skipModelCheck,
      mode,
      adapter: "choosekit/llama-cpp",
      packageVersion: "0.5.0",
    },
    interpretation: mode === "labels"
      ? "Package A/B/C label prompt and distinguishing-token likelihoods."
      : "Package original-key prompt and minimal distinguishing-prefix likelihoods.",
    summary: summarize(results, startedAt),
    results,
  };
  writeFileSync(output, `${JSON.stringify(checkpoint, null, 2)}\n`);
}

console.log(JSON.stringify(summarize(results, startedAt), null, 2));
