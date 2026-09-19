import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { performance } from "node:perf_hooks";

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

function numericUsage(value, snake, camel) {
  const candidate = value?.[snake] ?? value?.[camel];
  return typeof candidate === "number" && Number.isFinite(candidate) ? candidate : null;
}

function summarize(results, startedAt) {
  const successful = results.filter((row) => row.error === undefined);
  const families = grouped(successful, "family");
  const familyBalanced = Object.values(families).map((value) => value.balancedAccuracy)
    .filter((value) => value !== null);
  const latencies = successful.map((row) => row.latencyMs);
  const goldProbabilities = successful.map((row) => row.probabilities[row.gold]);
  const nll = goldProbabilities.reduce((sum, value) => sum - Math.log(Math.max(value, Number.MIN_VALUE)), 0);
  const brier = successful.reduce((sum, row) => sum + row.optionIds.reduce((inner, id) => {
    const target = id === row.gold ? 1 : 0;
    return inner + (row.probabilities[id] - target) ** 2;
  }, 0), 0);
  const usage = successful.reduce((total, row) => {
    const input = numericUsage(row.usage, "input_tokens", "inputTokens");
    const output = numericUsage(row.usage, "output_tokens", "outputTokens");
    const cost = numericUsage(row.usage, "cost", "cost");
    return {
      inputTokens: total.inputTokens === null || input === null ? null : total.inputTokens + input,
      outputTokens: total.outputTokens === null || output === null ? null : total.outputTokens + output,
      costUsd: total.costUsd === null || cost === null ? null : total.costUsd + cost,
    };
  }, { inputTokens: 0, outputTokens: 0, costUsd: 0 });
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
    meanConfidence: successful.length === 0 ? null : successful.reduce(
      (sum, row) => sum + row.confidence, 0,
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

function loadApiKey() {
  const key = process.env.OPENROUTER_API_KEY;
  if (typeof key !== "string" || key.length < 20) {
    throw new Error("OPENROUTER_API_KEY is missing or invalid.");
  }
  return key;
}

function validateAnswer(answer, optionIds) {
  if (answer === null || typeof answer !== "object" || Array.isArray(answer)) {
    throw new Error("Jev returned an invalid answer object.");
  }
  if (!optionIds.includes(answer.choice)) throw new Error("Jev selected an undeclared option.");
  if (answer.probabilities === null || typeof answer.probabilities !== "object"
    || Array.isArray(answer.probabilities)) {
    throw new Error("Jev did not return a probability distribution.");
  }
  let total = 0;
  for (const id of optionIds) {
    const probability = answer.probabilities[id];
    if (typeof probability !== "number" || !Number.isFinite(probability)
      || probability < 0 || probability > 1) {
      throw new Error(`Jev returned an invalid probability for ${id}.`);
    }
    total += probability;
  }
  if (Math.abs(total - 1) > 1e-6) throw new Error(`Jev probabilities sum to ${total}, not 1.`);
  if (typeof answer.confidence !== "number" || !Number.isFinite(answer.confidence)
    || answer.confidence < 0 || answer.confidence > 1) {
    throw new Error("Jev returned invalid confidence.");
  }
}

const input = option("--input", "benchmarks/data/semif-authored144.jsonl");
const output = option("--output", "benchmarks/results/semif-openrouter-jev-latest.json");
const model = option("--model", "typesafe/jev-1.13");
const rawLimit = option("--limit", undefined);
const limit = rawLimit === undefined ? undefined : Number.parseInt(rawLimit, 10);

if (existsSync(output)) throw new Error(`Refusing to overwrite existing output: ${output}`);
if (limit !== undefined && (!Number.isSafeInteger(limit) || limit < 1)) {
  throw new TypeError("--limit must be a positive integer.");
}

const apiKey = loadApiKey();
const source = readFileSync(input);
const allRows = source.toString("utf8").trim().split(/\r?\n/).map((line) => JSON.parse(line));
if (allRows.length !== 144) throw new Error(`Expected 144 SemIf rows, received ${allRows.length}.`);
const rows = limit === undefined ? allRows : allRows.slice(0, limit);
mkdirSync(dirname(output), { recursive: true });
const results = [];
const startedAt = performance.now();

for (let index = 0; index < rows.length; index++) {
  const row = rows[index];
  const gold = row.options[row.label]?.id;
  if (typeof gold !== "string") throw new Error(`Row ${row.id} has an invalid label.`);
  const optionIds = row.options.map((entry) => entry.id);
  const mark = performance.now();
  try {
    const response = await fetch("https://openrouter.ai/api/alpha/decisions", {
      method: "POST",
      headers: {
        authorization: `Bearer ${apiKey}`,
        "content-type": "application/json",
        "x-openrouter-title": "choosekit SemIf benchmark",
      },
      body: JSON.stringify({
        model,
        state: row.state,
        questions: {
          decision: {
            type: "choice",
            instructions: row.question,
            criteria: Object.fromEntries(row.options.map((entry) => [entry.id, entry.description])),
          },
        },
      }),
      signal: AbortSignal.timeout(120_000),
    });
    const body = await response.json();
    if (!response.ok) {
      throw new Error(`OpenRouter HTTP ${response.status}: ${JSON.stringify(body)}`);
    }
    const answer = body?.answers?.decision;
    validateAnswer(answer, optionIds);
    const result = {
      id: row.id,
      family: row.family,
      partition: row.provenance?.partition ?? "unknown",
      variant: row.provenance?.variant ?? "unknown",
      optionIds,
      gold,
      predicted: answer.choice,
      correct: answer.choice === gold,
      probabilities: answer.probabilities,
      confidence: answer.confidence,
      latencyMs: performance.now() - mark,
      resolvedModel: body.model,
      provider: body.provider,
      usage: body.usage ?? null,
    };
    results.push(result);
    console.log(`[${index + 1}/${rows.length}] ${row.id} ${result.correct ? "correct" : "wrong"}`
      + ` ${result.latencyMs.toFixed(1)}ms`);
  } catch (error) {
    results.push({
      id: row.id,
      family: row.family,
      partition: row.provenance?.partition ?? "unknown",
      variant: row.provenance?.variant ?? "unknown",
      optionIds,
      gold,
      latencyMs: performance.now() - mark,
      error: error instanceof Error ? `${error.name}: ${error.message}` : String(error),
    });
    console.error(`[${index + 1}/${rows.length}] ${row.id} ERROR ${results.at(-1).error}`);
  }

  const checkpoint = {
    benchmark: "SemIf authored144 through OpenRouter Jev Decisions API",
    dataset: {
      source: "https://github.com/TheoLeeCJ/SemIf",
      revision: "b9cb32537e78be65f19abfcb1de8fc504b627d84",
      path: input,
      sha256: createHash("sha256").update(source).digest("hex"),
      totalRows: allRows.length,
      selectedRows: rows.length,
    },
    runtime: { endpoint: "https://openrouter.ai/api/alpha/decisions", requestedModel: model },
    summary: summarize(results, startedAt),
    results,
  };
  writeFileSync(output, `${JSON.stringify(checkpoint, null, 2)}\n`);
}

console.log(JSON.stringify(summarize(results, startedAt), null, 2));
