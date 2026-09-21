import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import {
  SUPERGPQA_EVALUATION_SAMPLE_SEED,
  SUPERGPQA_PILOT_ROWS,
  SUPERGPQA_PREPARED_SHA256,
  sampleSuperGpqaEvaluationRows,
} from "./prepare-supergpqa.mjs";

const EVALUATION_ROWS = 1000;
const benchmarksDirectory = new URL("./", import.meta.url);
const runInputs = [
  {
    file: "supergpqa-granite-4.0-h-micro-cloudflare.json",
    id: "granite-4.0-h-micro",
    label: "Granite 4.0 H Micro",
    series: "openrouter",
    model: "ibm-granite/granite-4.0-h-micro",
    provider: "cloudflare",
    resolvedProvider: "Cloudflare",
  },
  {
    file: "supergpqa-llama-3.1-8b-novita.json",
    id: "llama-3.1-8b",
    label: "Llama 3.1 8B",
    series: "openrouter",
    model: "meta-llama/llama-3.1-8b-instruct",
    provider: "novita",
    resolvedProvider: "Novita",
    labelDy: 27,
  },
  {
    file: "supergpqa-glm-4.7-flash-cloudflare.json",
    id: "glm-4.7-flash",
    label: "GLM 4.7 Flash",
    series: "openrouter",
    model: "z-ai/glm-4.7-flash",
    provider: "cloudflare",
    resolvedProvider: "Cloudflare",
  },
  {
    file: "supergpqa-glm-5.2-cloudflare.json",
    id: "glm-5.2",
    label: "GLM 5.2",
    series: "openrouter",
    model: "z-ai/glm-5.2",
    provider: "cloudflare",
    resolvedProvider: "Cloudflare",
  },
  {
    file: "supergpqa-gemma-4-26b-dekallm.json",
    id: "gemma-4-26b",
    label: "Gemma 4 26B",
    series: "openrouter",
    model: "google/gemma-4-26b-a4b-it",
    provider: "dekallm",
    resolvedProvider: "DekaLLM",
  },
  {
    file: "supergpqa-granite-4.2-8b-coreweave.json",
    id: "granite-4.2-8b",
    label: "Granite 4.2 8B",
    series: "openrouter",
    model: "ibm-granite/granite-4.2-8b",
    provider: "coreweave",
    resolvedProvider: "CoreWeave",
    labelDy: 22,
  },
  {
    file: "supergpqa-deepseek-v4.1-flash-wafer.json",
    id: "deepseek-v4.1-flash",
    label: "DeepSeek V4.1 Flash",
    series: "openrouter",
    model: "deepseek/deepseek-v4.1-flash",
    provider: "wafer",
    resolvedProvider: "Wafer",
  },
  {
    file: "supergpqa-deepseek-v4-pro-cloudflare.json",
    id: "deepseek-v4-pro",
    label: "DeepSeek V4 Pro",
    series: "openrouter",
    model: "deepseek/deepseek-v4-pro-0813",
    provider: "cloudflare",
    resolvedProvider: "Cloudflare",
    labelDy: 27,
  },
  {
    file: "supergpqa-kimi-k3-morph.json",
    id: "kimi-k3",
    label: "Kimi K3",
    series: "openrouter",
    model: "moonshotai/kimi-k3",
    provider: "morph",
    resolvedProvider: "Morph",
    labelDx: -10,
    labelAnchor: "end",
  },
  {
    file: "supergpqa-jev-1.13.json",
    id: "jev-1.13",
    label: "Jev 1.13",
    series: "jev",
    model: "typesafe/jev-1.13",
  },
];
const localReferenceInput = {
  file: "supergpqa-qwen3.8-27b-local.json",
  id: "qwen3.8-27b-local",
  label: "Local Qwen3.8 27B Q4_XL",
  series: "llama-cpp",
  model: "qwen3.8-27b-text-64k",
};

function invariant(condition, message) {
  if (!condition) throw new Error(message);
}

function formatNumber(value, digits = 2) {
  return Number(value.toFixed(digits)).toString();
}

function wilson95(correct, total) {
  const z = 1.959963984540054;
  const p = correct / total;
  const z2 = z ** 2;
  const denominator = 1 + z2 / total;
  const center = (p + z2 / (2 * total)) / denominator;
  const margin = (z / denominator)
    * Math.sqrt((p * (1 - p) + z2 / (4 * total)) / total);
  return [center - margin, center + margin];
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[middle - 1] + sorted[middle]) / 2
    : sorted[middle];
}

async function loadRun(input) {
  const report = JSON.parse(
    await readFile(new URL(`results/${input.file}`, benchmarksDirectory), "utf8"),
  );
  invariant(
    report.dataset?.selectedRows === EVALUATION_ROWS,
    `${input.file}: expected ${EVALUATION_ROWS} selected rows`,
  );
  invariant(report.dataset?.sample?.method === "proportional-discipline-difficulty",
    `${input.file}: unexpected sample method`);
  invariant(report.dataset?.sha256 === SUPERGPQA_PREPARED_SHA256,
    `${input.file}: unexpected prepared dataset hash`);
  invariant(report.dataset?.sample?.seed === SUPERGPQA_EVALUATION_SAMPLE_SEED,
    `${input.file}: unexpected sample seed`);
  invariant(
    report.dataset?.sample?.excludedPilotSampleSize === SUPERGPQA_PILOT_ROWS,
    `${input.file}: unexpected excluded pilot sample size`,
  );
  invariant(
    report.summary?.rowsAttempted === EVALUATION_ROWS,
    `${input.file}: expected ${EVALUATION_ROWS} attempted rows`,
  );
  invariant(
    report.summary?.rowsSuccessful === EVALUATION_ROWS,
    `${input.file}: expected ${EVALUATION_ROWS} successful rows`,
  );
  invariant(report.summary?.errors === 0, `${input.file}: expected zero errors`);
  invariant(report.summary?.costComplete === true, `${input.file}: incomplete cost data`);
  invariant(Number.isFinite(report.summary?.costUsd)
    && (input.series === "llama-cpp" ? report.summary.costUsd === 0 : report.summary.costUsd > 0),
  `${input.file}: invalid cost`);
  invariant(Number.isFinite(report.summary?.decisionsPerSecond) && report.summary.decisionsPerSecond > 0,
    `${input.file}: invalid throughput`);
  invariant(Array.isArray(report.results) && report.results.length === EVALUATION_ROWS,
    `${input.file}: expected ${EVALUATION_ROWS} result rows`);
  invariant(report.runtime?.model === input.model, `${input.file}: unexpected model`);
  invariant(report.runtime?.backend === input.series, `${input.file}: unexpected backend`);
  if (input.series === "openrouter") {
    invariant(report.runtime?.provider === input.provider, `${input.file}: unexpected provider`);
    invariant(report.results.every((row) => row.resolvedModel === input.model),
      `${input.file}: OpenRouter resolved a different model`);
    invariant(report.results.every((row) => row.resolvedProvider === input.resolvedProvider),
      `${input.file}: OpenRouter resolved a different provider`);
  }
  invariant(report.results.every((row) => Number.isFinite(row.latencyMs) && row.latencyMs > 0),
    `${input.file}: invalid latency`);
  const correct = report.results.filter((row) => row.correct === true).length;
  invariant(report.summary.endToEndAccuracy === correct / EVALUATION_ROWS,
    `${input.file}: accuracy does not match result rows`);
  return {
    sha256: report.dataset.sha256,
    ids: report.results.map((row) => row.id),
    run: {
      id: input.id,
      label: input.label,
      series: input.series,
      model: input.model,
      ...(input.provider === undefined ? {} : { provider: input.provider }),
      ...(input.resolvedProvider === undefined ? {} : { resolvedProvider: input.resolvedProvider }),
      ...(input.labelDx === undefined ? {} : { labelDx: input.labelDx }),
      ...(input.labelDy === undefined ? {} : { labelDy: input.labelDy }),
      ...(input.labelAnchor === undefined ? {} : { labelAnchor: input.labelAnchor }),
      correct,
      totalCostUsd: report.summary.costUsd,
      medianLatencyMs: median(report.results.map((row) => row.latencyMs)),
      decisionsPerSecond: report.summary.decisionsPerSecond,
    },
  };
}

function renderSvg(aggregate) {
  const width = 920;
  const height = 540;
  const plot = { left: 82, right: 882, top: 76, bottom: 430 };
  const points = aggregate.runs.map((run) => ({
    ...run,
    xValue: (run.totalCostUsd / aggregate.dataset.rows) * (1 / run.decisionsPerSecond),
    yValue: run.correct / aggregate.dataset.rows,
  }));
  const logMin = -6.2;
  const logMax = -2.7;
  const yMax = 0.7;
  const scaleX = (value) => plot.left + ((Math.log10(value) - logMin) / (logMax - logMin))
    * (plot.right - plot.left);
  const scaleY = (value) => plot.bottom - (value / yMax) * (plot.bottom - plot.top);
  const xTicks = [0.000001, 0.00001, 0.0001, 0.001];
  const yTicks = [0, 0.2, 0.4, 0.6];
  const lines = [];

  lines.push(`<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="title desc">`);
  lines.push("  <title id=\"title\">SuperGPQA benchmark: ChooseKit models and Jev</title>");
  lines.push(`  <desc id="desc">Accuracy on the same ${aggregate.dataset.rows} proportionally sampled questions. Only OpenRouter models with top_logprobs are included. Lower cost per decision times seconds per decision and higher accuracy are better. Vertical bars show 95 percent Wilson intervals. A horizontal line shows local Qwen3.8 27B Q4_XL accuracy without assigning it a cloud cost coordinate.</desc>`);
  lines.push(`  <rect width="${width}" height="${height}" rx="16" fill="#ffffff"/>`);
  lines.push("  <g font-family=\"Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif\">");
  lines.push(`    <text x="${plot.left}" y="31" fill="#111827" font-size="20" font-weight="650">SuperGPQA decision benchmark</text>`);
  lines.push(`    <text x="${plot.left}" y="52" fill="#6b7280" font-size="12">proportional sample · ${aggregate.dataset.rows} questions · ChooseKit OpenRouter providers pinned · 2026-09-21</text>`);
  lines.push(`    <line x1="490" y1="29" x2="500" y2="29" stroke="#15966b" stroke-width="1.5"/><text x="508" y="33" fill="#4b5563" font-size="10">ChooseKit + llama.cpp</text>`);
  lines.push(`    <circle cx="650" cy="29" r="5" fill="#356ae6"/><text x="662" y="33" fill="#4b5563" font-size="10">ChooseKit + OpenRouter</text>`);
  lines.push(`    <path d="M 816 23 L 822 29 L 816 35 L 810 29 Z" fill="#e85d3f"/><text x="830" y="33" fill="#4b5563" font-size="10">Jev</text>`);

  for (const tick of yTicks) {
    const y = scaleY(tick);
    lines.push(`    <line x1="${plot.left}" y1="${formatNumber(y)}" x2="${plot.right}" y2="${formatNumber(y)}" stroke="#e5e7eb"/>`);
    lines.push(`    <text x="${plot.left - 12}" y="${formatNumber(y + 4)}" fill="#6b7280" font-size="11" text-anchor="end">${Math.round(tick * 100)}%</text>`);
  }
  for (const tick of xTicks) {
    const x = scaleX(tick);
    lines.push(`    <line x1="${formatNumber(x)}" y1="${plot.top}" x2="${formatNumber(x)}" y2="${plot.bottom}" stroke="#f3f4f6"/>`);
    lines.push(`    <text x="${formatNumber(x)}" y="${plot.bottom + 24}" fill="#6b7280" font-size="11" text-anchor="middle">${tick.toExponential(0)}</text>`);
  }
  const baselineY = scaleY(aggregate.dataset.randomBaseline);
  lines.push(`    <line x1="${plot.left}" y1="${formatNumber(baselineY)}" x2="${plot.right}" y2="${formatNumber(baselineY)}" stroke="#6b7280" stroke-width="1.2" stroke-dasharray="5 5"/>`);
  lines.push(`    <text x="${plot.right - 4}" y="${formatNumber(baselineY - 7)}" fill="#6b7280" font-size="10" text-anchor="end">random choice ${formatNumber(aggregate.dataset.randomBaseline * 100, 1)}%</text>`);
  const localY = scaleY(aggregate.localReference.correct / aggregate.dataset.rows);
  const [localLower, localUpper] = wilson95(
    aggregate.localReference.correct,
    aggregate.dataset.rows,
  );
  const localUpperY = scaleY(localUpper);
  const localLowerY = scaleY(localLower);
  lines.push(`    <rect x="${plot.left}" y="${formatNumber(localUpperY)}" width="${plot.right - plot.left}" height="${formatNumber(localLowerY - localUpperY)}" fill="#15966b" opacity="0.08"/>`);
  lines.push(`    <line x1="${plot.left}" y1="${formatNumber(localY)}" x2="${plot.right}" y2="${formatNumber(localY)}" stroke="#15966b" stroke-width="1.5" stroke-dasharray="7 5"/>`);
  lines.push(`    <text x="${plot.left + 4}" y="${formatNumber(localUpperY - 6)}" fill="#117a58" font-size="10" font-weight="600">${aggregate.localReference.label} · ${formatNumber((aggregate.localReference.correct / aggregate.dataset.rows) * 100, 1)}%</text>`);
  lines.push(`    <line x1="${plot.left}" y1="${plot.bottom}" x2="${plot.right}" y2="${plot.bottom}" stroke="#9ca3af"/>`);
  lines.push(`    <text x="${(plot.left + plot.right) / 2}" y="482" fill="#4b5563" font-size="12" text-anchor="middle">Cost–latency product ↓ (log scale)</text>`);
  lines.push(`    <text x="${(plot.left + plot.right) / 2}" y="511" fill="#6b7280" font-size="10" text-anchor="middle">Only OpenRouter models with top_logprobs are included · measured end to end · bars and green band show 95% Wilson intervals</text>`);
  lines.push(`    <text x="20" y="${(plot.top + plot.bottom) / 2}" fill="#4b5563" font-size="12" text-anchor="middle" transform="rotate(-90 20 ${(plot.top + plot.bottom) / 2})">Accuracy →</text>`);

  for (const point of points) {
    const x = scaleX(point.xValue);
    const y = scaleY(point.yValue);
    const color = point.series === "jev" ? "#e85d3f" : "#356ae6";
    const [lower, upper] = wilson95(point.correct, aggregate.dataset.rows);
    const upperY = scaleY(upper);
    const lowerY = scaleY(lower);
    lines.push(`    <line x1="${formatNumber(x)}" y1="${formatNumber(upperY)}" x2="${formatNumber(x)}" y2="${formatNumber(lowerY)}" stroke="${color}" stroke-width="1.5" opacity="0.5"/>`);
    lines.push(`    <line x1="${formatNumber(x - 4)}" y1="${formatNumber(upperY)}" x2="${formatNumber(x + 4)}" y2="${formatNumber(upperY)}" stroke="${color}" stroke-width="1.5" opacity="0.5"/>`);
    lines.push(`    <line x1="${formatNumber(x - 4)}" y1="${formatNumber(lowerY)}" x2="${formatNumber(x + 4)}" y2="${formatNumber(lowerY)}" stroke="${color}" stroke-width="1.5" opacity="0.5"/>`);
    if (point.series === "jev") {
      lines.push(`    <path d="M ${formatNumber(x)} ${formatNumber(y - 7)} L ${formatNumber(x + 7)} ${formatNumber(y)} L ${formatNumber(x)} ${formatNumber(y + 7)} L ${formatNumber(x - 7)} ${formatNumber(y)} Z" fill="${color}"/>`);
    } else {
      lines.push(`    <circle cx="${formatNumber(x)}" cy="${formatNumber(y)}" r="6" fill="${color}"/>`);
    }
    const labelX = x + (point.labelDx ?? 0);
    const labelY = y + (point.labelDy ?? -16);
    lines.push(`    <text x="${formatNumber(labelX)}" y="${formatNumber(labelY)}" fill="#111827" font-size="11" font-weight="600" text-anchor="${point.labelAnchor ?? "middle"}">${point.label} · ${formatNumber(point.yValue * 100, 1)}%</text>`);
  }
  lines.push("  </g>");
  lines.push("</svg>");
  return `${lines.join("\n")}\n`;
}

const loadedRuns = await Promise.all(runInputs.map(loadRun));
const localReference = await loadRun(localReferenceInput);
const allRuns = [...loadedRuns, localReference];
invariant(new Set(allRuns.map(({ sha256 }) => sha256)).size === 1,
  "all reports must use the same dataset hash");
const expectedIds = JSON.stringify(loadedRuns[0].ids);
invariant(allRuns.every(({ ids }) => JSON.stringify(ids) === expectedIds),
  "all reports must contain the same questions in the same order");

const preparedDataset = await readFile(
  new URL(".data/supergpqa/supergpqa.jsonl", benchmarksDirectory),
);
const preparedSha256 = createHash("sha256").update(preparedDataset).digest("hex");
invariant(preparedSha256 === SUPERGPQA_PREPARED_SHA256,
  "prepared dataset has an unexpected hash");
const preparedRows = preparedDataset.toString("utf8").trim()
  .split(/\r?\n/).map((line) => JSON.parse(line));
const rowsById = new Map(preparedRows.map((row) => [row.id, row]));
const evaluationRows = loadedRuns[0].ids.map((id) => rowsById.get(id));
invariant(evaluationRows.every(Boolean), "result IDs must exist in the prepared dataset");
const expectedEvaluationIds = sampleSuperGpqaEvaluationRows(
  preparedRows,
  EVALUATION_ROWS,
  SUPERGPQA_PILOT_ROWS,
).map(({ id }) => id);
invariant(JSON.stringify(loadedRuns[0].ids) === JSON.stringify(expectedEvaluationIds),
  "reports must contain the expected evaluation sample in order");
const randomBaseline = evaluationRows.reduce(
  (sum, row) => sum + 1 / Object.keys(row.choices).length,
  0,
) / evaluationRows.length;

const aggregate = {
  dataset: {
    rows: EVALUATION_ROWS,
    sha256: loadedRuns[0].sha256,
    randomBaseline,
    sample: "proportional-discipline-difficulty",
  },
  runs: loadedRuns.map(({ run }) => run),
  localReference: localReference.run,
};

await writeFile(
  new URL("supergpqa-benchmark.json", benchmarksDirectory),
  `${JSON.stringify(aggregate, null, 2)}\n`,
);
await writeFile(new URL("supergpqa-benchmark.svg", benchmarksDirectory), renderSvg(aggregate));
console.log(`Wrote ${fileURLToPath(new URL("supergpqa-benchmark.json", benchmarksDirectory))}`);
console.log(`Wrote ${fileURLToPath(new URL("supergpqa-benchmark.svg", benchmarksDirectory))}`);
