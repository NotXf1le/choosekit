import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";

function option(name) {
  const index = process.argv.indexOf(name);
  return index === -1 ? undefined : process.argv[index + 1];
}

function load(path, name) {
  if (!path) throw new TypeError(`--${name} is required.`);
  const report = JSON.parse(readFileSync(path, "utf8"));
  if (!Array.isArray(report.results) || report.results.length === 0) {
    throw new TypeError(`${name} report has no results.`);
  }
  return report;
}

function mean(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function validateReport(report, name, probabilityField) {
  const rowsById = new Map();

  for (const [index, row] of report.results.entries()) {
    if (row === null || typeof row !== "object"
      || typeof row.id !== "string" || row.id.length === 0) {
      throw new Error(`${name} report row ${index} has an invalid case ID.`);
    }
    if (rowsById.has(row.id)) {
      throw new Error(`${name} report contains duplicate case ID: ${row.id}.`);
    }
    if (row.error) throw new Error(`${name} row ${row.id} failed: ${row.error}`);
    if (!Array.isArray(row.optionIds) || row.optionIds.length === 0
      || row.optionIds.some((id) => typeof id !== "string" || id.length === 0)
      || new Set(row.optionIds).size !== row.optionIds.length) {
      throw new Error(`${name} row ${row.id} has invalid option IDs.`);
    }
    if (typeof row.gold !== "string" || !row.optionIds.includes(row.gold)) {
      throw new Error(`${name} row ${row.id} has a gold label outside its options.`);
    }

    const probabilities = row[probabilityField];
    if (probabilities === null || typeof probabilities !== "object" || Array.isArray(probabilities)) {
      throw new Error(`${name} row ${row.id} has an invalid probability distribution.`);
    }
    const probabilityKeys = Object.keys(probabilities);
    const optionSet = new Set(row.optionIds);
    if (probabilityKeys.length !== row.optionIds.length
      || probabilityKeys.some((id) => !optionSet.has(id))) {
      throw new Error(`${name} row ${row.id} probability keys do not match its options.`);
    }

    let total = 0;
    for (const id of row.optionIds) {
      const probability = probabilities[id];
      if (!Number.isFinite(probability) || probability < 0 || probability > 1) {
        throw new Error(`${name} row ${row.id} has an invalid probability for ${id}.`);
      }
      total += probability;
    }
    if (Math.abs(total - 1) > 1e-6) {
      throw new Error(`${name} row ${row.id} probabilities sum to ${total}, not 1.`);
    }

    rowsById.set(row.id, row);
  }

  return rowsById;
}

const qwen = load(option("--qwen"), "qwen");
const jev = load(option("--jev"), "jev");
const output = option("--output");
if (output && existsSync(output)) throw new Error(`Refusing to overwrite existing output: ${output}`);
if (output) mkdirSync(dirname(output), { recursive: true });

const qwenById = validateReport(qwen, "Qwen", "distribution");
const jevById = validateReport(jev, "Jev", "probabilities");
if (qwenById.size !== jevById.size
  || [...qwenById.keys()].some((id) => !jevById.has(id))) {
  throw new Error("The reports contain different case sets.");
}
const qwenProbabilities = [];
const jevProbabilities = [];
const distances = [];
let sameChoice = 0;

for (const row of qwen.results) {
  const other = jevById.get(row.id);
  if (row.optionIds.length !== other.optionIds.length
    || !row.optionIds.every((id, index) => id === other.optionIds[index])) {
    throw new Error(`The reports have different option IDs for case ${row.id}.`);
  }
  if (row.gold !== other.gold) {
    throw new Error(`The reports have different gold labels for case ${row.id}.`);
  }
  if (row.predicted === other.predicted) sameChoice++;
  let distance = 0;
  for (const id of row.optionIds) {
    const a = row.distribution[id];
    const b = other.probabilities[id];
    qwenProbabilities.push(a);
    jevProbabilities.push(b);
    distance += Math.abs(a - b);
  }
  distances.push(distance / 2);
}

const qwenMean = mean(qwenProbabilities);
const jevMean = mean(jevProbabilities);
let covariance = 0;
let qwenVariance = 0;
let jevVariance = 0;
for (let i = 0; i < qwenProbabilities.length; i++) {
  const a = qwenProbabilities[i] - qwenMean;
  const b = jevProbabilities[i] - jevMean;
  covariance += a * b;
  qwenVariance += a * a;
  jevVariance += b * b;
}
const ordered = [...distances].sort((a, b) => a - b);
const comparison = {
  cases: distances.length,
  optionProbabilities: qwenProbabilities.length,
  sameTopChoice: sameChoice,
  sameTopChoiceRate: sameChoice / distances.length,
  pearsonCorrelation: covariance / Math.sqrt(qwenVariance * jevVariance),
  medianTotalVariationDistance: ordered[Math.floor((ordered.length - 1) * 0.5)],
  meanTotalVariationDistance: mean(distances),
  casesWithinFivePercentTvd: distances.filter((value) => value <= 0.05).length,
  casesWithinTenPercentTvd: distances.filter((value) => value <= 0.10).length,
  casesAboveTwentyPercentTvd: distances.filter((value) => value > 0.20).length,
  maximumTotalVariationDistance: Math.max(...distances),
};

const serialized = `${JSON.stringify(comparison, null, 2)}\n`;
if (output) writeFileSync(output, serialized);
console.log(serialized.trimEnd());
