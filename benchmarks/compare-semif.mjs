import { existsSync, readFileSync, writeFileSync } from "node:fs";

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

const qwen = load(option("--qwen"), "qwen");
const jev = load(option("--jev"), "jev");
const output = option("--output");
if (output && existsSync(output)) throw new Error(`Refusing to overwrite existing output: ${output}`);

const jevById = new Map(jev.results.map((row) => [row.id, row]));
const qwenProbabilities = [];
const jevProbabilities = [];
const distances = [];
let sameChoice = 0;

for (const row of qwen.results) {
  if (row.error) throw new Error(`Qwen row ${row.id} failed: ${row.error}`);
  const other = jevById.get(row.id);
  if (!other || other.error) throw new Error(`Missing successful Jev row ${row.id}.`);
  if (row.predicted === other.predicted) sameChoice++;
  let distance = 0;
  for (const id of row.optionIds) {
    const a = row.distribution?.[id];
    const b = other.probabilities?.[id];
    if (!Number.isFinite(a) || !Number.isFinite(b)) {
      throw new Error(`Missing probability for ${row.id}/${id}.`);
    }
    qwenProbabilities.push(a);
    jevProbabilities.push(b);
    distance += Math.abs(a - b);
  }
  distances.push(distance / 2);
}
if (jevById.size !== qwen.results.length) throw new Error("The reports contain different case sets.");

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
