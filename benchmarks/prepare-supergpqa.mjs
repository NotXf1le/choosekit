import { createHash } from "node:crypto";
import { createReadStream, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { pathToFileURL } from "node:url";

export const SUPERGPQA_REVISION = "4430d4458112c7d4497fdcf94d7cc223313d6acf";
export const SUPERGPQA_SHA256 = "28b998e70205ee95e540317b5adc06a06552a3961fb50b153df126b833f7a910";
export const SUPERGPQA_PREPARED_SHA256 = "d17a48292cc8f081ca576afc7e8eaa6ca510de59fa9881282922409666bb3aff";
export const SUPERGPQA_ROWS = 26529;
export const SUPERGPQA_COMPATIBILITY_SAMPLE_SEED = "choosekit-supergpqa-v1";
export const SUPERGPQA_EVALUATION_SAMPLE_SEED = "choosekit-supergpqa-final-v2";

function option(name, fallback) {
  const index = process.argv.indexOf(name);
  if (index === -1) return fallback;
  const value = process.argv[index + 1];
  if (value === undefined || value.startsWith("--")) throw new TypeError(`${name} requires a value.`);
  return value;
}

function requireText(value, name, id) {
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new TypeError(`SuperGPQA row ${id} has an invalid ${name}.`);
  }
  return value.trim();
}

function stableRank(value, seed = SUPERGPQA_COMPATIBILITY_SAMPLE_SEED) {
  return createHash("sha256").update(`${seed}\0${value}`).digest("hex");
}

export function prepareSuperGpqaRows(rows) {
  if (!Array.isArray(rows)) throw new TypeError("SuperGPQA rows must be an array.");
  const ids = new Set();
  return Object.freeze(rows.map((row, index) => {
    if (row === null || typeof row !== "object" || Array.isArray(row)) {
      throw new TypeError(`SuperGPQA row ${index + 1} is not an object.`);
    }
    const id = requireText(row.uuid, "uuid", index + 1);
    if (ids.has(id)) throw new Error(`SuperGPQA contains duplicate row ID ${id}.`);
    ids.add(id);
    const question = requireText(row.question, "question", id);
    if (!Array.isArray(row.options) || row.options.length < 2 || row.options.length > 20) {
      throw new TypeError(`SuperGPQA row ${id} has invalid options.`);
    }
    const answerLetter = requireText(row.answer_letter, "answer_letter", id);
    const answerIndex = answerLetter.charCodeAt(0) - 65;
    if (!/^[A-T]$/.test(answerLetter) || answerIndex >= row.options.length) {
      throw new Error(`SuperGPQA row ${id} has an answer outside its options.`);
    }
    const choices = {};
    for (const [optionIndex, value] of row.options.entries()) {
      choices[String.fromCharCode(65 + optionIndex)] = requireText(value, `option ${optionIndex + 1}`, id);
    }
    const answer = requireText(row.answer, "answer", id);
    if (choices[answerLetter] !== answer) {
      throw new Error(`SuperGPQA row ${id} has inconsistent answer text and letter.`);
    }
    return Object.freeze({
      id,
      context: "",
      question,
      choices: Object.freeze(choices),
      gold: answerLetter,
      metadata: Object.freeze({
        discipline: requireText(row.discipline, "discipline", id),
        field: requireText(row.field, "field", id),
        subfield: requireText(row.subfield, "subfield", id),
        difficulty: requireText(row.difficulty, "difficulty", id),
        isCalculation: row.is_calculation === true,
      }),
    });
  }));
}

export function sampleSuperGpqaCompatibilityRows(rows, size) {
  if (!Array.isArray(rows)) throw new TypeError("SuperGPQA rows must be an array.");
  if (!Number.isSafeInteger(size) || size < 1 || size > rows.length) {
    throw new TypeError("SuperGPQA sample size must be a positive integer no larger than the dataset.");
  }
  const groups = new Map();
  for (const row of rows) {
    const key = `${row.metadata?.discipline}\0${row.metadata?.difficulty}`;
    const group = groups.get(key) ?? [];
    group.push(row);
    groups.set(key, group);
  }
  const orderedGroups = [...groups.entries()]
    .sort(([a], [b]) => stableRank(a).localeCompare(stableRank(b)))
    .map(([, group]) => [...group].sort((a, b) => stableRank(a.id).localeCompare(stableRank(b.id))));
  const selected = [];
  for (let round = 0; selected.length < size; round++) {
    let added = 0;
    for (const group of orderedGroups) {
      if (selected.length === size) break;
      if (round < group.length) {
        selected.push(group[round]);
        added++;
      }
    }
    if (added === 0) break;
  }
  return selected;
}

export function sampleSuperGpqaStratifiedRows(
  rows,
  size,
  seed = SUPERGPQA_EVALUATION_SAMPLE_SEED,
) {
  if (!Array.isArray(rows)) throw new TypeError("SuperGPQA rows must be an array.");
  if (!Number.isSafeInteger(size) || size < 1 || size > rows.length) {
    throw new TypeError("SuperGPQA sample size must be a positive integer no larger than the dataset.");
  }
  const groups = new Map();
  for (const row of rows) {
    const key = `${row.metadata?.discipline}\0${row.metadata?.difficulty}`;
    const group = groups.get(key) ?? [];
    group.push(row);
    groups.set(key, group);
  }
  const allocations = [...groups.entries()].map(([key, group]) => {
    const exact = (size * group.length) / rows.length;
    return {
      key,
      group: [...group].sort((a, b) => stableRank(a.id, seed).localeCompare(stableRank(b.id, seed))),
      count: Math.floor(exact),
      remainder: exact - Math.floor(exact),
    };
  });
  let remaining = size - allocations.reduce((total, allocation) => total + allocation.count, 0);
  allocations.sort((a, b) => b.remainder - a.remainder
    || stableRank(a.key, seed).localeCompare(stableRank(b.key, seed)));
  for (let index = 0; index < remaining; index++) allocations[index].count++;
  return allocations
    .flatMap(({ group, count }) => group.slice(0, count))
    .sort((a, b) => stableRank(a.id, seed).localeCompare(stableRank(b.id, seed)));
}

export function sampleSuperGpqaEvaluationRows(rows, size, compatibilitySampleSize = 100) {
  if (!Array.isArray(rows)) throw new TypeError("SuperGPQA rows must be an array.");
  if (!Number.isSafeInteger(compatibilitySampleSize)
    || compatibilitySampleSize < 1 || compatibilitySampleSize >= rows.length) {
    throw new TypeError(
      "SuperGPQA compatibility sample size must be a positive integer smaller than the dataset.",
    );
  }
  const compatibilityIds = new Set(
    sampleSuperGpqaCompatibilityRows(rows, compatibilitySampleSize).map(({ id }) => id),
  );
  const evaluationPool = rows.filter(({ id }) => !compatibilityIds.has(id));
  return sampleSuperGpqaStratifiedRows(
    evaluationPool,
    size,
    SUPERGPQA_EVALUATION_SAMPLE_SEED,
  );
}

async function sha256(path) {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest("hex");
}

export async function prepareSuperGpqaFile({ input, output, audit }) {
  if (existsSync(output)) throw new Error(`Refusing to overwrite existing output: ${output}`);
  if (existsSync(audit)) throw new Error(`Refusing to overwrite existing audit: ${audit}`);
  const inputSha256 = await sha256(input);
  if (inputSha256 !== SUPERGPQA_SHA256) {
    throw new Error(`Unexpected SuperGPQA SHA-256: ${inputSha256}. Expected ${SUPERGPQA_SHA256}.`);
  }
  const text = readFileSync(input, "utf8").trim();
  const sourceRows = text.length === 0 ? [] : text.split(/\r?\n/).map((line) => JSON.parse(line));
  const rows = prepareSuperGpqaRows(sourceRows);
  if (rows.length !== SUPERGPQA_ROWS) {
    throw new Error(`Expected ${SUPERGPQA_ROWS} SuperGPQA rows, received ${rows.length}.`);
  }
  const counts = {};
  for (const row of rows) {
    const key = `${row.metadata.discipline} / ${row.metadata.difficulty}`;
    counts[key] = (counts[key] ?? 0) + 1;
  }
  const report = {
    dataset: {
      source: "https://huggingface.co/datasets/m-a-p/SuperGPQA",
      revision: SUPERGPQA_REVISION,
      path: input,
      sha256: inputSha256,
    },
    summary: { total: rows.length, strata: counts },
  };
  mkdirSync(dirname(output), { recursive: true });
  mkdirSync(dirname(audit), { recursive: true });
  writeFileSync(output, `${rows.map((row) => JSON.stringify(row)).join("\n")}\n`);
  writeFileSync(audit, `${JSON.stringify(report, null, 2)}\n`);
  return Object.freeze(report);
}

async function main() {
  const input = option("--input", "benchmarks/.data/supergpqa/source/SuperGPQA-all.jsonl");
  const output = option("--output", "benchmarks/.data/supergpqa/supergpqa.jsonl");
  const audit = option("--audit", "benchmarks/.data/supergpqa/supergpqa.audit.json");
  console.log(JSON.stringify(await prepareSuperGpqaFile({ input, output, audit }), null, 2));
}

const entry = process.argv[1] === undefined ? undefined : pathToFileURL(process.argv[1]).href;
if (entry === import.meta.url) await main();
