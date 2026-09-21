import assert from "node:assert/strict";
import test from "node:test";
import {
  prepareSuperGpqaRows,
  sampleSuperGpqaPilotRows,
  sampleSuperGpqaEvaluationRows,
  sampleSuperGpqaStratifiedRows,
} from "../benchmarks/prepare-supergpqa.mjs";

function row(overrides = {}) {
  return {
    uuid: "case-1",
    question: "Which answer is correct?",
    options: ["First", "Second"],
    answer: "Second",
    answer_letter: "B",
    discipline: "Science",
    field: "Physics",
    subfield: "Mechanics",
    difficulty: "middle",
    is_calculation: false,
    ...overrides,
  };
}

test("prepares choices and validates the answer letter against its text", () => {
  const [prepared] = prepareSuperGpqaRows([row()]);
  assert.deepEqual(prepared.choices, { A: "First", B: "Second" });
  assert.equal(prepared.gold, "B");
  assert.deepEqual(prepared.metadata, {
    discipline: "Science",
    field: "Physics",
    subfield: "Mechanics",
    difficulty: "middle",
    isCalculation: false,
  });
  assert.throws(() => prepareSuperGpqaRows([row({ answer: "First" })]), /inconsistent answer/);
});

test("samples discipline and difficulty strata deterministically", () => {
  const source = [];
  for (const discipline of ["Science", "Law"]) {
    for (const difficulty of ["easy", "hard"]) {
      for (let index = 0; index < 3; index++) {
        source.push(row({
          uuid: `${discipline}-${difficulty}-${index}`,
          discipline,
          difficulty,
        }));
      }
    }
  }
  const prepared = prepareSuperGpqaRows(source);
  const sample = sampleSuperGpqaPilotRows(prepared, 8);
  const counts = sample.reduce((map, item) => {
    const key = `${item.metadata.discipline}/${item.metadata.difficulty}`;
    map[key] = (map[key] ?? 0) + 1;
    return map;
  }, {});
  assert.deepEqual(Object.values(counts).sort(), [2, 2, 2, 2]);
  assert.deepEqual(
    sample.map(({ id }) => id),
    sampleSuperGpqaPilotRows([...prepared].reverse(), 8).map(({ id }) => id),
  );
});

test("rejects malformed rows and sample sizes", () => {
  assert.throws(() => prepareSuperGpqaRows([row(), row()]), /duplicate row ID/);
  assert.throws(() => prepareSuperGpqaRows([row({ options: ["First"] })]), /invalid options/);
  assert.throws(() => prepareSuperGpqaRows([row({ answer_letter: "C" })]), /outside its options/);
  assert.throws(
    () => sampleSuperGpqaPilotRows(prepareSuperGpqaRows([row()]), 2),
    /sample size/,
  );
});

test("samples discipline and difficulty strata proportionally and deterministically", () => {
  const source = [];
  for (let index = 0; index < 2; index++) {
    source.push(row({ uuid: `Law-hard-${index}`, discipline: "Law", difficulty: "hard" }));
  }
  for (let index = 0; index < 8; index++) {
    source.push(row({ uuid: `Science-easy-${index}`, discipline: "Science", difficulty: "easy" }));
  }
  const prepared = prepareSuperGpqaRows(source);
  const sample = sampleSuperGpqaStratifiedRows(prepared, 5);
  assert.equal(sample.filter(({ metadata }) => metadata.discipline === "Law").length, 1);
  assert.equal(sample.filter(({ metadata }) => metadata.discipline === "Science").length, 4);
  assert.deepEqual(
    sample.map(({ id }) => id),
    sampleSuperGpqaStratifiedRows([...prepared].reverse(), 5).map(({ id }) => id),
  );
});

test("keeps the evaluation sample disjoint from the pilot sample", () => {
  const source = [];
  for (const discipline of ["Science", "Law"]) {
    for (let index = 0; index < 10; index++) {
      source.push(row({ uuid: `${discipline}-${index}`, discipline }));
    }
  }
  const prepared = prepareSuperGpqaRows(source);
  const pilotIds = new Set(
    sampleSuperGpqaPilotRows(prepared, 4).map(({ id }) => id),
  );
  const evaluation = sampleSuperGpqaEvaluationRows(prepared, 10, 4);
  assert.equal(evaluation.some(({ id }) => pilotIds.has(id)), false);
  assert.deepEqual(
    evaluation.map(({ id }) => id),
    sampleSuperGpqaEvaluationRows([...prepared].reverse(), 10, 4).map(({ id }) => id),
  );
});
