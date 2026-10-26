import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { test } from "node:test";
import { ScoringError } from "../dist/esm/index.js";
import { fromOpenRouter } from "../dist/esm/openrouter.js";
import { fromOllama } from "../dist/esm/ollama.js";
import { fromLlamaCpp } from "../dist/esm/llama-cpp.js";
import { encode, request } from "./helpers.mjs";

const { ScoringError: CjsScoringError } = createRequire(import.meta.url)("choosekit");

for (const [format, ErrorClass] of [["ESM", ScoringError], ["CJS", CjsScoringError]]) {
  test(`${format} ScoringError preserves Error construction and optional HTTP metadata`, () => {
    const cause = new Error("upstream");
    const empty = new ErrorClass();
    assert.ok(empty instanceof Error);
    assert.ok(empty instanceof ErrorClass);
    assert.equal(empty.name, "ScoringError");
    assert.equal(empty.message, "");
    assert.equal(empty.httpStatus, undefined);

    const legacy = new ErrorClass("legacy", { cause });
    assert.equal(legacy.message, "legacy");
    assert.equal(legacy.cause, cause);
    assert.equal(legacy.httpStatus, undefined);

    const httpError = new ErrorClass(undefined, { cause, httpStatus: 429 });
    assert.ok(httpError instanceof ErrorClass);
    assert.equal(httpError.message, "");
    assert.equal(httpError.cause, cause);
    assert.equal(httpError.httpStatus, 429);
  });
}

const httpCases = [
  ["OpenRouter POST", (fetch) => fromOpenRouter({ apiKey: "fixture-key", model: "fixture/model", fetch }), request],
  ["Ollama POST", (fetch) => fromOllama({ model: "fixture-model", fetch }), request],
  ["llama.cpp POST", (fetch) => fromLlamaCpp({ baseURL: "http://fixture.test", fetch }), request],
  ["llama.cpp GET", (fetch) => fromLlamaCpp({ baseURL: "http://fixture.test", fetch }),
    { ...request, images: [{ mediaType: "image/png", base64: "cG5n" }] }],
];

function createFetchFixture(boundary, failure) {
  return async (url, init) => {
    if (boundary === "llama.cpp GET" && new URL(url).pathname === "/tokenize") {
      const { content } = JSON.parse(init.body);
      return Response.json({
        tokens: Array.isArray(content)
          ? content.flatMap((part) => part === -1 ? [-1] : encode(part))
          : encode(content),
      });
    }
    return failure();
  };
}

for (const [boundary, create, input] of httpCases) {
  test(`${boundary} preserves the final failing HTTP status`, async () => {
    for (const status of [302, 401, 403, 429, 503]) {
      const choose = create(createFetchFixture(boundary,
        () => Response.json({ error: "secret upstream details" }, { status })));
      await assert.rejects(choose(input), (error) => {
        assert.ok(error instanceof ScoringError);
        assert.equal(error.httpStatus, status);
        assert.doesNotMatch(error.message, /secret/);
        return true;
      });
    }
  });

  test(`${boundary} does not label invalid JSON as an HTTP failure`, async () => {
    const choose = create(createFetchFixture(boundary, () => new Response("{")));
    await assert.rejects(choose(input), (error) => {
      assert.ok(error instanceof ScoringError);
      assert.equal(error.httpStatus, undefined);
      return true;
    });
  });
}
