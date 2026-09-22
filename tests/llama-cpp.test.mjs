import assert from "node:assert/strict";
import { test } from "node:test";
import { fromLlamaCpp } from "../dist/esm/llama-cpp.js";
import { ScoringError } from "../dist/esm/index.js";
import { encode } from "./helpers.mjs";

const choiceRequest = Object.freeze({
  context: "A production change is pending.",
  question: "What should happen next?",
  choices: Object.freeze({ wait: "Wait for approval.", deploy: "Deploy immediately." }),
});

const decode = (tokens) => new TextDecoder().decode(
  Uint8Array.from(tokens, (tokenId) => tokenId - 1),
);

function fixture({ tokenizer = encode, detokenizer = decode, logprob = () => -1,
  topTokenIds = (target) => [target], transform,
  props = { modalities: { vision: true }, media_marker: "<__media_test__>" } } = {}) {
  const calls = [];
  const fetch = async (url, init) => {
    const parsedURL = new URL(url);
    const path = parsedURL.pathname;
    const body = init.body === undefined ? undefined : JSON.parse(init.body);
    calls.push({ url: parsedURL, path, body, headers: init.headers, signal: init.signal });
    if (path === "/props") return json(props);
    if (path === "/tokenize") return json({ tokens: tokenizer(body.content, body.add_special) });
    if (path === "/detokenize") return json({ content: detokenizer(body.tokens) });
    if (path !== "/completion") return json({}, 404);

    const target = body.logit_bias[0][0];
    const value = {
      tokens: [target],
      completion_probabilities: [{
        id: target,
        logprob: logprob(target),
        top_logprobs: topTokenIds(target, body).map((id) => ({ id, logprob: logprob(id) })),
      }],
      generation_settings: { post_sampling_probs: false, backend_sampling: false },
      tokens_evaluated: Array.isArray(body.prompt) ? body.prompt.length : 123,
      tokens_predicted: 1,
      timings: { cache_n: 0 },
    };
    return json(transform ? transform(value, body) : value);
  };
  return { calls, fetch };
}

function json(value, status = 200) {
  return new Response(JSON.stringify(value), {
    status, headers: { "content-type": "application/json" },
  });
}

const completions = (f) => f.calls.filter((call) => call.path === "/completion");

test("maps label scores back to choice keys and reuses sibling logprobs", async () => {
  const a = encode("A")[0];
  const b = encode("B")[0];
  const scores = new Map([[a, -2], [b, -0.25]]);
  const f = fixture({
    logprob: (id) => scores.get(id) ?? -10,
    topTokenIds: () => [a, b],
  });
  const choose = fromLlamaCpp({
    baseURL: "http://localhost:8080/v1", model: "local-model", fetch: f.fetch,
  });

  const decision = await choose(choiceRequest);

  assert.equal(decision.choice, "deploy");
  assert.deepEqual(decision.scores, { wait: -2, deploy: -0.25 });
  assert.equal(decision.usage.requests, f.calls.length);
  assert.equal(completions(f).length, 1);
  assert.equal(completions(f)[0].body.model, "local-model");
});

test("scores image labels through native multimodal completion", async () => {
  const a = encode("A")[0];
  const b = encode("B")[0];
  const scores = new Map([[a, -0.2], [b, -1.5]]);
  const f = fixture({
    logprob: (id) => scores.get(id) ?? -10,
    topTokenIds: (target) => [target],
  });
  const controller = new AbortController();
  const choose = fromLlamaCpp({
    baseURL: "http://localhost:8080/v1",
    model: "vision/model",
    headers: { authorization: "Bearer test" },
    fetch: f.fetch,
  });

  const decision = await choose({
    ...choiceRequest,
    images: [
      { mediaType: "image/png", base64: "Zmlyc3Q=" },
      { mediaType: "image/jpeg", base64: "c2Vjb25k" },
    ],
    signal: controller.signal,
  });

  assert.equal(decision.choice, "wait");
  assert.deepEqual(decision.scores, { wait: -0.2, deploy: -1.5 });
  assert.equal(decision.usage.promptTokens, 246);
  assert.equal(decision.usage.requests, f.calls.length);
  const propsCall = f.calls.find((call) => call.path === "/props");
  assert.equal(propsCall.url.searchParams.get("model"), "vision/model");
  assert.equal(propsCall.signal, controller.signal);
  assert.equal(propsCall.headers.authorization, "Bearer test");
  assert.equal(completions(f).length, 2);
  const detokenizeCall = f.calls.find((call) => call.path === "/detokenize");
  const promptString = `<__media_test__>\n<__media_test__>\n${decode(detokenizeCall.body.tokens)}`;
  for (const call of completions(f)) {
    assert.deepEqual(call.body.prompt, {
      prompt_string: promptString,
      multimodal_data: ["Zmlyc3Q=", "c2Vjb25k"],
    });
    assert.equal(call.body.model, "vision/model");
    assert.equal(call.body.post_sampling_probs, false);
    assert.equal(call.body.backend_sampling, false);
    assert.equal(call.signal, controller.signal);
  }
  assert.equal(completions(f)[1].body.n_probs, 1);
});

test("preserves a tokenization-boundary rollback for image labels", async () => {
  const pairs = (text) => {
    const bytes = new TextEncoder().encode(text);
    const ids = [];
    for (let i = 0; i < bytes.length; i += 2) {
      ids.push(i + 1 < bytes.length ? 257 + bytes[i] * 256 + bytes[i + 1] : bytes[i] + 1);
    }
    return ids;
  };
  const a = pairs("abcA").at(-1);
  const b = pairs("abcB").at(-1);
  const scores = new Map([[a, -0.1], [b, -1.1]]);
  const f = fixture({
    tokenizer: pairs,
    detokenizer: (tokens) => {
      assert.deepEqual(tokens, pairs("ab"));
      return "ab";
    },
    logprob: (id) => scores.get(id) ?? -10,
    topTokenIds: () => [a, b],
  });
  const choose = fromLlamaCpp({
    baseURL: "http://localhost:8080",
    fetch: f.fetch,
    formatPrompt: ({ context }) => context,
  });

  const decision = await choose({
    context: "abc",
    question: "Which?",
    choices: { first: "A", second: "B" },
    images: [{ mediaType: "image/png", base64: "aW1hZ2U=" }],
  });

  assert.equal(decision.choice, "first");
  assert.equal(decision.boundaryTokens, 1);
  assert.equal(completions(f)[0].body.prompt.prompt_string, "<__media_test__>\nab");
});

test("scores image labels that share an initial token", async () => {
  const tokenizations = new Map([
    ["p", [1]],
    ["pA", [1, 100, 101]],
    ["pB", [1, 100, 102]],
    ["pC", [1, 200]],
    ["px", [1, 100]],
  ]);
  const textByTokens = new Map([["1", "p"], ["1,100", "px"]]);
  const scores = new Map([[100, -0.2], [200, -2], [101, -0.3], [102, -1]]);
  const f = fixture({
    tokenizer: (text) => tokenizations.get(text),
    detokenizer: (tokens) => textByTokens.get(tokens.join(",")),
    logprob: (id) => scores.get(id) ?? -10,
  });
  const choose = fromLlamaCpp({
    baseURL: "http://localhost:8080",
    fetch: f.fetch,
    formatPrompt: ({ context }) => context,
  });

  const decision = await choose({
    context: "p",
    question: "Which?",
    choices: { first: "First", second: "Second", third: "Third" },
    images: [{ mediaType: "image/png", base64: "aW1hZ2U=" }],
  });

  assert.equal(decision.choice, "first");
  assert.deepEqual(decision.scores, { first: -0.5, second: -1.2, third: -2 });
  assert.deepEqual(
    [...new Set(completions(f).map((call) => call.body.prompt.prompt_string))],
    ["<__media_test__>\np", "<__media_test__>\npx"],
  );
});

test("rejects an image prefix that does not survive a tokenization round trip", async () => {
  const f = fixture({ detokenizer: () => "different text" });
  const choose = fromLlamaCpp({ baseURL: "http://localhost:8080", fetch: f.fetch });

  await assert.rejects(choose({
    ...choiceRequest,
    images: [{ mediaType: "image/png", base64: "aW1hZ2U=" }],
  }), /could not preserve the image prompt token prefix/i);
  assert.equal(completions(f).length, 0);
});

test("rejects a formatted prompt containing llama.cpp's media marker", async () => {
  const marker = "<__media_test__>";
  const f = fixture({ detokenizer: () => `prompt ${marker}` });
  const choose = fromLlamaCpp({ baseURL: "http://localhost:8080", fetch: f.fetch });

  await assert.rejects(choose({
    ...choiceRequest,
    images: [{ mediaType: "image/png", base64: "aW1hZ2U=" }],
  }), /formatted prompt contains llama\.cpp's multimodal media marker/i);
  assert.equal(completions(f).length, 0);
});

test("rejects image inputs when llama.cpp does not advertise vision", async () => {
  const f = fixture({ props: { modalities: { vision: false }, media_marker: "<marker>" } });
  const choose = fromLlamaCpp({ baseURL: "http://localhost:8080", fetch: f.fetch });

  await assert.rejects(choose({
    ...choiceRequest,
    images: [{ mediaType: "image/png", base64: "aW1hZ2U=" }],
  }), /does not advertise vision support/i);
  assert.deepEqual(f.calls.map((call) => call.path), ["/props"]);
});

test("rejects image inputs when llama.cpp omits the media marker", async () => {
  const f = fixture({ props: { modalities: { vision: true } } });
  const choose = fromLlamaCpp({ baseURL: "http://localhost:8080", fetch: f.fetch });

  await assert.rejects(choose({
    ...choiceRequest,
    images: [{ mediaType: "image/png", base64: "aW1hZ2U=" }],
  }), /multimodal media marker/i);
  assert.deepEqual(f.calls.map((call) => call.path), ["/props"]);
});

test("rejects an invalid llama.cpp detokenization response", async () => {
  const f = fixture({ detokenizer: () => 42 });
  const choose = fromLlamaCpp({ baseURL: "http://localhost:8080", fetch: f.fetch });

  await assert.rejects(choose({
    ...choiceRequest,
    images: [{ mediaType: "image/png", base64: "aW1hZ2U=" }],
  }), /invalid detokenization/i);
  assert.equal(completions(f).length, 0);
});

test("rejects image inputs in minimal-prefix mode before contacting llama.cpp", async () => {
  const f = fixture();
  const choose = fromLlamaCpp({
    baseURL: "http://localhost:8080", mode: "minimal-prefix", fetch: f.fetch,
  });

  await assert.rejects(choose({
    ...choiceRequest,
    images: [{ mediaType: "image/png", base64: "aW1hZ2U=" }],
  }), /require labels mode/i);
  assert.equal(f.calls.length, 0);
});

test("rejects special-token insertion for llama.cpp image inputs", async () => {
  const f = fixture();
  const choose = fromLlamaCpp({
    baseURL: "http://localhost:8080", addSpecialTokens: true, fetch: f.fetch,
  });

  await assert.rejects(choose({
    ...choiceRequest,
    images: [{ mediaType: "image/png", base64: "aW1hZ2U=" }],
  }), /require addSpecialTokens to be false/i);
  assert.equal(f.calls.length, 0);
});

test("scores candidates separately when top logprobs contain none of them", async () => {
  const scores = new Map([[encode("A")[0], -0.3], [encode("B")[0], -1.4]]);
  const f = fixture({
    logprob: (id) => scores.get(id) ?? -10,
    topTokenIds: () => [999],
  });

  const decision = await fromLlamaCpp({ baseURL: "http://localhost:8080", fetch: f.fetch })(choiceRequest);

  assert.equal(decision.choice, "wait");
  assert.deepEqual(decision.scores, { wait: -0.3, deploy: -1.4 });
  assert.equal(completions(f).length, 2);
  assert.equal(completions(f)[1].body.n_probs, 1);
  assert.equal(decision.usage.requests, f.calls.length);
});

test("minimal-prefix scores only the first token that distinguishes shared keys", async () => {
  const quote = encode('"')[0];
  const space = encode(" ")[0];
  const scores = new Map([[quote, -0.1], [space, -1.2]]);
  const f = fixture({
    logprob: (id) => scores.get(id) ?? -10,
    topTokenIds: () => [quote, space],
  });
  const choose = fromLlamaCpp({
    baseURL: "http://localhost:8080", mode: "minimal-prefix", fetch: f.fetch,
  });

  const decision = await choose({
    context: "Choose a product.",
    question: "Which product?",
    choices: { watermelon: "The fruit.", "watermelon juice": "A drink." },
  });

  assert.equal(decision.choice, "watermelon");
  assert.deepEqual(decision.scores, { watermelon: -0.1, "watermelon juice": -1.2 });
  assert.equal(completions(f).length, 1);
  assert.deepEqual(completions(f)[0].body.prompt.slice(-encode('"watermelon').length),
    encode('"watermelon'));
});

test("continues scoring a group that separates at a deeper branch", async () => {
  const a = encode("a")[0];
  const b = encode("b")[0];
  const scores = new Map([[a, -0.2], [b, -1]]);
  const f = fixture({
    logprob: (id) => scores.get(id) ?? -10,
    topTokenIds: () => [a, b],
  });
  const choose = fromLlamaCpp({
    baseURL: "http://localhost:8080", mode: "minimal-prefix", fetch: f.fetch,
  });

  const decision = await choose({
    context: "Choose a key.",
    question: "Which key?",
    choices: { a: "A", ba: "BA", bb: "BB" },
  });

  assert.deepEqual(decision.scores, { a: -0.2, ba: -1.2, bb: -2 });
  assert.equal(completions(f).length, 2);
  assert.equal(completions(f)[1].body.prompt.length,
    completions(f)[0].body.prompt.length + 1);
});

test("reports a tokenization-boundary rollback", async () => {
  const pairs = (text, special = false) => {
    const bytes = new TextEncoder().encode(text);
    const ids = special ? [0] : [];
    for (let i = 0; i < bytes.length; i += 2) {
      ids.push(i + 1 < bytes.length ? 257 + bytes[i] * 256 + bytes[i + 1] : bytes[i] + 1);
    }
    return ids;
  };
  const a = pairs("abcA").at(-1);
  const b = pairs("abcB").at(-1);
  const scores = new Map([[a, -0.1], [b, -1.1]]);
  const f = fixture({
    tokenizer: pairs,
    logprob: (id) => scores.get(id) ?? -10,
    topTokenIds: () => [a, b],
  });
  const choose = fromLlamaCpp({
    baseURL: "http://localhost:8080", fetch: f.fetch,
    formatPrompt: ({ context }) => context,
  });

  const decision = await choose({
    context: "abc", question: "Which?", choices: { first: "A", second: "B" },
  });

  assert.equal(decision.choice, "first");
  assert.equal(decision.boundaryTokens, 1);
  assert.equal(completions(f)[0].body.prompt.length, 1);
});

test("rejects a probability attached to a different token", async () => {
  const f = fixture({ transform: (value) => ({
    ...value,
    completion_probabilities: [{ ...value.completion_probabilities[0], id: 999 }],
  }) });
  const choose = fromLlamaCpp({ baseURL: "http://localhost:8080", fetch: f.fetch });

  await assert.rejects(choose(choiceRequest),
    (error) => error instanceof ScoringError && /different token/.test(error.message));
});

test("rejects conflicting logprobs for the forced token", async () => {
  const f = fixture({ transform: (value) => {
    const forced = value.completion_probabilities[0];
    return {
      ...value,
      completion_probabilities: [{
        ...forced,
        top_logprobs: forced.top_logprobs.map((entry) => entry.id === forced.id
          ? { ...entry, logprob: entry.logprob - 0.5 }
          : entry),
      }],
    };
  } });
  const choose = fromLlamaCpp({ baseURL: "http://localhost:8080", fetch: f.fetch });

  await assert.rejects(choose(choiceRequest),
    (error) => error instanceof ScoringError && /conflicting logprobs/.test(error.message));
});
