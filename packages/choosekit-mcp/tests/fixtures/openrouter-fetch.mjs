import assert from "node:assert/strict";

globalThis.fetch = async (url, init) => {
  assert.equal(url, "https://openrouter.ai/api/v1/chat/completions");
  assert.equal(new Headers(init.headers).get("authorization"),
    "Bearer openrouter-secret-that-must-not-leak");
  const body = JSON.parse(init.body);
  assert.equal(body.model, "fixture/model");
  assert.deepEqual(body.provider, {
    only: ["fixture-provider"],
    allow_fallbacks: false,
  });

  return new Response(JSON.stringify({
    choices: [{
      finish_reason: "length",
      message: { role: "assistant", content: "B" },
      logprobs: {
        content: [{
          token: "B",
          bytes: [66],
          logprob: -0.1,
          top_logprobs: [
            { token: "A", bytes: [65], logprob: -1.1 },
            { token: "B", bytes: [66], logprob: -0.1 },
          ],
        }],
      },
    }],
  }), {
    status: 200,
    headers: { "content-type": "application/json" },
  });
};
