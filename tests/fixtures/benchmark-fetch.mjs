import { existsSync } from "node:fs";

function json(value, status = 200) {
  return new Response(JSON.stringify(value), {
    status,
    headers: { "content-type": "application/json" },
  });
}

function tokens(text, special = false) {
  const ids = [...new TextEncoder().encode(text)].map((byte) => byte + 1);
  return special ? [0, ...ids] : ids;
}

globalThis.fetch = async (url, init = {}) => {
  const expectedParent = process.env.CHOOSEKIT_EXPECT_OUTPUT_PARENT;
  if (expectedParent && !existsSync(expectedParent)) {
    throw new Error("Benchmark output directory was not created before the first request.");
  }

  const body = JSON.parse(init.body);
  const pathname = new URL(url).pathname;
  if (pathname === "/tokenize") {
    return json({ tokens: tokens(body.content, body.add_special) });
  }
  if (pathname === "/completion") {
    const target = body.logit_bias[0][0];
    const optionTokens = ["A", "B", "C", "D"].map((value) => tokens(value)[0]);
    const candidates = [...new Set([target, ...optionTokens])];
    return json({
      tokens: [target],
      completion_probabilities: [{
        id: target,
        logprob: target === optionTokens[0] ? -0.1 : -1,
        top_logprobs: candidates.map((id) => ({
          id,
          logprob: id === optionTokens[0] ? -0.1 : -1,
        })),
      }],
      generation_settings: { post_sampling_probs: false, backend_sampling: false },
      tokens_evaluated: body.prompt.length,
      tokens_predicted: 1,
      timings: { cache_n: 0 },
    });
  }
  if (url === "https://openrouter.ai/api/alpha/decisions") {
    const optionIds = Object.keys(body.questions.decision.criteria);
    const probability = 1 / optionIds.length;
    return json({
      answers: {
        decision: {
          choice: optionIds[0],
          probabilities: Object.fromEntries(optionIds.map((id) => [id, probability])),
          confidence: probability,
        },
      },
      model: body.model,
      provider: "test",
      usage: { input_tokens: 1, output_tokens: 1, cost: 0 },
    });
  }
  return json({}, 404);
};
