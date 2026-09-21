import { existsSync } from "node:fs";

/** The llama.cpp benchmark lists /v1/models before inference; everything else fails on purpose. */
function servedModels() {
  const ids = (process.env.CHOOSEKIT_SERVED_MODELS ?? "qwen3.8-27b-text-64k")
    .split(",").filter((id) => id.length > 0);
  return new Response(JSON.stringify({ object: "list", data: ids.map((id) => ({ id, object: "model" })) }), {
    status: 200,
    headers: { "content-type": "application/json" },
  });
}

globalThis.fetch = async (input) => {
  const url = typeof input === "string" ? input : input.url;
  if (new URL(url).pathname.endsWith("/v1/models")) {
    if (process.env.CHOOSEKIT_MODELS_STATUS) {
      return new Response("model list unavailable", { status: Number(process.env.CHOOSEKIT_MODELS_STATUS) });
    }
    return servedModels();
  }
  const expectedParent = process.env.CHOOSEKIT_EXPECT_OUTPUT_PARENT;
  if (!expectedParent || !existsSync(expectedParent)) {
    throw new Error("Benchmark output directory was not created before the first request.");
  }
  throw new Error("Intentional benchmark test failure.");
};
