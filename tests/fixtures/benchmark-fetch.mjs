import { existsSync } from "node:fs";

globalThis.fetch = async () => {
  const expectedParent = process.env.CHOOSEKIT_EXPECT_OUTPUT_PARENT;
  if (!expectedParent || !existsSync(expectedParent)) {
    throw new Error("Benchmark output directory was not created before the first request.");
  }
  throw new Error("Intentional benchmark test failure.");
};
