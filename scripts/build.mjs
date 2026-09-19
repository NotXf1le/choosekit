import { mkdirSync, rmSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../", import.meta.url));
const tsc = createRequire(import.meta.url).resolve("typescript/bin/tsc");
rmSync(new URL("../dist/", import.meta.url), { recursive: true, force: true });
for (const [directory, module, resolution, type] of [
  ["esm", "ES2022", "Bundler", "module"],
  ["cjs", "CommonJS", "Node10", "commonjs"],
]) {
  const result = spawnSync(
    process.execPath,
    [tsc, "-p", "tsconfig.json", "--module", module, "--moduleResolution", resolution,
      "--outDir", `dist/${directory}`],
    { cwd: root, stdio: "inherit" },
  );
  if (result.error) throw result.error;
  if (result.status !== 0) process.exit(result.status ?? 1);
  mkdirSync(new URL(`../dist/${directory}/`, import.meta.url), { recursive: true });
  writeFileSync(new URL(`../dist/${directory}/package.json`, import.meta.url),
    JSON.stringify({ type }) + "\n");
}
