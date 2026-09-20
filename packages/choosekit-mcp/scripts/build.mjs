import { rmSync } from "node:fs";
import { createRequire } from "node:module";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const packageRoot = fileURLToPath(new URL("../", import.meta.url));
const tsc = createRequire(import.meta.url).resolve("typescript/bin/tsc");

rmSync(new URL("../dist/", import.meta.url), { recursive: true, force: true });

const result = spawnSync(process.execPath, [tsc, "-p", "tsconfig.json"], {
  cwd: packageRoot,
  stdio: "inherit",
});

if (result.error) throw result.error;
if (result.status !== 0) process.exit(result.status ?? 1);
