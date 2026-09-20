import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

const packageRoot = fileURLToPath(new URL("../", import.meta.url));
const npmCli = process.platform === "win32"
  ? join(dirname(process.execPath), "node_modules", "npm", "bin", "npm-cli.js")
  : undefined;

function llamaEnvironment() {
  const env = { ...process.env };
  for (const key of [
    "CHOOSEKIT_BACKEND",
    "CHOOSEKIT_BASE_URL",
    "CHOOSEKIT_MODEL",
    "CHOOSEKIT_MODE",
    "OPENROUTER_API_KEY",
    "OPENROUTER_PROVIDER",
  ]) delete env[key];
  return { ...env, CHOOSEKIT_BACKEND: "llama-cpp", CHOOSEKIT_BASE_URL: "" };
}

function spawnNpm(args, options = {}) {
  return npmCli
    ? spawnSync(process.execPath, [npmCli, ...args], { encoding: "utf8", ...options })
    : spawnSync("npm", args, { encoding: "utf8", ...options });
}

function runNpm(args, options) {
  const result = spawnNpm(args, { timeout: 120_000, ...options });
  assert.equal(result.error, undefined, result.error?.message);
  assert.equal(result.status, 0, result.stderr || result.stdout);
  return result;
}

test("the packed package installs and exposes its executable", (t) => {
  const temporary = mkdtempSync(join(tmpdir(), "choosekit-mcp-package-"));
  t.after(() => rmSync(temporary, { recursive: true, force: true }));

  const packed = runNpm(["pack", "--json", "--pack-destination", temporary], {
    cwd: packageRoot,
  });
  const [{ filename }] = JSON.parse(packed.stdout);
  const tarball = join(temporary, filename);
  runNpm(["init", "--yes"], { cwd: temporary });
  runNpm([
    "install", tarball, "--ignore-scripts", "--no-audit", "--no-fund",
  ], { cwd: temporary });

  const launched = spawnNpm(["exec", "--no", "--", "choosekit-mcp"], {
    cwd: temporary,
    env: llamaEnvironment(),
    timeout: 10_000,
  });
  assert.equal(launched.error, undefined, launched.error?.message);
  assert.equal(launched.status, 1);
  assert.equal(launched.stdout, "");
  assert.match(launched.stderr, /^choosekit-mcp:/);
});
