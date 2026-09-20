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
    env: { ...process.env, CHOOSEKIT_BASE_URL: "" },
    timeout: 10_000,
  });
  assert.equal(launched.error, undefined, launched.error?.message);
  assert.equal(launched.status, 1);
  assert.equal(launched.stdout, "");
  assert.match(launched.stderr, /^choosekit-mcp:/);
});
