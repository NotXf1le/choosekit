import assert from "node:assert/strict";
import { mkdir, mkdtemp, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { createImageLoader } from "../dist/images.js";

const PNG = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
const JPEG = Buffer.from([0xff, 0xd8, 0xff, 0x00]);
const WEBP = Buffer.from([0x52, 0x49, 0x46, 0x46, 0, 0, 0, 0, 0x57, 0x45, 0x42, 0x50]);

test("loads supported images in path order", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "choosekit-images-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  await writeFile(join(root, "first.bin"), WEBP);
  await writeFile(join(root, "second.bin"), PNG);
  await writeFile(join(root, "third.bin"), JPEG);

  const images = await createImageLoader(root)(["first.bin", "second.bin", "third.bin"]);

  assert.deepEqual(images.map(({ mediaType }) => mediaType),
    ["image/webp", "image/png", "image/jpeg"]);
  assert.deepEqual(images.map(({ base64 }) => base64),
    [WEBP, PNG, JPEG].map((data) => data.toString("base64")));
});

test("rejects paths outside the image root, including symlink escapes", async (t) => {
  const parent = await mkdtemp(join(tmpdir(), "choosekit-images-"));
  t.after(() => rm(parent, { recursive: true, force: true }));
  const root = join(parent, "root");
  const outside = join(parent, "outside");
  await mkdir(root);
  await mkdir(outside);
  await writeFile(join(outside, "image.png"), PNG);
  await symlink(outside, join(root, "escape"), process.platform === "win32" ? "junction" : "dir");
  const load = createImageLoader(root);

  await assert.rejects(load([join("..", "outside", "image.png")]), /outside/i);
  await assert.rejects(load([join("escape", "image.png")]), /outside/i);
});

test("rejects directories and unsupported file contents", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "choosekit-images-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  await mkdir(join(root, "directory"));
  await writeFile(join(root, "not-an-image.png"), "plain text");
  const load = createImageLoader(root);

  await assert.rejects(load(["directory"]), /regular file/i);
  await assert.rejects(load(["not-an-image.png"]), /PNG, JPEG, or WebP/i);
});
