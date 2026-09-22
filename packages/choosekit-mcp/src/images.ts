import type { ImageInput, ImageMediaType } from "choosekit";
import { readFile, realpath, stat } from "node:fs/promises";
import { isAbsolute, relative, resolve, sep } from "node:path";

export type ImageLoader = (
  paths: readonly string[],
  signal?: AbortSignal,
) => Promise<readonly ImageInput[]>;

export class ImageLoadError extends Error {
  override readonly name = "ImageLoadError";
}

function isInside(root: string, path: string): boolean {
  const fromRoot = relative(root, path);
  return fromRoot !== ".." && !fromRoot.startsWith(`..${sep}`) && !isAbsolute(fromRoot);
}

function mediaType(data: Uint8Array): ImageMediaType | undefined {
  if (data.length >= 8
    && data[0] === 0x89 && data[1] === 0x50 && data[2] === 0x4e && data[3] === 0x47
    && data[4] === 0x0d && data[5] === 0x0a && data[6] === 0x1a && data[7] === 0x0a) {
    return "image/png";
  }
  if (data.length >= 3 && data[0] === 0xff && data[1] === 0xd8 && data[2] === 0xff) {
    return "image/jpeg";
  }
  if (data.length >= 12
    && data[0] === 0x52 && data[1] === 0x49 && data[2] === 0x46 && data[3] === 0x46
    && data[8] === 0x57 && data[9] === 0x45 && data[10] === 0x42 && data[11] === 0x50) {
    return "image/webp";
  }
  return undefined;
}

export function createImageLoader(root = process.cwd()): ImageLoader {
  let resolvedRoot: Promise<string> | undefined;
  return async (paths, signal) => {
    try {
      resolvedRoot ??= realpath(resolve(root)).then(async (path) => {
        if (!(await stat(path)).isDirectory()) {
          throw new ImageLoadError("image root must be a directory.");
        }
        return path;
      });
      const rootPath = await resolvedRoot;
      const images: ImageInput[] = [];
      for (const input of paths) {
        signal?.throwIfAborted();
        if (typeof input !== "string" || input.trim().length === 0) {
          throw new ImageLoadError("image paths must be non-empty strings.");
        }
        const path = await realpath(resolve(rootPath, input));
        if (!isInside(rootPath, path)) {
          throw new ImageLoadError("image path is outside the configured root.");
        }
        if (!(await stat(path)).isFile()) {
          throw new ImageLoadError("image path must identify a regular file.");
        }
        const data = await readFile(path, signal ? { signal } : undefined);
        const detected = mediaType(data);
        if (detected === undefined) {
          throw new ImageLoadError("image must be PNG, JPEG, or WebP.");
        }
        images.push(Object.freeze({ mediaType: detected, base64: data.toString("base64") }));
      }
      return Object.freeze(images);
    } catch (error) {
      if (signal?.aborted || (error instanceof Error && error.name === "AbortError")) throw error;
      if (error instanceof ImageLoadError) throw error;
      throw new ImageLoadError("image file could not be read.", { cause: error });
    }
  };
}
