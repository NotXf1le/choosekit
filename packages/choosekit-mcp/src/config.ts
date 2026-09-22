import { z } from "zod";

const urlSchema = z.string().url();

export type Config =
  | {
    readonly backend: "llama-cpp";
    readonly baseURL: string;
    readonly imageRoot?: string;
    readonly model?: string;
    readonly mode: "labels" | "minimal-prefix";
  }
  | {
    readonly backend: "openrouter";
    readonly apiKey: string;
    readonly imageRoot?: string;
    readonly model: string;
    readonly provider?: string;
    readonly mode: "labels";
  }
  | {
    readonly backend: "ollama";
    readonly baseURL?: string;
    readonly imageRoot?: string;
    readonly model: string;
    readonly mode: "labels";
  };

function requiredText(value: string | undefined, name: string): string {
  const text = value?.trim();
  if (!text) throw new TypeError(`${name} is required.`);
  return text;
}

function optionalText(value: string | undefined, name: string): string | undefined {
  if (value === undefined) return undefined;
  const text = value.trim();
  if (!text) throw new TypeError(`${name} must not be empty.`);
  return text;
}

export function loadConfig(env: NodeJS.ProcessEnv = process.env): Config {
  const backend = env.CHOOSEKIT_BACKEND?.trim() ?? "llama-cpp";
  const imageRoot = optionalText(env.CHOOSEKIT_IMAGE_ROOT, "CHOOSEKIT_IMAGE_ROOT");
  if (backend !== "llama-cpp" && backend !== "ollama" && backend !== "openrouter") {
    throw new TypeError("CHOOSEKIT_BACKEND must be llama-cpp, ollama, or openrouter.");
  }

  if (backend === "ollama") {
    const mode = env.CHOOSEKIT_MODE?.trim() ?? "labels";
    if (mode !== "labels") {
      throw new TypeError("CHOOSEKIT_MODE must be labels when using Ollama.");
    }
    const baseURL = optionalText(env.CHOOSEKIT_BASE_URL, "CHOOSEKIT_BASE_URL");
    if (baseURL !== undefined && !urlSchema.safeParse(baseURL).success) {
      throw new TypeError("CHOOSEKIT_BASE_URL must be a valid URL.");
    }
    return {
      backend,
      ...(baseURL === undefined ? {} : { baseURL }),
      ...(imageRoot === undefined ? {} : { imageRoot }),
      model: requiredText(env.CHOOSEKIT_MODEL, "CHOOSEKIT_MODEL"),
      mode,
    };
  }

  if (backend === "openrouter") {
    const mode = env.CHOOSEKIT_MODE?.trim() ?? "labels";
    if (mode !== "labels") {
      throw new TypeError("CHOOSEKIT_MODE must be labels when using OpenRouter.");
    }
    const provider = optionalText(env.OPENROUTER_PROVIDER, "OPENROUTER_PROVIDER");
    return {
      backend,
      apiKey: requiredText(env.OPENROUTER_API_KEY, "OPENROUTER_API_KEY"),
      ...(imageRoot === undefined ? {} : { imageRoot }),
      model: requiredText(env.CHOOSEKIT_MODEL, "CHOOSEKIT_MODEL"),
      ...(provider === undefined ? {} : { provider }),
      mode,
    };
  }

  const baseURL = requiredText(env.CHOOSEKIT_BASE_URL, "CHOOSEKIT_BASE_URL");
  if (!urlSchema.safeParse(baseURL).success) {
    throw new TypeError("CHOOSEKIT_BASE_URL must be a valid URL.");
  }
  const model = optionalText(env.CHOOSEKIT_MODEL, "CHOOSEKIT_MODEL");
  const mode = env.CHOOSEKIT_MODE?.trim() ?? "labels";
  if (mode !== "labels" && mode !== "minimal-prefix") {
    throw new TypeError("CHOOSEKIT_MODE must be labels or minimal-prefix.");
  }
  return {
    backend,
    baseURL,
    ...(imageRoot === undefined ? {} : { imageRoot }),
    ...(model === undefined ? {} : { model }),
    mode,
  };
}
