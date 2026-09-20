import { z } from "zod";

const urlSchema = z.string().url();

export type Config =
  | {
    readonly backend: "llama-cpp";
    readonly baseURL: string;
    readonly model?: string;
    readonly mode: "labels" | "minimal-prefix";
  }
  | {
    readonly backend: "openrouter";
    readonly apiKey: string;
    readonly model: string;
    readonly provider?: string;
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
  if (backend !== "llama-cpp" && backend !== "openrouter") {
    throw new TypeError("CHOOSEKIT_BACKEND must be llama-cpp or openrouter.");
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
    ...(model === undefined ? {} : { model }),
    mode,
  };
}
