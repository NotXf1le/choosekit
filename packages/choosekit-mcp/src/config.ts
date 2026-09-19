import { z } from "zod";

const configSchema = z.object({
  CHOOSEKIT_BASE_URL: z.string().trim().url(),
  CHOOSEKIT_MODEL: z.string().trim().min(1).optional(),
  CHOOSEKIT_MODE: z.enum(["labels", "minimal-prefix"]).default("labels"),
});

export interface Config {
  baseURL: string;
  model?: string;
  mode: "labels" | "minimal-prefix";
}

export function loadConfig(env: NodeJS.ProcessEnv = process.env): Config {
  const parsed = configSchema.parse({
    CHOOSEKIT_BASE_URL: env.CHOOSEKIT_BASE_URL,
    CHOOSEKIT_MODEL: env.CHOOSEKIT_MODEL,
    CHOOSEKIT_MODE: env.CHOOSEKIT_MODE,
  });

  return {
    baseURL: parsed.CHOOSEKIT_BASE_URL,
    ...(parsed.CHOOSEKIT_MODEL === undefined
      ? {}
      : { model: parsed.CHOOSEKIT_MODEL }),
    mode: parsed.CHOOSEKIT_MODE,
  };
}
