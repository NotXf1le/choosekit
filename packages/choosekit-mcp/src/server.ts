import { McpServer } from "@modelcontextprotocol/server";
import { ScoringError, type Chooser, type Decision, type Usage } from "choosekit";
import { z } from "zod";

export type ChoiceMode = "labels" | "minimal-prefix";

export interface ServerOptions {
  readonly mode?: ChoiceMode;
}

const usageSchema = z.object({
  promptTokens: z.number().int().nonnegative(),
  cachedTokens: z.number().int().nonnegative().nullable(),
  completionTokens: z.number().int().nonnegative(),
  requests: z.number().int().nonnegative(),
}).strict();

const decisionSchema = z.object({
  choice: z.string(),
  distribution: z.record(z.string(), z.number().finite().min(0).max(1)),
  scores: z.record(z.string(), z.number().finite().max(0)),
  margin: z.number().finite().min(0).max(1),
  entropy: z.number().finite().nonnegative(),
  boundaryTokens: z.number().int().nonnegative(),
  usage: usageSchema.optional(),
}).strict();

function inputSchema(mode: ChoiceMode) {
  const choicesSchema = z.fromJSONSchema({
    type: "object",
    propertyNames: {
      type: "string",
      pattern: "^(?!__proto__$)(?!\\s)(?!.*\\s$).+$",
    },
    additionalProperties: {
      type: "string",
      pattern: "\\S",
    },
    minProperties: 2,
    ...(mode === "labels" ? { maxProperties: 26 } : {}),
  }) as z.ZodType<Record<string, string>>;

  return z.object({
    context: z.string().describe("The context to use when making the decision."),
    question: z.string()
      .refine((question) => question.trim().length > 0, "The question must not be empty."),
    choices: choicesSchema,
  }).strict();
}

function toolDescription(mode: ChoiceMode): string {
  if (mode === "labels") {
    return "Choose the best option and return its probability distribution. Each description must contain the option's full meaning because choice keys identify the returned result but are not shown to the model. If none of the choices may apply, include an explicit insufficient-information or none-of-the-above choice.";
  }
  return "Choose the best option and return its probability distribution. Each description must contain the option's full meaning. The model scores the shortest token prefixes that distinguish the supplied choice keys. If none of the choices may apply, include an explicit insufficient-information or none-of-the-above choice.";
}

function errorResult(error: unknown, signal: AbortSignal) {
  if (signal.aborted || (error instanceof Error && error.name === "AbortError")) {
    throw error;
  }
  if (error instanceof ScoringError) {
    return {
      isError: true as const,
      content: [{ type: "text" as const, text: "llama.cpp could not score the supplied choices." }],
    };
  }
  return {
    isError: true as const,
    content: [{ type: "text" as const, text: "The choice request failed unexpectedly." }],
  };
}

export function buildServer(chooser: Chooser, options: ServerOptions = {}): McpServer {
  if (typeof chooser !== "function") throw new TypeError("chooser must be a function.");
  const mode = options.mode ?? "labels";
  if (mode !== "labels" && mode !== "minimal-prefix") {
    throw new TypeError("mode must be labels or minimal-prefix.");
  }

  const server = new McpServer({ name: "choosekit-mcp", version: "0.1.0" });
  server.registerTool(
    "choose",
    {
      description: toolDescription(mode),
      inputSchema: inputSchema(mode),
      outputSchema: decisionSchema,
      annotations: {
        readOnlyHint: true,
        openWorldHint: false,
      },
    },
    async ({ context, question, choices }, ctx) => {
      try {
        const decision = await chooser({
          context,
          question,
          choices,
          signal: ctx.mcpReq.signal,
        }) as Decision<string>;
        const structuredContent: {
          choice: string;
          distribution: Readonly<Record<string, number>>;
          scores: Readonly<Record<string, number>>;
          margin: number;
          entropy: number;
          boundaryTokens: number;
          usage?: Usage;
        } = {
          choice: decision.choice,
          distribution: decision.distribution,
          scores: decision.scores,
          margin: decision.margin,
          entropy: decision.entropy,
          boundaryTokens: decision.boundaryTokens,
          ...(decision.usage === undefined ? {} : { usage: decision.usage }),
        };
        return {
          content: [{ type: "text", text: JSON.stringify(structuredContent) }],
          structuredContent,
        };
      } catch (error) {
        return errorResult(error, ctx.mcpReq.signal);
      }
    },
  );
  return server;
}
