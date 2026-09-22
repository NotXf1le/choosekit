#!/usr/bin/env node

import { serveStdio } from "@modelcontextprotocol/server/stdio";
import type { Chooser } from "choosekit";
import { fromLlamaCpp } from "choosekit/llama-cpp";
import { fromOllama } from "choosekit/ollama";
import { fromOpenRouter } from "choosekit/openrouter";
import { loadConfig } from "./config.js";
import { createImageLoader } from "./images.js";
import { buildServer } from "./server.js";

try {
  const config = loadConfig();
  let chooser: Chooser;
  switch (config.backend) {
    case "llama-cpp":
      chooser = fromLlamaCpp({
        baseURL: config.baseURL,
        ...(config.model === undefined ? {} : { model: config.model }),
        mode: config.mode,
      });
      break;
    case "ollama":
      chooser = fromOllama({
        model: config.model,
        ...(config.baseURL === undefined ? {} : { baseURL: config.baseURL }),
      });
      break;
    case "openrouter":
      chooser = fromOpenRouter({
        apiKey: config.apiKey,
        model: config.model,
        ...(config.provider === undefined ? {} : { provider: config.provider }),
      });
      break;
  }
  serveStdio(() => buildServer(chooser, {
    backend: config.backend,
    imageLoader: createImageLoader(config.imageRoot),
    mode: config.mode,
  }));
} catch (error) {
  const message = error instanceof Error ? error.message : "Unknown configuration error.";
  console.error(`choosekit-mcp: ${message}`);
  process.exitCode = 1;
}
