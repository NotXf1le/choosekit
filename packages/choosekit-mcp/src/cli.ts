#!/usr/bin/env node

import { serveStdio } from "@modelcontextprotocol/server/stdio";
import { fromLlamaCpp } from "choosekit/llama-cpp";
import { loadConfig } from "./config.js";
import { buildServer } from "./server.js";

try {
  const config = loadConfig();
  const chooser = fromLlamaCpp(config);
  serveStdio(() => buildServer(chooser, { mode: config.mode }));
} catch (error) {
  const message = error instanceof Error ? error.message : "Unknown configuration error.";
  console.error(`choosekit-mcp: ${message}`);
  process.exitCode = 1;
}
