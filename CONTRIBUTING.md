# Contributing

Thank you for your interest in contributing to `choosekit`. Bug reports, documentation, examples, and code contributions are all welcome. You don't need to be an expert in language models to help.

## Find something to work on

Browse the [open issues](https://github.com/NotXf1le/choosekit/issues), including those labeled [`good first issue`](https://github.com/NotXf1le/choosekit/issues?q=is%3Aissue%20is%3Aopen%20label%3A%22good%20first%20issue%22). If you'd like to work on an issue, leave a comment so others know. Questions and ideas are welcome too.

Small fixes and documentation improvements can be submitted as pull requests without an issue first. For larger changes, new backends, or public API changes, please open an issue to discuss the approach before implementation.

To report a bug, describe the expected and actual behavior and include a minimal reproduction, Node.js version, and relevant backend, model, and server versions. Remove API keys, private prompts, and other sensitive data before posting.

## Set up

Use Node.js 20 or newer. Fork the repository on GitHub, clone your fork, and create a branch for your changes. The core library lives in `src/` and its tests in `tests/`; the MCP server and its tests live in `packages/choosekit-mcp/`.

From the repository root, install dependencies and check the core package:

```sh
npm ci
npm run check
npm pack --dry-run
```

For changes to the MCP server, install its separate dependencies and run its checks:

```sh
cd packages/choosekit-mcp
npm ci
npm run check
npm pack --dry-run
```

The MCP package depends on the published `choosekit` package: installing its dependencies does not automatically use changes in this checkout's `src/`. For changes spanning both packages, explain how you verified them together in the pull request.

The automated tests use fixtures and local test servers; they do not require a running model server or a real API key. Installing dependencies and the MCP package-install test may need access to the npm registry or a populated npm cache. Real-backend experiments and benchmarks have separate requirements; see the [benchmark instructions](benchmarks/README.md).

The MCP tests need to open local loopback connections and create a symbolic link. On Windows, creating the link may require additional permissions. Restricted environments may block these operations.

## Submit a pull request

Keep changes focused and add or update tests for changed behavior when relevant. Update documentation when the public behavior or setup changes.

Run the checks for each affected package and describe what you changed, why, and which checks you ran in the pull request. Mention any checks you could not run. Link the related issue, if any. Use the [pull request template](.github/pull_request_template.md) as a guide.

If you get stuck or aren't sure about an approach, ask in the issue or pull request.
