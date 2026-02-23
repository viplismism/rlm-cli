#!/usr/bin/env tsx
/**
 * Standalone RLM CLI — run Recursive Language Model queries from the terminal.
 *
 * Usage:
 *   npx tsx src/cli.ts --model claude-sonnet-4-20250514 --file large-file.txt "What are the main themes?"
 *   npx tsx src/cli.ts --model claude-sonnet-4-20250514 --url https://example.com/big.txt "Summarize this"
 *   cat data.txt | npx tsx src/cli.ts --model claude-sonnet-4-20250514 --stdin "Count the errors"
 *
 * Environment:
 *   ANTHROPIC_API_KEY — required for Anthropic models
 *   OPENAI_API_KEY    — required for OpenAI models
 *   (etc. per @mariozechner/pi-ai provider)
 */

import "./env.js";
import * as fs from "node:fs";

// Dynamic imports — ensures env.js has set process.env BEFORE pi-ai loads
const { getModels, getProviders } = await import("@mariozechner/pi-ai");
const { PythonRepl } = await import("./repl.js");
const { runRlmLoop } = await import("./rlm.js");

import type { Api, Model } from "@mariozechner/pi-ai";

// ── Arg parsing ─────────────────────────────────────────────────────────────

function usage(): never {
	console.error(`
rlm-cli — Recursive Language Model CLI (arXiv:2512.24601)

USAGE
  rlm run [OPTIONS] "<query>"

OPTIONS
  --model <id>     Model ID (default: RLM_MODEL from .env)
  --file <path>    Read context from a file
  --url <url>      Fetch context from a URL
  --stdin          Read context from stdin (pipe data in)
  --verbose        Show iteration progress

EXAMPLES
  rlm run --file big.txt "List all classes"
  curl -s https://example.com/large.py | rlm run --stdin "Summarize"
  rlm run --url https://raw.githubusercontent.com/.../typing.py "Count public classes"
`.trim());
	process.exit(1);
}

interface CliArgs {
	modelId: string;
	file?: string;
	url?: string;
	useStdin: boolean;
	verbose: boolean;
	query: string;
}

function parseArgs(): CliArgs {
	const args = process.argv.slice(2);
	let modelId: string | undefined;
	let file: string | undefined;
	let url: string | undefined;
	let useStdin = false;
	let verbose = false;
	const positional: string[] = [];

	for (let i = 0; i < args.length; i++) {
		const arg = args[i];
		if (arg === "--model" && i + 1 < args.length) {
			modelId = args[++i];
		} else if (arg === "--file" && i + 1 < args.length) {
			file = args[++i];
		} else if (arg === "--url" && i + 1 < args.length) {
			url = args[++i];
		} else if (arg === "--stdin") {
			useStdin = true;
		} else if (arg === "--verbose") {
			verbose = true;
		} else if (arg === "--help" || arg === "-h") {
			usage();
		} else if (!arg.startsWith("--")) {
			positional.push(arg);
		} else {
			console.error(`Unknown option: ${arg}`);
			usage();
		}
	}

	if (!modelId) {
		modelId = process.env.RLM_MODEL || "claude-sonnet-4-5-20250929";
	}

	if (positional.length === 0) {
		console.error("Error: query argument is required");
		usage();
	}

	const query = positional.join(" ");

	if (!file && !url && !useStdin) {
		console.error("Error: one of --file, --url, or --stdin is required");
		usage();
	}

	return { modelId, file, url, useStdin, verbose, query };
}

// ── Helpers ─────────────────────────────────────────────────────────────────

async function readStdin(): Promise<string> {
	const chunks: Buffer[] = [];
	for await (const chunk of process.stdin) {
		chunks.push(chunk as Buffer);
	}
	return Buffer.concat(chunks).toString("utf-8");
}

async function fetchUrl(url: string): Promise<string> {
	const resp = await fetch(url);
	if (!resp.ok) {
		throw new Error(`Failed to fetch ${url}: ${resp.status} ${resp.statusText}`);
	}
	return resp.text();
}

// ── Main ────────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
	const args = parseArgs();

	// Resolve model by scanning all providers
	let model: Model<Api> | undefined;
	const allModelIds: string[] = [];
	for (const provider of getProviders()) {
		const providerModels = getModels(provider);
		for (const m of providerModels) {
			allModelIds.push(m.id);
			if (m.id === args.modelId) {
				model = m;
			}
		}
	}
	if (!model) {
		console.error(`Error: unknown model "${args.modelId}"`);
		console.error(`Available models: ${allModelIds.join(", ")}`);
		process.exit(1);
	}

	// Load context
	let context: string;
	if (args.file) {
		console.error(`Reading context from file: ${args.file}`);
		context = fs.readFileSync(args.file, "utf-8");
	} else if (args.url) {
		console.error(`Fetching context from URL: ${args.url}`);
		context = await fetchUrl(args.url);
	} else {
		console.error("Reading context from stdin...");
		context = await readStdin();
	}

	console.error(`Context loaded: ${context.length.toLocaleString()} characters`);
	console.error(`Model: ${model.id}`);
	console.error(`Query: ${args.query}`);
	console.error("---");

	// Start REPL
	const repl = new PythonRepl();
	const ac = new AbortController();

	process.on("SIGINT", () => {
		console.error("\nAborting...");
		ac.abort();
	});

	try {
		await repl.start(ac.signal);

		const startTime = Date.now();
		const result = await runRlmLoop({
			context,
			query: args.query,
			model,
			repl,
			signal: ac.signal,
			onProgress: args.verbose
				? (info) => {
						const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
						console.error(
							`[${elapsed}s] Iteration ${info.iteration}/${info.maxIterations} | ` +
								`Sub-queries: ${info.subQueries} | Phase: ${info.phase}`,
						);
					}
				: undefined,
		});

		const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
		console.error("---");
		console.error(
			`Completed in ${elapsed}s | ${result.iterations} iterations | ${result.totalSubQueries} sub-queries | ${result.completed ? "success" : "incomplete"}`,
		);
		console.error("---");

		// Write the answer to stdout (not stderr) so it can be piped
		console.log(result.answer);
	} finally {
		repl.shutdown();
	}
}

main().catch((err) => {
	console.error("Fatal error:", err);
	process.exit(1);
});
