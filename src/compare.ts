#!/usr/bin/env tsx
/**
 * Comparison test: runs the same query with and without RLM, then prints both results.
 *
 * Usage:
 *   npx tsx src/compare.ts --model claude-sonnet-4-20250514 --url https://raw.githubusercontent.com/.../typing.py "List all public classes"
 *   npx tsx src/compare.ts --model claude-sonnet-4-20250514 --file large-file.txt "Summarize the main concepts"
 */

import "./env.js";
import * as fs from "node:fs";

// Dynamic imports — ensures env.js has set process.env BEFORE pi-ai loads
const { completeSimple, getModels, getProviders } = await import("@mariozechner/pi-ai");
const { PythonRepl } = await import("./repl.js");
const { runRlmLoop } = await import("./rlm.js");

import type { Api, Model, TextContent } from "@mariozechner/pi-ai";

// ── Arg parsing (reused from cli.ts) ────────────────────────────────────────

interface CompareArgs {
	modelId: string;
	file?: string;
	url?: string;
	useStdin: boolean;
	query: string;
}

function usage(): never {
	console.error(`
rlm-compare — Compare direct LLM vs RLM-enhanced responses

USAGE
  npx tsx src/compare.ts --model <model-id> [OPTIONS] "<query>"

OPTIONS
  --model <id>     Model ID  [required]
  --file <path>    Context from file
  --url <url>      Context from URL
  --stdin          Context from stdin
`.trim());
	process.exit(1);
}

function parseArgs(): CompareArgs {
	const args = process.argv.slice(2);
	let modelId: string | undefined;
	let file: string | undefined;
	let url: string | undefined;
	let useStdin = false;
	const positional: string[] = [];

	for (let i = 0; i < args.length; i++) {
		const a = args[i];
		if (a === "--model" && i + 1 < args.length) {
			modelId = args[++i];
		} else if (a === "--file" && i + 1 < args.length) {
			file = args[++i];
		} else if (a === "--url" && i + 1 < args.length) {
			url = args[++i];
		} else if (a === "--stdin") {
			useStdin = true;
		} else if (a === "--help" || a === "-h") {
			usage();
		} else if (!a.startsWith("--")) {
			positional.push(a);
		}
	}

	if (!modelId || positional.length === 0) usage();
	if (!file && !url && !useStdin) {
		console.error("Error: one of --file, --url, or --stdin required");
		usage();
	}

	return { modelId, file, url, useStdin, query: positional.join(" ") };
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
	if (!resp.ok) throw new Error(`Fetch failed: ${resp.status} ${resp.statusText}`);
	return resp.text();
}

// ── Main ────────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
	const args = parseArgs();

	let model: Model<Api> | undefined;
	for (const provider of getProviders()) {
		const providerModels = getModels(provider);
		for (const m of providerModels) {
			if (m.id === args.modelId) {
				model = m;
			}
		}
	}
	if (!model) {
		console.error(`Unknown model: ${args.modelId}`);
		process.exit(1);
	}

	// Load context
	let context: string;
	if (args.file) {
		context = fs.readFileSync(args.file, "utf-8");
	} else if (args.url) {
		console.error(`Fetching: ${args.url}`);
		context = await fetchUrl(args.url);
	} else {
		context = await readStdin();
	}

	console.error(`Context: ${context.length.toLocaleString()} chars`);
	console.error(`Model: ${model.id}`);
	console.error(`Query: ${args.query}\n`);

	// ── Run 1: Direct LLM (no RLM) ──────────────────────────────────────────
	console.error("=== WITHOUT RLM (direct LLM) ===");
	const t1 = Date.now();

	const directResponse = await completeSimple(model, {
		messages: [
			{
				role: "user",
				content: `Context:\n${context}\n\nQuery: ${args.query}`,
				timestamp: Date.now(),
			},
		],
	});

	const directTime = ((Date.now() - t1) / 1000).toFixed(1);
	const directText = directResponse.content
		.filter((b): b is TextContent => b.type === "text")
		.map((b) => b.text)
		.join("\n");

	console.error(`Completed in ${directTime}s (${directText.length} chars)\n`);

	// ── Run 2: With RLM ─────────────────────────────────────────────────────
	console.error("=== WITH RLM ===");
	const repl = new PythonRepl();
	const t2 = Date.now();

	try {
		await repl.start();
		const result = await runRlmLoop({
			context,
			query: args.query,
			model,
			repl,
			onProgress: (info) => {
				const elapsed = ((Date.now() - t2) / 1000).toFixed(1);
				console.error(
					`  [${elapsed}s] iter ${info.iteration}/${info.maxIterations} | subs: ${info.subQueries} | ${info.phase}`,
				);
			},
		});

		const rlmTime = ((Date.now() - t2) / 1000).toFixed(1);
		console.error(
			`Completed in ${rlmTime}s | ${result.iterations} iters | ${result.totalSubQueries} sub-queries | ${result.answer.length} chars\n`,
		);

		// ── Print results ────────────────────────────────────────────────────
		console.log("╔══════════════════════════════════════════════════════╗");
		console.log("║              WITHOUT RLM (direct)                   ║");
		console.log("╚══════════════════════════════════════════════════════╝");
		console.log(directText);
		console.log();
		console.log("╔══════════════════════════════════════════════════════╗");
		console.log("║              WITH RLM (recursive)                   ║");
		console.log("╚══════════════════════════════════════════════════════╝");
		console.log(result.answer);
		console.log();
		console.log("--- Summary ---");
		console.log(`Direct:  ${directTime}s | ${directText.length} chars`);
		console.log(
			`RLM:     ${rlmTime}s | ${result.answer.length} chars | ${result.iterations} iters | ${result.totalSubQueries} sub-queries`,
		);
	} finally {
		repl.shutdown();
	}
}

main().catch((err) => {
	console.error("Fatal error:", err);
	process.exit(1);
});
