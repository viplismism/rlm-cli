#!/usr/bin/env tsx
/**
 * RLM Trajectory TUI — real-time visualization of the RLM loop.
 *
 * Shows each iteration step-by-step: generated code, REPL output,
 * sub-queries, and the final answer. Also saves the full trajectory
 * to a JSON file for later inspection.
 *
 * Usage:
 *   npx tsx src/tui.ts --model claude-sonnet-4-20250514 --file big.txt "Your query"
 *   npx tsx src/tui.ts --model claude-sonnet-4-20250514 --url https://example.com/data.txt "Summarize"
 */

import "./env.js";
import * as fs from "node:fs";
import * as path from "node:path";

// Dynamic imports — ensures env.js has set process.env BEFORE pi-ai loads
const { getModels, getProviders } = await import("@mariozechner/pi-ai");
const { PythonRepl } = await import("./repl.js");
const { runRlmLoop } = await import("./rlm.js");

import type { Api, Model } from "@mariozechner/pi-ai";
import type { RlmProgress, RlmResult, SubQueryInfo } from "./rlm.js";

// ── ANSI colors ─────────────────────────────────────────────────────────────

const c = {
	reset: "\x1b[0m",
	bold: "\x1b[1m",
	dim: "\x1b[2m",
	italic: "\x1b[3m",
	red: "\x1b[31m",
	green: "\x1b[32m",
	yellow: "\x1b[33m",
	blue: "\x1b[34m",
	magenta: "\x1b[35m",
	cyan: "\x1b[36m",
	white: "\x1b[37m",
	gray: "\x1b[90m",
	bgBlue: "\x1b[44m",
	bgGreen: "\x1b[42m",
	bgYellow: "\x1b[43m",
	bgRed: "\x1b[41m",
};

// ── Box drawing helpers ─────────────────────────────────────────────────────

const FULL_WIDTH = 72;

function line(char: string = "━", width: number = FULL_WIDTH): string {
	return char.repeat(width);
}

function header(text: string, char: string = "━"): string {
	const padding = Math.max(0, FULL_WIDTH - text.length - 4);
	const left = Math.floor(padding / 2);
	const right = padding - left;
	return `${char.repeat(left)} ${text} ${char.repeat(right)}`;
}

function indent(text: string, prefix: string = "  │ "): string {
	return text
		.split("\n")
		.map((l) => `${prefix}${l}`)
		.join("\n");
}

function truncate(text: string, maxLines: number = 30): string {
	const lines = text.split("\n");
	if (lines.length <= maxLines) return text;
	const half = Math.floor(maxLines / 2);
	return [
		...lines.slice(0, half),
		`  ... (${lines.length - maxLines} lines omitted) ...`,
		...lines.slice(-half),
	].join("\n");
}

// ── Trajectory data types ───────────────────────────────────────────────────

interface TrajectoryStep {
	iteration: number;
	code: string | null;
	stdout: string;
	stderr: string;
	subQueries: { index: number; contextLength: number; instruction: string; resultLength: number; resultPreview: string }[];
	hasFinal: boolean;
	elapsedMs: number;
	/** The user message (prompt) sent to the LLM this iteration. */
	userMessage: string;
	/** The raw LLM response text (before code extraction). */
	rawResponse: string;
	/** System prompt (only on iteration 1). */
	systemPrompt?: string;
}

interface Trajectory {
	model: string;
	query: string;
	contextLength: number;
	contextLines: number;
	startTime: string;
	iterations: TrajectoryStep[];
	result: RlmResult | null;
	totalElapsedMs: number;
}

// ── TUI Renderer ────────────────────────────────────────────────────────────

class TrajectoryTUI {
	private trajectory: Trajectory;
	private currentStep: TrajectoryStep | null = null;
	private iterationStart: number = 0;
	private globalStart: number = Date.now();

	constructor(model: string, query: string, context: string) {
		this.trajectory = {
			model,
			query,
			contextLength: context.length,
			contextLines: context.split("\n").length,
			startTime: new Date().toISOString(),
			iterations: [],
			result: null,
			totalElapsedMs: 0,
		};
	}

	printHeader(): void {
		const w = process.stderr.write.bind(process.stderr);
		w(`\n${c.cyan}${line("━")}${c.reset}\n`);
		w(`${c.cyan}${header(`${c.bold}RLM Trajectory Viewer${c.reset}${c.cyan}`)}${c.reset}\n`);
		w(`${c.cyan}${line("━")}${c.reset}\n`);
		w(`  ${c.gray}Model:${c.reset}   ${c.bold}${this.trajectory.model}${c.reset}\n`);
		w(`  ${c.gray}Query:${c.reset}   ${c.yellow}${this.trajectory.query}${c.reset}\n`);
		w(`  ${c.gray}Context:${c.reset} ${this.trajectory.contextLength.toLocaleString()} chars │ ${this.trajectory.contextLines.toLocaleString()} lines\n`);
		w(`${c.cyan}${line("━")}${c.reset}\n\n`);
	}

	onProgress(info: RlmProgress): void {
		const w = process.stderr.write.bind(process.stderr);
		const elapsed = () => ((Date.now() - this.globalStart) / 1000).toFixed(1);

		if (info.phase === "generating_code") {
			// Start of a new iteration
			this.iterationStart = Date.now();
			this.currentStep = {
				iteration: info.iteration,
				code: null,
				stdout: "",
				stderr: "",
				subQueries: [],
				hasFinal: false,
				elapsedMs: 0,
				userMessage: info.userMessage || "",
				rawResponse: "",
				systemPrompt: info.systemPrompt,
			};

			w(`${c.blue}${c.bold}── Iteration ${info.iteration} / ${info.maxIterations} ${c.reset}${c.gray}${"─".repeat(Math.max(0, FULL_WIDTH - 24 - String(info.iteration).length - String(info.maxIterations).length))} ${elapsed()}s${c.reset}\n`);
			w(`\n  ${c.dim}⏳ Generating code...${c.reset}\n`);
		}

		if (info.phase === "executing" && info.code) {
			if (this.currentStep) {
				this.currentStep.code = info.code;
				this.currentStep.rawResponse = info.rawResponse || "";
			}

			w(`\n  ${c.green}${c.bold}✦ Generated Code:${c.reset}\n`);
			w(`${c.green}  ┌${"─".repeat(FULL_WIDTH - 4)}┐${c.reset}\n`);
			const codeDisplay = truncate(info.code, 25);
			for (const codeLine of codeDisplay.split("\n")) {
				w(`${c.green}  │${c.reset} ${c.white}${codeLine}${c.reset}\n`);
			}
			w(`${c.green}  └${"─".repeat(FULL_WIDTH - 4)}┘${c.reset}\n`);
			w(`\n  ${c.dim}⏳ Executing...${c.reset}\n`);
		}

		if (info.phase === "checking_final") {
			if (this.currentStep) {
				this.currentStep.stdout = info.stdout || "";
				this.currentStep.stderr = info.stderr || "";
				this.currentStep.elapsedMs = Date.now() - this.iterationStart;
			}

			if (info.stdout) {
				w(`\n  ${c.yellow}${c.bold}✦ REPL Output:${c.reset}\n`);
				w(`${c.yellow}  ┌${"─".repeat(FULL_WIDTH - 4)}┐${c.reset}\n`);
				const outDisplay = truncate(info.stdout, 20);
				for (const outLine of outDisplay.split("\n")) {
					w(`${c.yellow}  │${c.reset} ${outLine}\n`);
				}
				w(`${c.yellow}  └${"─".repeat(FULL_WIDTH - 4)}┘${c.reset}\n`);
			}

			if (info.stderr) {
				w(`\n  ${c.red}${c.bold}⚠ Stderr:${c.reset}\n`);
				w(indent(truncate(info.stderr, 10), `  ${c.red}│${c.reset} `) + "\n");
			}

			const iterElapsed = ((Date.now() - this.iterationStart) / 1000).toFixed(1);
			const finalStatus = this.currentStep?.hasFinal
				? `${c.green}● FINAL SET${c.reset}`
				: `${c.gray}◌ not final${c.reset}`;
			w(`\n  ${finalStatus} │ Sub-queries: ${c.cyan}${info.subQueries}${c.reset} │ ${c.dim}${iterElapsed}s${c.reset}\n\n`);

			if (this.currentStep) {
				this.trajectory.iterations.push({ ...this.currentStep });
			}
		}
	}

	onSubQuery(info: SubQueryInfo): void {
		const w = process.stderr.write.bind(process.stderr);
		if (this.currentStep) {
			this.currentStep.subQueries.push(info);
		}

		const instrPreview = info.instruction.length > 60
			? info.instruction.slice(0, 57) + "..."
			: info.instruction;
		const resultPreview = info.resultPreview.length > 60
			? info.resultPreview.slice(0, 57) + "..."
			: info.resultPreview;

		w(`    ${c.magenta}↳ Sub-query #${info.index}${c.reset} ${c.dim}(${(info.contextLength / 1000).toFixed(1)}K chars)${c.reset}\n`);
		w(`      ${c.dim}Ask:${c.reset} ${instrPreview}\n`);
		w(`      ${c.dim}Got:${c.reset} ${resultPreview}\n`);
	}

	printResult(result: RlmResult): void {
		this.trajectory.result = result;
		this.trajectory.totalElapsedMs = Date.now() - this.globalStart;

		// Mark last step as final if applicable
		if (result.completed && this.trajectory.iterations.length > 0) {
			this.trajectory.iterations[this.trajectory.iterations.length - 1].hasFinal = true;
		}

		const w = process.stderr.write.bind(process.stderr);
		const totalSec = (this.trajectory.totalElapsedMs / 1000).toFixed(1);

		w(`${c.green}${line("━")}${c.reset}\n`);
		w(`${c.green}${header(`${c.bold}RESULT${c.reset}${c.green}`)}${c.reset}\n`);
		w(`${c.green}${line("━")}${c.reset}\n\n`);

		// Print to stdout (the actual answer, pipeable)
		console.log(result.answer);

		w(`\n${c.cyan}${line("─")}${c.reset}\n`);
		w(`  ${c.gray}Iterations:${c.reset}   ${result.iterations}\n`);
		w(`  ${c.gray}Sub-queries:${c.reset}  ${result.totalSubQueries}\n`);
		w(`  ${c.gray}Completed:${c.reset}    ${result.completed ? `${c.green}yes${c.reset}` : `${c.red}no${c.reset}`}\n`);
		w(`  ${c.gray}Total time:${c.reset}   ${totalSec}s\n`);
		w(`${c.cyan}${line("─")}${c.reset}\n`);
	}

	async saveTrajectory(): Promise<string> {
		const dir = path.resolve(process.cwd(), "trajectories");
		if (!fs.existsSync(dir)) {
			fs.mkdirSync(dir, { recursive: true });
		}
		const timestamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
		const filePath = path.join(dir, `trajectory-${timestamp}.json`);
		fs.writeFileSync(filePath, JSON.stringify(this.trajectory, null, 2), "utf-8");
		return filePath;
	}
}

// ── CLI arg parsing ─────────────────────────────────────────────────────────

interface TuiArgs {
	modelId: string;
	file?: string;
	url?: string;
	useStdin: boolean;
	query: string;
	noSave: boolean;
}

function usage(): never {
	console.error(`
${c.cyan}${c.bold}rlm-tui${c.reset} — RLM Trajectory Viewer

${c.bold}USAGE${c.reset}
  npx tsx src/tui.ts --model <model-id> [OPTIONS] "<query>"

${c.bold}OPTIONS${c.reset}
  --model <id>     Model ID (e.g. claude-sonnet-4-20250514)  ${c.dim}[required]${c.reset}
  --file <path>    Read context from a file
  --url <url>      Fetch context from a URL
  --stdin          Read context from stdin
  --no-save        Don't save trajectory JSON

${c.bold}EXAMPLES${c.reset}
  npx tsx src/tui.ts --model claude-sonnet-4-20250514 --file big.txt "List all classes"
  cat data.txt | npx tsx src/tui.ts --model claude-sonnet-4-20250514 --stdin "Summarize"
`.trim());
	process.exit(1);
}

function parseArgs(): TuiArgs {
	const args = process.argv.slice(2);
	let modelId: string | undefined;
	let file: string | undefined;
	let url: string | undefined;
	let useStdin = false;
	let noSave = false;
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
		} else if (arg === "--no-save") {
			noSave = true;
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
		console.error("Error: --model is required");
		usage();
	}
	if (positional.length === 0) {
		console.error("Error: query argument is required");
		usage();
	}
	if (!file && !url && !useStdin) {
		console.error("Error: one of --file, --url, or --stdin is required");
		usage();
	}

	return { modelId, file, url, useStdin, query: positional.join(" "), noSave };
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

	// Resolve model
	let model: Model<Api> | undefined;
	const allModelIds: string[] = [];
	for (const provider of getProviders()) {
		for (const m of getModels(provider)) {
			allModelIds.push(m.id);
			if (m.id === args.modelId) model = m;
		}
	}
	if (!model) {
		console.error(`${c.red}Error: unknown model "${args.modelId}"${c.reset}`);
		console.error(`${c.dim}Available: ${allModelIds.join(", ")}${c.reset}`);
		process.exit(1);
	}

	// Load context
	let context: string;
	if (args.file) {
		context = fs.readFileSync(args.file, "utf-8");
	} else if (args.url) {
		process.stderr.write(`${c.dim}Fetching: ${args.url}${c.reset}\n`);
		context = await fetchUrl(args.url);
	} else {
		process.stderr.write(`${c.dim}Reading stdin...${c.reset}\n`);
		context = await readStdin();
	}

	// Initialize TUI
	const tui = new TrajectoryTUI(model.id, args.query, context);
	tui.printHeader();

	// Start REPL
	const repl = new PythonRepl();
	const ac = new AbortController();

	process.on("SIGINT", () => {
		process.stderr.write(`\n${c.red}${c.bold}Aborted by user${c.reset}\n`);
		ac.abort();
	});

	try {
		await repl.start(ac.signal);

		const result = await runRlmLoop({
			context,
			query: args.query,
			model,
			repl,
			signal: ac.signal,
			onProgress: (info) => tui.onProgress(info),
			onSubQuery: (info) => tui.onSubQuery(info),
		});

		tui.printResult(result);

		if (!args.noSave) {
			const trajPath = await tui.saveTrajectory();
			process.stderr.write(`\n  ${c.dim}Trajectory saved: ${trajPath}${c.reset}\n\n`);
		}
	} finally {
		repl.shutdown();
	}
}

main().catch((err) => {
	console.error(`${c.red}Fatal error:${c.reset}`, err);
	process.exit(1);
});
