#!/usr/bin/env tsx
/**
 * RLM Interactive — Production-quality interactive terminal REPL.
 *
 * Launch with `rlm` and get a persistent session where you can:
 *   - Set context (file/URL/paste)
 *   - Type queries and watch the RLM loop run with smooth, real-time output
 *   - Browse previous trajectories
 */

import "./env.js";
import * as fs from "node:fs";
import * as path from "node:path";
import * as readline from "node:readline";
import { stdin, stdout } from "node:process";

const { getModels, getProviders } = await import("@mariozechner/pi-ai");
const { PythonRepl } = await import("./repl.js");
const { runRlmLoop } = await import("./rlm.js");
const { loadConfig } = await import("./config.js");

import type { Api, Model } from "@mariozechner/pi-ai";
import type { RlmProgress, SubQueryStartInfo, SubQueryInfo } from "./rlm.js";

const config = loadConfig();

// ── ANSI helpers ────────────────────────────────────────────────────────────

const c = {
	reset: "\x1b[0m",
	bold: "\x1b[1m",
	dim: "\x1b[2m",
	italic: "\x1b[3m",
	underline: "\x1b[4m",
	red: "\x1b[31m",
	green: "\x1b[32m",
	yellow: "\x1b[33m",
	blue: "\x1b[34m",
	magenta: "\x1b[35m",
	cyan: "\x1b[36m",
	white: "\x1b[37m",
	gray: "\x1b[90m",
	clearLine: "\x1b[2K\r",
};

// ── Spinner ─────────────────────────────────────────────────────────────────

const SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

class Spinner {
	private interval: ReturnType<typeof setInterval> | null = null;
	private frameIndex = 0;
	private message = "";
	private startTime = Date.now();

	start(message: string): void {
		this.stop();
		this.message = message;
		this.startTime = Date.now();
		this.frameIndex = 0;
		this.render();
		this.interval = setInterval(() => this.render(), 80);
	}

	update(message: string): void {
		this.message = message;
	}

	stop(): void {
		if (this.interval) {
			clearInterval(this.interval);
			this.interval = null;
			process.stdout.write(c.clearLine);
		}
	}

	private render(): void {
		const frame = SPINNER_FRAMES[this.frameIndex % SPINNER_FRAMES.length];
		const elapsed = ((Date.now() - this.startTime) / 1000).toFixed(1);
		process.stdout.write(
			`${c.clearLine}    ${c.cyan}${frame}${c.reset} ${this.message} ${c.dim}${elapsed}s${c.reset}`
		);
		this.frameIndex++;
	}
}

// ── Constants ───────────────────────────────────────────────────────────────

const DEFAULT_MODEL = process.env.RLM_MODEL || "claude-sonnet-4-5-20250929";
const TRAJ_DIR = path.resolve(process.cwd(), "trajectories");
const W = Math.min(process.stdout.columns || 80, 100);

// ── Session state ───────────────────────────────────────────────────────────

let currentModelId = DEFAULT_MODEL;
let currentModel: Model<Api> | undefined;
let contextText = "";
let contextSource = "";
let queryCount = 0;
let isRunning = false;

// Exposed so the readline SIGINT handler can abort the running query
let activeAc: AbortController | null = null;
let activeRepl: InstanceType<typeof PythonRepl> | null = null;
let activeSpinner: Spinner | null = null;

// ── Resolve model ───────────────────────────────────────────────────────────

function resolveModel(modelId: string): Model<Api> | undefined {
	for (const provider of getProviders()) {
		for (const m of getModels(provider)) {
			if (m.id === modelId) return m;
		}
	}
	return undefined;
}

function detectProvider(): string {
	if (process.env.ANTHROPIC_API_KEY) return "anthropic";
	if (process.env.OPENAI_API_KEY) {
		if (process.env.OPENAI_BASE_URL?.includes("openrouter")) return "openrouter";
		return "openai";
	}
	return "unknown";
}

// ── Paste detection ─────────────────────────────────────────────────────────

function isMultiLineInput(input: string): boolean {
	return input.includes("\n");
}

function handleMultiLineAsContext(input: string): { context: string; query: string } | null {
	const lines = input.split("\n");
	if (lines.length > 3) {
		const sizeKB = (input.length / 1024).toFixed(1);
		console.log(`  ${c.green}✓${c.reset} Pasted ${c.bold}${lines.length} lines${c.reset} ${c.dim}(${sizeKB}KB)${c.reset}`);
		return { context: input, query: "" };
	}
	return null;
}

// ── Banner ──────────────────────────────────────────────────────────────────

function printBanner(): void {
	console.log(`
${c.cyan}${c.bold}
                         ██████╗ ██╗     ███╗   ███╗
                         ██╔══██╗██║     ████╗ ████║
                         ██████╔╝██║     ██╔████╔██║
                         ██╔══██╗██║     ██║╚██╔╝██║
                         ██║  ██║███████╗██║ ╚═╝ ██║
                         ╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝
${c.reset}
${c.dim}         Recursive Language Models — arXiv:2512.24601${c.reset}
`);
}

// ── Status line ─────────────────────────────────────────────────────────────

function printStatusLine(): void {
	const provider = detectProvider();
	const modelShort = currentModelId.length > 35
		? currentModelId.slice(0, 32) + "..."
		: currentModelId;
	const ctx = contextText
		? `${c.green}●${c.reset} ${(contextText.length / 1024).toFixed(1)}KB${contextSource ? ` ${c.dim}(${contextSource})${c.reset}` : ""}`
		: `${c.dim}○${c.reset}`;

	console.log(
		`  ${c.dim}${modelShort}${c.reset} ${c.dim}(${provider})${c.reset}  ${ctx}  ${c.dim}Q:${queryCount}${c.reset}`
	);
}

// ── Welcome ─────────────────────────────────────────────────────────────────

function printWelcome(): void {
	console.clear();
	printBanner();

	printStatusLine();
	console.log(`  ${c.dim}max ${config.max_iterations} iterations · depth ${config.max_depth} · ${config.max_sub_queries} sub-queries${c.reset}`);
	console.log();
	console.log(`  ${c.dim}/help for commands${c.reset}`);
	console.log();
}

// ── Help ────────────────────────────────────────────────────────────────────

function printCommandHelp(): void {
	console.log(`
${c.bold}Context${c.reset}
  ${c.yellow}/file${c.reset} <path>         Load file as context
  ${c.yellow}/url${c.reset} <url>           Fetch URL as context
  ${c.yellow}/paste${c.reset}               Multi-line paste mode (EOF to finish)
  ${c.yellow}/context${c.reset}             Show loaded context info
  ${c.yellow}/clear-context${c.reset}       Unload context

${c.bold}Tools${c.reset}
  ${c.yellow}/trajectories${c.reset}        List saved runs

${c.bold}General${c.reset}
  ${c.yellow}/clear${c.reset}               Clear screen
  ${c.yellow}/help${c.reset}                Show this help
  ${c.yellow}/quit${c.reset}                Exit

  ${c.dim}Or just paste a URL or 4+ lines of code, then type your query.${c.reset}
`);
}

// ── Slash command handlers ──────────────────────────────────────────────────

async function handleFile(arg: string): Promise<void> {
	if (!arg) {
		console.log(`  ${c.red}Usage: /file <path>${c.reset}`);
		return;
	}
	const filePath = path.resolve(arg);
	if (!fs.existsSync(filePath)) {
		console.log(`  ${c.red}File not found: ${filePath}${c.reset}`);
		return;
	}
	contextText = fs.readFileSync(filePath, "utf-8");
	contextSource = arg;
	const lines = contextText.split("\n").length;
	console.log(
		`  ${c.green}✓${c.reset} Loaded ${c.bold}${contextText.length.toLocaleString()}${c.reset} chars (${lines.toLocaleString()} lines) from ${c.underline}${arg}${c.reset}`
	);
}

async function handleUrl(arg: string): Promise<void> {
	if (!arg) {
		console.log(`  ${c.red}Usage: /url <url>${c.reset}`);
		return;
	}
	console.log(`  ${c.dim}Fetching ${arg}...${c.reset}`);
	try {
		const resp = await fetch(arg);
		if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`);
		contextText = await resp.text();
		contextSource = arg;
		const lines = contextText.split("\n").length;
		console.log(
			`  ${c.green}✓${c.reset} Fetched ${c.bold}${contextText.length.toLocaleString()}${c.reset} chars (${lines.toLocaleString()} lines)`
		);
	} catch (err: any) {
		console.log(`  ${c.red}Failed: ${err.message}${c.reset}`);
	}
}

function handlePaste(rl: readline.Interface): Promise<void> {
	return new Promise((resolve) => {
		console.log(`  ${c.dim}Paste your context below. Type ${c.bold}EOF${c.reset}${c.dim} on an empty line to finish.${c.reset}`);
		const lines: string[] = [];
		const onLine = (line: string) => {
			if (line.trim() === "EOF") {
				rl.removeListener("line", onLine);
				contextText = lines.join("\n");
				contextSource = "(pasted)";
				console.log(
					`  ${c.green}✓${c.reset} Loaded ${c.bold}${contextText.length.toLocaleString()}${c.reset} chars (${lines.length} lines) from paste`
				);
				resolve();
				return;
			}
			lines.push(line);
		};
		rl.on("line", onLine);
	});
}

function handleContext(): void {
	if (!contextText) {
		console.log(`  ${c.dim}No context loaded. Use /file, /url, or /paste.${c.reset}`);
		return;
	}
	const lines = contextText.split("\n").length;
	console.log(`  ${c.bold}Context:${c.reset} ${contextText.length.toLocaleString()} chars, ${lines.toLocaleString()} lines`);
	console.log(`  ${c.bold}Source:${c.reset}  ${contextSource}`);
	console.log();
	const preview = contextText.slice(0, 500);
	const previewLines = preview.split("\n").slice(0, 8);
	for (const l of previewLines) {
		console.log(`  ${c.dim}│${c.reset} ${l}`);
	}
	if (contextText.length > 500) {
		console.log(`  ${c.dim}│ ...${c.reset}`);
	}
}

function handleTrajectories(): void {
	if (!fs.existsSync(TRAJ_DIR)) {
		console.log(`  ${c.dim}No trajectories yet.${c.reset}`);
		return;
	}
	const files = fs
		.readdirSync(TRAJ_DIR)
		.filter((f) => f.endsWith(".json"))
		.sort()
		.reverse();
	if (files.length === 0) {
		console.log(`  ${c.dim}No trajectories yet.${c.reset}`);
		return;
	}
	console.log(`\n  ${c.bold}Saved trajectories:${c.reset}\n`);
	for (const f of files.slice(0, 15)) {
		const stat = fs.statSync(path.join(TRAJ_DIR, f));
		const size = (stat.size / 1024).toFixed(1);
		console.log(`  ${c.dim}•${c.reset} ${f} ${c.dim}(${size}K)${c.reset}`);
	}
	if (files.length > 15) {
		console.log(`  ${c.dim}... and ${files.length - 15} more${c.reset}`);
	}
	console.log();
}

// ── Display helpers ─────────────────────────────────────────────────────────

const BOX_W = Math.min(process.stdout.columns || 80, 96) - 6; // panel inner width
const MAX_CONTENT_W = BOX_W - 4; // usable chars inside │ … │

/** Wrap a raw text line into chunks that fit within maxWidth. */
function wrapText(text: string, maxWidth: number): string[] {
	if (text.length <= maxWidth) return [text];
	const result: string[] = [];
	for (let i = 0; i < text.length; i += maxWidth) {
		result.push(text.slice(i, i + maxWidth));
	}
	return result;
}

function boxTop(title: string, color: string): string {
	const inner = BOX_W - 2;
	const t = ` ${title} `;
	const right = Math.max(0, inner - t.length);
	return `    ${color}╭${t}${"─".repeat(right)}╮${c.reset}`;
}

function boxBottom(color: string): string {
	return `    ${color}╰${"─".repeat(BOX_W - 2)}╯${c.reset}`;
}

function boxLine(text: string, color: string): string {
	const stripped = text.replace(/\x1b\[[0-9;]*m/g, "");
	const pad = Math.max(0, MAX_CONTENT_W - stripped.length);
	return `    ${color}│${c.reset} ${text}${" ".repeat(pad)} ${color}│${c.reset}`;
}

function displayCode(code: string): void {
	const lines = code.split("\n");
	const lineNumWidth = String(lines.length).length;
	const codeMaxW = MAX_CONTENT_W - lineNumWidth - 1;

	console.log(boxTop("Code", c.blue));
	for (let i = 0; i < lines.length; i++) {
		const wrapped = wrapText(lines[i], codeMaxW);
		for (let j = 0; j < wrapped.length; j++) {
			const prefix = j === 0
				? `${c.dim}${String(i + 1).padStart(lineNumWidth)}${c.reset}`
				: " ".repeat(lineNumWidth);
			console.log(boxLine(`${prefix} ${c.cyan}${wrapped[j]}${c.reset}`, c.blue));
		}
	}
	console.log(boxBottom(c.blue));
}

function displayOutput(output: string): void {
	const lines = output.split("\n").filter(l => l.trim() !== "");

	console.log(boxTop("Output", c.green));
	for (const line of lines) {
		for (const chunk of wrapText(line, MAX_CONTENT_W)) {
			console.log(boxLine(`${c.green}${chunk}${c.reset}`, c.green));
		}
	}
	console.log(boxBottom(c.green));
}

function displayError(stderr: string): void {
	const lines = stderr.split("\n").filter(l => l.trim() !== "");
	console.log(boxTop("Error", c.red));
	for (const line of lines) {
		for (const chunk of wrapText(line, MAX_CONTENT_W)) {
			console.log(boxLine(`${c.red}${chunk}${c.reset}`, c.red));
		}
	}
	console.log(boxBottom(c.red));
}

function formatSize(chars: number): string {
	return chars >= 1000 ? `${(chars / 1000).toFixed(1)}K` : `${chars}`;
}

function displaySubQueryStart(info: SubQueryStartInfo): void {
	console.log(`    ${c.magenta}┌─ Sub-query #${info.index}${c.reset} ${c.dim}sending ${formatSize(info.contextLength)} chars${c.reset}`);

	const instrLines = info.instruction.split("\n").filter(l => l.trim());
	for (const line of instrLines) {
		for (const chunk of wrapText(line, MAX_CONTENT_W)) {
			console.log(`    ${c.magenta}│${c.reset}  ${c.dim}${chunk}${c.reset}`);
		}
	}
}

function displaySubQueryResult(info: SubQueryInfo): void {
	const elapsed = (info.elapsedMs / 1000).toFixed(1);
	const resultLines = info.resultPreview.split("\n");

	console.log(`    ${c.magenta}│${c.reset}`);
	console.log(`    ${c.magenta}│${c.reset} ${c.green}${c.bold}Response:${c.reset}`);
	for (const line of resultLines) {
		for (const chunk of wrapText(line, MAX_CONTENT_W)) {
			console.log(`    ${c.magenta}│${c.reset}  ${c.green}${chunk}${c.reset}`);
		}
	}

	console.log(`    ${c.magenta}└─${c.reset} ${c.dim}${elapsed}s · ${formatSize(info.resultLength)} received${c.reset}`);
}

// ── Truncate helper ─────────────────────────────────────────────────────────

function truncateStr(text: string, max: number): string {
	return text.length <= max ? text : text.slice(0, max - 3) + "...";
}

// ── Run RLM query ───────────────────────────────────────────────────────────

async function runQuery(query: string): Promise<void> {
	const effectiveContext = contextText || query;
	const isDirectMode = !contextText;

	if (!currentModel) {
		currentModel = resolveModel(currentModelId);
	}
	if (!currentModel) {
		console.log(`\n  ${c.red}Model "${currentModelId}" not found.${c.reset}`);
		console.log(`  ${c.dim}Check RLM_MODEL in your .env file.${c.reset}\n`);
		return;
	}

	isRunning = true;
	queryCount++;
	const startTime = Date.now();
	let subQueryCount = 0;
	const spinner = new Spinner();

	// Header — just model + context info, query is already visible on the prompt line
	const ctxLabel = isDirectMode
		? `${c.dim}direct mode${c.reset}`
		: `${c.dim}${(effectiveContext.length / 1024).toFixed(1)}KB context${c.reset}`;
	console.log(`\n  ${c.dim}${currentModelId}${c.reset} · ${ctxLabel}\n`);

	// Trajectory bookkeeping
	const trajectory: any = {
		model: currentModelId,
		query,
		contextLength: effectiveContext.length,
		contextLines: effectiveContext.split("\n").length,
		startTime: new Date().toISOString(),
		iterations: [],
		result: null,
		totalElapsedMs: 0,
	};

	let currentStep: any = null;
	let iterStart = Date.now();

	const repl = new PythonRepl();
	const ac = new AbortController();

	// Expose to the readline SIGINT handler
	activeAc = ac;
	activeRepl = repl;
	activeSpinner = spinner;

	try {
		await repl.start(ac.signal);

		const result = await runRlmLoop({
			context: effectiveContext,
			query,
			model: currentModel,
			repl,
			signal: ac.signal,
			onProgress: (info: RlmProgress) => {
				if (info.phase === "generating_code") {
					iterStart = Date.now();
					currentStep = {
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

					const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
					const bar = "─".repeat(BOX_W - 2);
					console.log(`  ${c.blue}${bar}${c.reset}`);
					console.log(`  ${c.blue}${c.bold} Step ${info.iteration}${c.reset}${c.dim}/${info.maxIterations}${c.reset}  ${c.dim}${elapsed}s elapsed${c.reset}`);
					console.log(`  ${c.blue}${bar}${c.reset}`);
					spinner.start("Generating code");
				}

				if (info.phase === "executing" && info.code) {
					spinner.stop();

					if (currentStep) {
						currentStep.code = info.code;
						currentStep.rawResponse = info.rawResponse || "";
					}

					displayCode(info.code);
					spinner.start("Executing");
				}

				if (info.phase === "checking_final") {
					spinner.stop();

					if (currentStep) {
						currentStep.stdout = info.stdout || "";
						currentStep.stderr = info.stderr || "";
						currentStep.elapsedMs = Date.now() - iterStart;
						trajectory.iterations.push({ ...currentStep });
					}

					if (info.stdout) {
						displayOutput(info.stdout);
					}

					if (info.stderr) {
						displayError(info.stderr);
					}

					const iterElapsed = ((Date.now() - iterStart) / 1000).toFixed(1);
					const sqLabel = info.subQueries > 0 ? ` · ${info.subQueries} sub-queries` : "";
					console.log(`    ${c.dim}${iterElapsed}s${sqLabel}${c.reset}`);
					console.log();
				}
			},
			onSubQueryStart: (info: SubQueryStartInfo) => {
				spinner.stop();
				displaySubQueryStart(info);
				spinner.start("");
			},
			onSubQuery: (info: SubQueryInfo) => {
				subQueryCount++;
				if (currentStep) {
					currentStep.subQueries.push(info);
				}

				spinner.stop();
				displaySubQueryResult(info);
				spinner.start("Executing");
			},
		});

		trajectory.result = result;
		trajectory.totalElapsedMs = Date.now() - startTime;

		if (result.completed && trajectory.iterations.length > 0) {
			trajectory.iterations[trajectory.iterations.length - 1].hasFinal = true;
		}

		// Final answer
		const totalSec = ((Date.now() - startTime) / 1000).toFixed(1);
		const stats = `${result.iterations} step${result.iterations !== 1 ? "s" : ""} · ${result.totalSubQueries} sub-quer${result.totalSubQueries !== 1 ? "ies" : "y"} · ${totalSec}s`;

		const answerLines = result.answer.split("\n");
		console.log(boxTop(`✔ Result  ${c.dim}${stats}`, c.green));
		for (const line of answerLines) {
			for (const chunk of wrapText(line, MAX_CONTENT_W)) {
				console.log(boxLine(chunk, c.green));
			}
		}
		console.log(boxBottom(c.green));
		console.log();

		// Save trajectory
		if (!fs.existsSync(TRAJ_DIR)) fs.mkdirSync(TRAJ_DIR, { recursive: true });
		const ts = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
		const trajFile = `trajectory-${ts}.json`;
		fs.writeFileSync(path.join(TRAJ_DIR, trajFile), JSON.stringify(trajectory, null, 2), "utf-8");
		console.log(`  ${c.dim}Saved: ${trajFile}${c.reset}\n`);
	} catch (err: any) {
		spinner.stop();
		const msg = err?.message || String(err);
		// Suppress expected abort/shutdown errors
		if (
			err.name !== "AbortError" &&
			!msg.includes("Aborted") &&
			!msg.includes("not running") &&
			!msg.includes("REPL subprocess") &&
			!msg.includes("REPL shut down")
		) {
			console.log(`\n  ${c.red}Error: ${msg}${c.reset}\n`);
		}
	} finally {
		spinner.stop();
		activeAc = null;
		activeRepl = null;
		activeSpinner = null;
		try { repl.shutdown(); } catch { /* already dead */ }
		isRunning = false;
	}
}

// ── @file shorthand and auto-detect file paths ─────────────────────────────

function extractFilePath(input: string): { filePath: string | null; query: string } {
	const atMatch = input.match(/@(\S+)/);
	if (atMatch) {
		const filePath = path.resolve(atMatch[1]);
		if (fs.existsSync(filePath)) {
			const query = input.replace(atMatch[0], "").trim();
			return { filePath, query };
		}
	}

	const absPathMatch = input.match(/(\/[^\s]+)/);
	if (absPathMatch) {
		const filePath = absPathMatch[1];
		if (fs.existsSync(filePath)) {
			const query = input.replace(absPathMatch[1], "").trim();
			return { filePath, query };
		}
	}

	const relPathMatch = input.match(/([\w\-\.]+\/[\w\-\./]+\.\w{2,6})/);
	if (relPathMatch) {
		const filePath = path.resolve(relPathMatch[1]);
		if (fs.existsSync(filePath)) {
			const query = input.replace(relPathMatch[1], "").trim();
			return { filePath, query };
		}
	}

	return { filePath: null, query: input };
}

function expandAtFiles(input: string): string {
	const atMatch = input.match(/^@(\S+)\s*(.*)/);
	if (atMatch) {
		const filePath = path.resolve(atMatch[1]);
		if (fs.existsSync(filePath)) {
			contextText = fs.readFileSync(filePath, "utf-8");
			contextSource = atMatch[1];
			const lines = contextText.split("\n").length;
			console.log(
				`  ${c.green}✓${c.reset} Loaded ${c.bold}${contextText.length.toLocaleString()}${c.reset} chars (${lines} lines) from ${c.underline}${atMatch[1]}${c.reset}`
			);
			return atMatch[2] || "";
		} else {
			console.log(`  ${c.red}File not found: ${atMatch[1]}${c.reset}`);
			return "";
		}
	}
	return input;
}

// ── Auto-detect URLs ────────────────────────────────────────────────────────

async function detectAndLoadUrl(input: string): Promise<boolean> {
	const urlMatch = input.match(/^https?:\/\/\S+$/);
	if (urlMatch) {
		const url = urlMatch[0];
		console.log(`  ${c.dim}Fetching ${url}...${c.reset}`);
		try {
			const resp = await fetch(url);
			if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`);
			contextText = await resp.text();
			contextSource = url;
			const lines = contextText.split("\n").length;
			console.log(
				`  ${c.green}✓${c.reset} Loaded ${c.bold}${contextText.length.toLocaleString()}${c.reset} chars (${lines} lines)`
			);
			return true;
		} catch (err: any) {
			console.log(`  ${c.red}Failed: ${err.message}${c.reset}`);
			return false;
		}
	}
	return false;
}

// ── Main interactive loop ───────────────────────────────────────────────────

async function interactive(): Promise<void> {
	// Validate env
	const hasApiKey = process.env.ANTHROPIC_API_KEY || process.env.OPENAI_API_KEY;
	if (!hasApiKey) {
		console.log(`\n  ${c.red}No API key found.${c.reset}`);
		console.log(`  Set ${c.bold}ANTHROPIC_API_KEY${c.reset} or ${c.bold}OPENAI_API_KEY${c.reset} in your .env file.\n`);
		process.exit(1);
	}

	// Resolve model
	currentModel = resolveModel(currentModelId);
	if (!currentModel) {
		console.log(`\n  ${c.red}Model "${currentModelId}" not found.${c.reset}`);
		console.log(`  Check ${c.bold}RLM_MODEL${c.reset} in your .env file.\n`);
		process.exit(1);
	}

	printWelcome();

	const rl = readline.createInterface({
		input: stdin,
		output: stdout,
		prompt: `${c.cyan}>${c.reset} `,
		terminal: true,
	});

	rl.prompt();

	rl.on("line", async (rawLine: string) => {
		if (isRunning) return; // ignore input while a query is active
		const line = rawLine.trim();

		// URL auto-detect
		if (line.startsWith("http://") || line.startsWith("https://")) {
			const loaded = await detectAndLoadUrl(line);
			if (loaded) {
				printStatusLine();
				console.log(`\n  ${c.dim}Now type your query...${c.reset}\n`);
				rl.prompt();
				return;
			}
		}

		// Multi-line paste detect
		if (isMultiLineInput(rawLine)) {
			const result = handleMultiLineAsContext(rawLine);
			if (result) {
				contextText = result.context;
				contextSource = "(pasted)";
				printStatusLine();
				console.log(`\n  ${c.dim}Now type your query...${c.reset}\n`);
				rl.prompt();
				return;
			}
		}

		if (!line) {
			rl.prompt();
			return;
		}

		// Slash commands
		if (line.startsWith("/")) {
			const [cmd, ...rest] = line.slice(1).split(/\s+/);
			const arg = rest.join(" ");

			switch (cmd) {
				case "help":
				case "h":
					printCommandHelp();
					break;
				case "file":
				case "f":
					await handleFile(arg);
					break;
				case "url":
				case "u":
					await handleUrl(arg);
					break;
				case "paste":
				case "p":
					await handlePaste(rl);
					break;
				case "context":
				case "ctx":
					handleContext();
					break;
				case "clear-context":
				case "cc":
					contextText = "";
					contextSource = "";
					console.log(`  ${c.green}✓${c.reset} Context cleared.`);
					break;
				case "trajectories":
				case "traj":
					handleTrajectories();
					break;
					case "clear":
					printWelcome();
					break;
				case "quit":
				case "q":
				case "exit":
					console.log(`\n  ${c.dim}Goodbye!${c.reset}\n`);
					process.exit(0);
					break;
				default:
					console.log(`  ${c.red}Unknown command: /${cmd}${c.reset}. Type ${c.yellow}/help${c.reset} for commands.`);
			}

			rl.prompt();
			return;
		}

		// @file shorthand
		let query = expandAtFiles(line);
		if (!query && line.startsWith("@")) {
			rl.prompt();
			return;
		}
		if (!query) query = line;

		// Inline URL detection — extract URL from query, fetch as context
		if (!contextText) {
			const urlInline = query.match(/(https?:\/\/\S+)/);
			if (urlInline) {
				const url = urlInline[1];
				const queryWithoutUrl = query.replace(url, "").trim();
				console.log(`  ${c.dim}Fetching ${url}...${c.reset}`);
				try {
					const resp = await fetch(url);
					if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`);
					contextText = await resp.text();
					contextSource = url;
					const lines = contextText.split("\n").length;
					console.log(
						`  ${c.green}✓${c.reset} Loaded ${c.bold}${contextText.length.toLocaleString()}${c.reset} chars (${lines.toLocaleString()} lines) from URL`
					);
					if (queryWithoutUrl) {
						query = queryWithoutUrl;
					} else {
						// URL only, no query — prompt for one
						printStatusLine();
						console.log(`\n  ${c.dim}Now type your query...${c.reset}\n`);
						rl.prompt();
						return;
					}
				} catch (err: any) {
					console.log(`  ${c.red}Failed to fetch URL: ${err.message}${c.reset}`);
					console.log(`  ${c.dim}Running query as-is...${c.reset}`);
				}
			}
		}

		// Auto-detect file paths
		if (!contextText) {
			const { filePath, query: extractedQuery } = extractFilePath(query);
			if (filePath) {
				contextText = fs.readFileSync(filePath, "utf-8");
				contextSource = path.basename(filePath);
				const lines = contextText.split("\n").length;
				console.log(
					`  ${c.green}✓${c.reset} Loaded ${c.bold}${contextText.length.toLocaleString()}${c.reset} chars (${lines} lines) from ${c.underline}${filePath}${c.reset}`
				);
				query = extractedQuery || query;
			}
		}

		// Run query
		await runQuery(query);

		printStatusLine();
		console.log();
		rl.prompt();
	});

	// Ctrl+C: abort running query, or double-tap to exit
	let lastSigint = 0;
	rl.on("SIGINT", () => {
		if (isRunning && activeAc) {
			activeSpinner?.stop();
			console.log(`\n  ${c.red}Stopped${c.reset}\n`);
			activeAc.abort();
			try { activeRepl?.shutdown(); } catch { /* ok */ }
			isRunning = false;
			lastSigint = 0;
		} else {
			const now = Date.now();
			if (now - lastSigint < 1000) {
				// Double Ctrl+C — exit
				console.log(`\n  ${c.dim}Goodbye!${c.reset}\n`);
				process.exit(0);
			}
			lastSigint = now;
			console.log(`\n  ${c.dim}Press Ctrl+C again to exit${c.reset}`);
			rl.prompt();
		}
	});

	rl.on("close", () => {
		console.log(`\n  ${c.dim}Goodbye!${c.reset}\n`);
		process.exit(0);
	});
}

interactive().catch((err) => {
	console.error(`${c.red}Fatal error:${c.reset}`, err);
	process.exit(1);
});
