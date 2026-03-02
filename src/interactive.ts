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
import * as os from "node:os";
import * as readline from "node:readline";
import { stdin, stdout } from "node:process";

// Global error handlers — prevent raw stack traces from leaking to terminal
process.on("uncaughtException", (err) => {
	console.error(`\n  \x1b[31mUnexpected error: ${err.message}\x1b[0m\n`);
	process.exit(1);
});
process.on("unhandledRejection", (err: any) => {
	console.error(`\n  \x1b[31mUnexpected error: ${err?.message || err}\x1b[0m\n`);
	process.exit(1);
});

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

const DEFAULT_MODEL = process.env.RLM_MODEL || "claude-sonnet-4-6";
const RLM_HOME = path.join(os.homedir(), ".rlm");
const TRAJ_DIR = path.join(RLM_HOME, "trajectories");
const W = Math.min(process.stdout.columns || 80, 100);

// ── Session state ───────────────────────────────────────────────────────────

let currentModelId = DEFAULT_MODEL;
let currentModel: Model<Api> | undefined;
let currentProviderName = "";
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

// Provider → env var mapping for well-known providers
const PROVIDER_KEYS: Record<string, string> = {
	anthropic: "ANTHROPIC_API_KEY",
	openai: "OPENAI_API_KEY",
	google: "GEMINI_API_KEY",
	"google-gemini-cli": "GEMINI_API_KEY",
	"google-vertex": "GOOGLE_VERTEX_API_KEY",
	groq: "GROQ_API_KEY",
	xai: "XAI_API_KEY",
	mistral: "MISTRAL_API_KEY",
	openrouter: "OPENROUTER_API_KEY",
	huggingface: "HUGGINGFACE_API_KEY",
	cerebras: "CEREBRAS_API_KEY",
};

// User-facing provider list for setup & /provider command
const SETUP_PROVIDERS = [
	{ name: "Anthropic", label: "Claude", env: "ANTHROPIC_API_KEY", piProvider: "anthropic" },
	{ name: "OpenAI", label: "GPT", env: "OPENAI_API_KEY", piProvider: "openai" },
	{ name: "Google", label: "Gemini", env: "GEMINI_API_KEY", piProvider: "google" },
	{ name: "Groq", label: "Groq", env: "GROQ_API_KEY", piProvider: "groq" },
	{ name: "xAI", label: "Grok", env: "XAI_API_KEY", piProvider: "xai" },
	{ name: "Mistral", label: "Mistral", env: "MISTRAL_API_KEY", piProvider: "mistral" },
	{ name: "OpenRouter", label: "OpenRouter", env: "OPENROUTER_API_KEY", piProvider: "openrouter" },
];

function providerEnvKey(provider: string): string {
	return PROVIDER_KEYS[provider] || `${provider.toUpperCase().replace(/-/g, "_")}_API_KEY`;
}

function detectProvider(): string {
	for (const provider of Object.keys(PROVIDER_KEYS)) {
		if (process.env[PROVIDER_KEYS[provider]]) return provider;
	}
	return "unknown";
}

function hasAnyApiKey(): boolean {
	return detectProvider() !== "unknown";
}

/** Returns the pi-ai provider name + model for a given model ID, searching all providers.
 *  Prioritises SETUP_PROVIDERS so e.g. "gpt-4o" resolves to "openai" not "azure-openai-responses". */
function resolveModelWithProvider(modelId: string): { model: Model<Api>; provider: string } | undefined {
	// First pass: check well-known (setup) providers
	const knownNames = new Set(SETUP_PROVIDERS.map((p) => p.piProvider));
	for (const provider of getProviders()) {
		if (!knownNames.has(provider)) continue;
		for (const m of getModels(provider)) {
			if (m.id === modelId) return { model: m, provider };
		}
	}
	// Second pass: all remaining providers
	for (const provider of getProviders()) {
		if (knownNames.has(provider)) continue;
		for (const m of getModels(provider)) {
			if (m.id === modelId) return { model: m, provider };
		}
	}
	return undefined;
}

/** Sensible default model per provider. */
const PROVIDER_DEFAULT_MODELS: Record<string, string> = {
	anthropic: "claude-sonnet-4-6",
	openai: "gpt-4o",
	google: "gemini-2.5-flash",
	groq: "llama-3.3-70b-versatile",
	xai: "grok-4",
	mistral: "mistral-large-latest",
	openrouter: "claude-sonnet-4-6",
};

/** Returns the recommended default model for a provider. */
function getDefaultModelForProvider(provider: string): string | undefined {
	const preferred = PROVIDER_DEFAULT_MODELS[provider];
	if (preferred) {
		const model = resolveModel(preferred);
		if (model) return preferred;
	}
	// Fallback: first non-excluded model
	const models = getModelsForProvider(provider);
	return models.length > 0 ? models[0].id : undefined;
}

/** Wrap rl.question with ESC-to-cancel. Returns user input or null on ESC/empty. */
function questionWithEsc(rlInstance: readline.Interface, promptText: string): Promise<string | null> {
	return new Promise((resolve) => {
		let escaped = false;
		const onKeypress = (_str: string | undefined, key: { name?: string } | undefined) => {
			if (key?.name === "escape" && !escaped) {
				escaped = true;
				stdin.removeListener("keypress", onKeypress);
				// Clear the prompt line visually
				process.stdout.write("\r\x1b[2K");
				// Programmatically submit empty to close the pending question
				rlInstance.write("\n");
			}
		};
		stdin.on("keypress", onKeypress);
		rlInstance.question(promptText, (answer) => {
			stdin.removeListener("keypress", onKeypress);
			resolve(escaped ? null : answer.trim() || null);
		});
	});
}

/** Prompt user for a provider's API key if not already set.
 *  Returns true (got key), false (empty input), or null (ESC pressed). */
async function promptForProviderKey(
	rlInstance: readline.Interface,
	providerInfo: { name: string; env: string }
): Promise<boolean | null> {
	if (process.env[providerInfo.env]) return true;

	const rawKey = await questionWithEsc(rlInstance, `  ${c.cyan}${providerInfo.env}:${c.reset} `);
	if (rawKey === null) return null; // ESC
	if (!rawKey) return false; // empty

	// Sanitize: strip newlines, control chars, whitespace
	const key = rawKey.replace(/[\r\n\x00-\x1f]/g, "").trim();
	if (!key) return false;

	process.env[providerInfo.env] = key;

	// Save to ~/.rlm/credentials (persistent across sessions)
	const credPath = path.join(RLM_HOME, "credentials");
	try {
		if (!fs.existsSync(RLM_HOME)) fs.mkdirSync(RLM_HOME, { recursive: true });
		fs.appendFileSync(credPath, `${providerInfo.env}=${key}\n`);
		// Restrict permissions (owner-only read/write)
		try { fs.chmodSync(credPath, 0o600); } catch { /* Windows etc. */ }
		console.log(`\n  ${c.green}✓${c.reset} ${providerInfo.name} key saved to ${c.dim}~/.rlm/credentials${c.reset}`);
	} catch {
		console.log(`\n  ${c.yellow}!${c.reset} Could not save key. Add manually:`);
		console.log(`    ${c.yellow}export ${providerInfo.env}=${key}${c.reset}`);
	}
	return true;
}

/** Find the SETUP_PROVIDERS entry that owns a given pi-ai provider name. */
function findSetupProvider(piProvider: string): (typeof SETUP_PROVIDERS)[number] | undefined {
	return SETUP_PROVIDERS.find((p) => p.piProvider === piProvider);
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
	const provider = currentProviderName || detectProvider();
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
${c.bold}Loading Context${c.reset}
  ${c.cyan}/file${c.reset} <path>              Load a single file
  ${c.cyan}/file${c.reset} <p1> <p2> ...       Load multiple files
  ${c.cyan}/file${c.reset} <dir>/              Load all files in a directory (recursive)
  ${c.cyan}/file${c.reset} src/**/*.ts         Load files matching a glob pattern
  ${c.cyan}/url${c.reset} <url>                Fetch URL as context
  ${c.cyan}/paste${c.reset}                    Multi-line paste mode (type EOF to finish)
  ${c.cyan}/context${c.reset}                  Show loaded context info + file list
  ${c.cyan}/clear-context${c.reset}            Unload context

${c.bold}@ Shorthand${c.reset}  ${c.dim}(inline file loading)${c.reset}
  ${c.cyan}@file.ts${c.reset} <query>           Load file and ask in one shot
  ${c.cyan}@a.ts @b.ts${c.reset} <query>        Load multiple files + query
  ${c.cyan}@src/${c.reset} <query>              Load directory + query
  ${c.cyan}@src/**/*.ts${c.reset} <query>       Load glob + query

${c.bold}Model & Provider${c.reset}
  ${c.cyan}/model${c.reset}                    List models for current provider
  ${c.cyan}/model${c.reset} <#|id>              Switch model by number or ID
  ${c.cyan}/provider${c.reset}                 Switch provider (Anthropic, OpenAI, Google, ...)

${c.bold}Tools${c.reset}
  ${c.cyan}/trajectories${c.reset}             List saved runs

${c.bold}General${c.reset}
  ${c.cyan}/clear${c.reset}                    Clear screen
  ${c.cyan}/help${c.reset}                     Show this help
  ${c.cyan}/quit${c.reset}                     Exit

${c.bold}Tips${c.reset}
  ${c.dim}•${c.reset} Just type a question — no context needed for general queries
  ${c.dim}•${c.reset} Paste a URL directly to fetch it as context
  ${c.dim}•${c.reset} Paste 4+ lines of text to set it as context
  ${c.dim}•${c.reset} ${c.bold}Ctrl+C${c.reset} stops a running query, ${c.bold}Ctrl+C twice${c.reset} exits
  ${c.dim}•${c.reset} Directories skip node_modules, .git, dist, binaries, etc.
  ${c.dim}•${c.reset} Limits: ${MAX_FILES} files max, ${MAX_TOTAL_BYTES / 1024 / 1024}MB total
`);
}

// ── Slash command handlers ──────────────────────────────────────────────────

async function handleFile(arg: string): Promise<void> {
	if (!arg) {
		console.log(`  ${c.red}Usage: /file <path|dir|glob> [...]${c.reset}`);
		console.log(`  ${c.dim}Examples: /file src/main.ts  |  /file src/  |  /file src/**/*.ts${c.reset}`);
		return;
	}
	const args = arg.split(/\s+/).filter(Boolean);
	const filePaths = resolveFileArgs(args);

	if (filePaths.length === 0) {
		console.log(`  ${c.red}No files found.${c.reset}`);
		return;
	}

	if (filePaths.length === 1) {
		try {
			contextText = fs.readFileSync(filePaths[0], "utf-8");
			contextSource = path.relative(process.cwd(), filePaths[0]) || filePaths[0];
			const lines = contextText.split("\n").length;
			console.log(
				`  ${c.green}✓${c.reset} Loaded ${c.bold}${contextText.length.toLocaleString()}${c.reset} chars (${lines.toLocaleString()} lines) from ${c.underline}${contextSource}${c.reset}`
			);
		} catch (err: any) {
			console.log(`  ${c.red}Could not read file: ${err.message}${c.reset}`);
		}
	} else {
		const { text, count, totalBytes } = loadMultipleFiles(filePaths);
		contextText = text;
		contextSource = `${count} files`;
		console.log(
			`  ${c.green}✓${c.reset} Loaded ${c.bold}${count}${c.reset} files (${(totalBytes / 1024).toFixed(1)}KB total)`
		);
		// Show file list
		for (const fp of filePaths.slice(0, 20)) {
			console.log(`    ${c.dim}•${c.reset} ${path.relative(process.cwd(), fp)}`);
		}
		if (filePaths.length > 20) {
			console.log(`    ${c.dim}... and ${filePaths.length - 20} more${c.reset}`);
		}
	}
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
		console.log(`  ${c.dim}No context loaded. Use /file, /url, @file, or /paste.${c.reset}`);
		return;
	}
	const lines = contextText.split("\n").length;
	const sizeKB = (contextText.length / 1024).toFixed(1);
	console.log(`  ${c.bold}Context:${c.reset} ${contextText.length.toLocaleString()} chars (${sizeKB}KB), ${lines.toLocaleString()} lines`);
	console.log(`  ${c.bold}Source:${c.reset}  ${contextSource}`);

	// For multi-file context, extract and display individual file paths
	const fileSeparators = contextText.match(/^=== .+ ===$/gm);
	if (fileSeparators && fileSeparators.length > 1) {
		console.log(`  ${c.bold}Files:${c.reset}   ${fileSeparators.length}`);
		for (const sep of fileSeparators.slice(0, 20)) {
			const name = sep.replace(/^=== /, "").replace(/ ===$/, "");
			console.log(`    ${c.dim}•${c.reset} ${name}`);
		}
		if (fileSeparators.length > 20) {
			console.log(`    ${c.dim}... and ${fileSeparators.length - 20} more${c.reset}`);
		}
	} else {
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

// ── Available models list ────────────────────────────────────────────────────

/** Filter out deprecated, retired, and non-chat models (Feb 2026). */
const EXCLUDED_MODEL_PATTERNS = [
	// ── Anthropic retired / old gen ──
	/^claude-3-/,                // all claude 3.x retired (haiku, sonnet, opus, 3-5-*, 3-7-*)
	// ── OpenAI legacy / specialized ──
	/^gpt-4$/,                   // superseded by gpt-4.1
	/^gpt-4-turbo/,              // superseded by gpt-4.1
	/^gpt-4o-2024-/,             // dated snapshots
	/-chat-latest$/,             // chat variants (use base model)
	/^codex-/,                   // code-only
	/-codex/,                    // all codex variants
	// ── Google retired / deprecated ──
	/^gemini-1\.5-/,             // all 1.5 retired
	/^gemini-3-pro-preview$/,    // deprecated, shuts down Mar 9, 2026
	/^gemini-live-/,             // real-time streaming, not standard chat
	// ── xAI non-chat ──
	/^grok-beta$/,
	/^grok-vision-beta$/,
	/^grok-2-vision/,
	/^grok-2-1212$/,             // dated snapshot
	// ── Mistral legacy ──
	/^open-mistral-7b$/,
	/^open-mixtral-/,
	/^mistral-nemo$/,
	// ── Dated snapshots / previews ──
	/preview-\d{2}-\d{2}$/,      // e.g. preview-04-17
	/preview-\d{2}-\d{4}$/,      // e.g. preview-09-2025
	/^labs-/,
	/-customtools$/,
	/deep-research$/,
	// Mistral dated snapshots (use -latest instead)
	/^mistral-large-\d{4}$/,
	/^mistral-medium-\d{4}$/,
	/^mistral-small-\d{4}$/,
	/^devstral-\d{4}$/,
	/^devstral-\w+-\d{4}$/,
	// Groq dated snapshots
	/kimi-k2-instruct-\d+$/,
];

function isModelExcluded(modelId: string): boolean {
	return EXCLUDED_MODEL_PATTERNS.some((p) => p.test(modelId));
}

/** Collect models from providers that have an API key set. */
function getAvailableModels(): { id: string; provider: string }[] {
	const items: { id: string; provider: string }[] = [];
	for (const provider of getProviders()) {
		const key = providerEnvKey(provider);
		if (!process.env[key] && provider !== detectProvider()) continue;
		for (const m of getModels(provider)) {
			if (!isModelExcluded(m.id)) items.push({ id: m.id, provider });
		}
	}
	return items;
}

/** Get models for a specific provider (matching by pi-ai provider name or SETUP_PROVIDERS piProvider). */
function getModelsForProvider(providerName: string): { id: string; provider: string }[] {
	const items: { id: string; provider: string }[] = [];
	for (const provider of getProviders()) {
		if (provider !== providerName) continue;
		for (const m of getModels(provider)) {
			if (!isModelExcluded(m.id)) items.push({ id: m.id, provider });
		}
	}
	return items;
}

// ── Truncate helper ─────────────────────────────────────────────────────────

function truncateStr(text: string, max: number): string {
	return text.length <= max ? text : text.slice(0, max - 3) + "...";
}

// ── Multi-file context loading ──────────────────────────────────────────────

const MAX_FILES = 100;
const MAX_TOTAL_BYTES = 10 * 1024 * 1024; // 10MB

const BINARY_EXTENSIONS = new Set([
	".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".svg",
	".mp3", ".mp4", ".wav", ".ogg", ".flac", ".avi", ".mov", ".mkv",
	".zip", ".gz", ".tar", ".bz2", ".7z", ".rar", ".xz",
	".exe", ".dll", ".so", ".dylib", ".bin", ".o", ".a",
	".woff", ".woff2", ".ttf", ".otf", ".eot",
	".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
	".pyc", ".pyo", ".class", ".jar",
	".db", ".sqlite", ".sqlite3",
	".DS_Store",
]);

const SKIP_DIRS = new Set([
	"node_modules", ".git", "dist", "build", "__pycache__", ".venv",
	"venv", ".next", ".nuxt", "coverage", ".cache", ".tsc-output",
	".svelte-kit", "target", "out",
]);

function isBinaryFile(filePath: string): boolean {
	const ext = path.extname(filePath).toLowerCase();
	if (BINARY_EXTENSIONS.has(ext)) return true;
	// Quick null-byte check on first 512 bytes
	try {
		const fd = fs.openSync(filePath, "r");
		const buf = Buffer.alloc(512);
		const bytesRead = fs.readSync(fd, buf, 0, 512, 0);
		fs.closeSync(fd);
		for (let i = 0; i < bytesRead; i++) {
			if (buf[i] === 0) return true;
		}
	} catch { /* unreadable → skip */ return true; }
	return false;
}

function walkDir(dir: string): string[] {
	const results: string[] = [];
	let entries: fs.Dirent[];
	try {
		entries = fs.readdirSync(dir, { withFileTypes: true });
	} catch { return results; }

	for (const entry of entries) {
		if (entry.name.startsWith(".") && entry.name !== ".env") continue;
		const full = path.join(dir, entry.name);
		if (entry.isDirectory()) {
			if (SKIP_DIRS.has(entry.name)) continue;
			results.push(...walkDir(full));
		} else if (entry.isFile()) {
			if (!isBinaryFile(full)) results.push(full);
		}
		if (results.length > MAX_FILES) break;
	}
	return results;
}

function simpleGlobMatch(pattern: string, filePath: string): boolean {
	// Expand {a,b,c} braces into alternatives
	const braceMatch = pattern.match(/\{([^}]+)\}/);
	if (braceMatch) {
		const alternatives = braceMatch[1].split(",");
		return alternatives.some((alt) =>
			simpleGlobMatch(pattern.replace(braceMatch[0], alt.trim()), filePath)
		);
	}

	// Convert glob to regex
	let regex = "^";
	let i = 0;
	while (i < pattern.length) {
		const ch = pattern[i];
		if (ch === "*" && pattern[i + 1] === "*") {
			// ** matches any path segment(s)
			regex += ".*";
			i += 2;
			if (pattern[i] === "/") i++; // skip trailing slash after **
		} else if (ch === "*") {
			regex += "[^/]*";
			i++;
		} else if (ch === "?") {
			regex += "[^/]";
			i++;
		} else if (".+^$|()[]\\".includes(ch)) {
			regex += "\\" + ch;
			i++;
		} else {
			regex += ch;
			i++;
		}
	}
	regex += "$";
	return new RegExp(regex).test(filePath);
}

function resolveFileArgs(args: string[]): string[] {
	const files: string[] = [];
	for (const arg of args) {
		const resolved = path.resolve(arg);

		// Glob pattern (contains * or ?)
		if (arg.includes("*") || arg.includes("?")) {
			// Find the base directory (portion before the first glob char)
			const firstGlob = arg.search(/[*?{]/);
			const baseDir = firstGlob > 0 ? path.resolve(arg.slice(0, arg.lastIndexOf("/", firstGlob) + 1) || ".") : process.cwd();
			const allFiles = walkDir(baseDir);
			for (const f of allFiles) {
				const rel = path.relative(process.cwd(), f);
				if (simpleGlobMatch(arg, rel) || simpleGlobMatch(arg, f)) {
					files.push(f);
				}
			}
			continue;
		}

		// Directory
		if (fs.existsSync(resolved) && fs.statSync(resolved).isDirectory()) {
			files.push(...walkDir(resolved));
			continue;
		}

		// Regular file
		if (fs.existsSync(resolved)) {
			if (!isBinaryFile(resolved)) files.push(resolved);
			continue;
		}

		console.log(`  ${c.yellow}⚠${c.reset} Not found: ${arg}`);
	}
	return [...new Set(files)]; // deduplicate
}

function loadMultipleFiles(filePaths: string[]): { text: string; count: number; totalBytes: number } {
	if (filePaths.length > MAX_FILES) {
		console.log(`  ${c.yellow}⚠${c.reset} Too many files (${filePaths.length}). Limit is ${MAX_FILES}.`);
		filePaths = filePaths.slice(0, MAX_FILES);
	}

	const parts: string[] = [];
	let totalBytes = 0;

	for (const fp of filePaths) {
		try {
			const content = fs.readFileSync(fp, "utf-8");
			if (totalBytes + content.length > MAX_TOTAL_BYTES) {
				console.log(`  ${c.yellow}⚠${c.reset} Size limit reached (${(MAX_TOTAL_BYTES / 1024 / 1024).toFixed(0)}MB). Loaded ${parts.length} of ${filePaths.length} files.`);
				break;
			}
			const rel = path.relative(process.cwd(), fp);
			parts.push(`=== ${rel} ===\n${content}`);
			totalBytes += content.length;
		} catch { /* skip unreadable */ }
	}

	return { text: parts.join("\n\n"), count: parts.length, totalBytes };
}

// ── Run RLM query ───────────────────────────────────────────────────────────

async function runQuery(query: string): Promise<void> {
	const effectiveContext = contextText || query;
	const isDirectMode = !contextText;

	if (!currentModel) {
		const resolved = resolveModelWithProvider(currentModelId);
		if (resolved) {
			currentModel = resolved.model;
			currentProviderName = resolved.provider;
		}
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
		try {
			if (!fs.existsSync(TRAJ_DIR)) fs.mkdirSync(TRAJ_DIR, { recursive: true });
			const ts = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
			const trajFile = `trajectory-${ts}.json`;
			fs.writeFileSync(path.join(TRAJ_DIR, trajFile), JSON.stringify(trajectory, null, 2), "utf-8");
			console.log(`  ${c.dim}Saved: ~/.rlm/trajectories/${trajFile}${c.reset}\n`);
		} catch {
			console.log(`  ${c.yellow}Could not save trajectory.${c.reset}\n`);
		}
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
	// Extract all @tokens from input
	const tokens: string[] = [];
	const remaining: string[] = [];

	for (const part of input.split(/\s+/)) {
		if (part.startsWith("@") && part.length > 1) {
			tokens.push(part.slice(1));
		} else {
			remaining.push(part);
		}
	}

	if (tokens.length === 0) return input;

	const filePaths = resolveFileArgs(tokens);
	if (filePaths.length === 0) {
		console.log(`  ${c.red}No files found for: ${tokens.join(", ")}${c.reset}`);
		return "";
	}

	if (filePaths.length === 1) {
		// Single file — simple load
		try {
			contextText = fs.readFileSync(filePaths[0], "utf-8");
			contextSource = path.relative(process.cwd(), filePaths[0]) || filePaths[0];
			const lines = contextText.split("\n").length;
			console.log(
				`  ${c.green}✓${c.reset} Loaded ${c.bold}${contextText.length.toLocaleString()}${c.reset} chars (${lines} lines) from ${c.underline}${contextSource}${c.reset}`
			);
		} catch (err: any) {
			console.log(`  ${c.red}Could not read file: ${err.message}${c.reset}`);
			return "";
		}
	} else {
		// Multiple files — concatenate with separators
		const { text, count, totalBytes } = loadMultipleFiles(filePaths);
		contextText = text;
		contextSource = `${count} files`;
		console.log(
			`  ${c.green}✓${c.reset} Loaded ${c.bold}${count}${c.reset} files (${(totalBytes / 1024).toFixed(1)}KB total)`
		);
	}

	return remaining.join(" ");
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
	if (!hasAnyApiKey()) {
		printBanner();
		console.log(`  ${c.bold}Welcome! Let's get you set up.${c.reset}\n`);

		const setupRl = readline.createInterface({ input: stdin, output: stdout, terminal: true });
		let setupDone = false;

		while (!setupDone) {
			console.log(`  ${c.bold}Select your provider:${c.reset}\n`);
			for (let i = 0; i < SETUP_PROVIDERS.length; i++) {
				console.log(`  ${c.dim}${i + 1}${c.reset}  ${SETUP_PROVIDERS[i].name} ${c.dim}(${SETUP_PROVIDERS[i].label})${c.reset}`);
			}
			console.log();

			const choice = await questionWithEsc(setupRl, `  ${c.cyan}Provider [1-${SETUP_PROVIDERS.length}]:${c.reset} `);
			if (choice === null) {
				// ESC at provider selection → exit
				console.log(`\n  ${c.dim}Exiting.${c.reset}\n`);
				setupRl.close();
				process.exit(0);
			}
			const idx = parseInt(choice, 10) - 1;
			if (isNaN(idx) || idx < 0 || idx >= SETUP_PROVIDERS.length) {
				console.log(`\n  ${c.dim}Invalid choice.${c.reset}\n`);
				continue;
			}

			const provider = SETUP_PROVIDERS[idx];
			const gotKey = await promptForProviderKey(setupRl, provider);
			if (gotKey === null) {
				// ESC at key entry → back to provider selection
				console.log();
				continue;
			}
			if (!gotKey) {
				console.log(`\n  ${c.dim}No key provided. Exiting.${c.reset}\n`);
				setupRl.close();
				process.exit(0);
			}

			// Auto-select default model for chosen provider
			const defaultModel = getDefaultModelForProvider(provider.piProvider);
			if (defaultModel) {
				currentModelId = defaultModel;
				console.log(`  ${c.green}✓${c.reset} Default model: ${c.bold}${currentModelId}${c.reset}`);
			}
			console.log();
			setupDone = true;
		}
		setupRl.close();
	}

	// Resolve model
	const initialResolved = resolveModelWithProvider(currentModelId);
	if (initialResolved) {
		currentModel = initialResolved.model;
		currentProviderName = initialResolved.provider;
	}
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

	// Color slash commands cyan as the user types
	const rlAny = rl as any;
	const promptStr = rl.getPrompt();
	rlAny._writeToOutput = function (str: string) {
		if (!rlAny.line?.startsWith("/")) {
			rlAny.output.write(str);
			return;
		}
		if (str.startsWith(promptStr)) {
			rlAny.output.write(promptStr + c.cyan + str.slice(promptStr.length) + c.reset);
		} else {
			rlAny.output.write(c.cyan + str + c.reset);
		}
	};

	rl.prompt();

	rl.on("line", async (rawLine: string) => {
	  try {
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
				case "model":
				case "m": {
					const curProvider = detectProvider();
					if (arg) {
						// Accept a number (from current provider list) or a model ID
						const curModels = getModelsForProvider(curProvider);
						let pick: string | undefined;
						if (/^\d+$/.test(arg)) {
							pick = curModels[parseInt(arg, 10) - 1]?.id;
						} else {
							pick = arg;
						}
						if (!pick) {
							console.log(`  ${c.red}Invalid selection.${c.reset} Use ${c.cyan}/model${c.reset} to list available models.`);
							break;
						}

						// Check if this model belongs to a different provider
						const resolved = resolveModelWithProvider(pick);
						if (!resolved) {
							console.log(`  ${c.red}Model "${arg}" not found.${c.reset} Use ${c.cyan}/model${c.reset} to list available models.`);
							break;
						}

						if (resolved.provider !== curProvider) {
							// Cross-provider switch
							const setupInfo = findSetupProvider(resolved.provider);
							const envVar = setupInfo?.env || providerEnvKey(resolved.provider);
							const provName = setupInfo?.name || resolved.provider;

							if (!process.env[envVar]) {
								console.log(`  ${c.yellow}That model requires ${provName}.${c.reset}`);
								const gotKey = await promptForProviderKey(rl, { name: provName, env: envVar });
								if (!gotKey) {
									console.log(`  ${c.dim}Cancelled.${c.reset}`);
									break;
								}
							}
						}

						currentModelId = pick;
						currentModel = resolved.model;
						currentProviderName = resolved.provider;
						console.log(`  ${c.green}✓${c.reset} Switched to ${c.bold}${currentModelId}${c.reset}`);
						console.log();
						printStatusLine();
					} else {
						// List models for current provider
						const models = getModelsForProvider(curProvider);
						const provLabel = findSetupProvider(curProvider)?.name || curProvider;
						console.log(`\n  ${c.bold}Current model:${c.reset} ${c.cyan}${currentModelId}${c.reset} ${c.dim}(${provLabel})${c.reset}\n`);
						const pad = String(models.length).length;
						for (let i = 0; i < models.length; i++) {
							const m = models[i];
							const num = String(i + 1).padStart(pad);
							const dot = m.id === currentModelId ? `${c.green}●${c.reset}` : ` `;
							const label = m.id === currentModelId
								? `${c.cyan}${m.id}${c.reset}`
								: `${c.dim}${m.id}${c.reset}`;
							console.log(`  ${c.dim}${num}${c.reset} ${dot} ${label}`);
						}
						console.log(`\n  ${c.dim}${models.length} models · scroll up to see full list.${c.reset}`);
						console.log(`  ${c.dim}Type${c.reset} ${c.cyan}/model <number>${c.reset} ${c.dim}or${c.reset} ${c.cyan}/model <id>${c.reset} ${c.dim}to switch.${c.reset}`);
						console.log(`  ${c.dim}Type${c.reset} ${c.cyan}/provider${c.reset} ${c.dim}to switch provider.${c.reset}`);
					}
					break;
				}
				case "provider":
				case "prov": {
					const curProvider = detectProvider();
					const curLabel = findSetupProvider(curProvider)?.name || curProvider;
					console.log(`\n  ${c.bold}Current provider:${c.reset} ${c.cyan}${curLabel}${c.reset}\n`);

					for (let i = 0; i < SETUP_PROVIDERS.length; i++) {
						const p = SETUP_PROVIDERS[i];
						const isCurrent = p.piProvider === curProvider;
						const dot = isCurrent ? `${c.green}●${c.reset}` : ` `;
						// Only show ✓ for non-current providers that have a key
						const hasKey = !isCurrent && process.env[p.env] ? `${c.green}✓${c.reset}` : ` `;
						const label = isCurrent
							? `${c.cyan}${p.name}${c.reset} ${c.dim}(${p.label})${c.reset}`
							: `${p.name} ${c.dim}(${p.label})${c.reset}`;
						console.log(`  ${c.dim}${i + 1}${c.reset} ${dot}${hasKey} ${label}`);
					}
					console.log();

					const provChoice = await questionWithEsc(rl, `  ${c.cyan}Provider [1-${SETUP_PROVIDERS.length}]:${c.reset} ${c.dim}(ESC to cancel)${c.reset} `);
					if (provChoice === null) break; // ESC
					const idx = parseInt(provChoice, 10) - 1;
					if (isNaN(idx) || idx < 0 || idx >= SETUP_PROVIDERS.length) {
						console.log(`  ${c.dim}Cancelled.${c.reset}`);
						break;
					}

					const chosen = SETUP_PROVIDERS[idx];
					const gotKey = await promptForProviderKey(rl, chosen);

					if (!gotKey) {
						// null (ESC) or false (empty) → cancel
						break;
					}

					// Auto-select first model from new provider
					const defaultModel = getDefaultModelForProvider(chosen.piProvider);
					if (defaultModel) {
						currentModelId = defaultModel;
						const provResolved = resolveModelWithProvider(currentModelId);
						currentModel = provResolved?.model;
						currentProviderName = chosen.piProvider;
						console.log(`  ${c.green}✓${c.reset} Switched to ${c.bold}${chosen.name}${c.reset}`);
						console.log(`  ${c.green}✓${c.reset} Default model: ${c.bold}${currentModelId}${c.reset}`);
						console.log();
						printStatusLine();
					} else {
						console.log(`  ${c.red}No models available for ${chosen.name}.${c.reset}`);
					}
					break;
				}
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
					console.log(`  ${c.red}Unknown command: /${cmd}${c.reset}. Type ${c.cyan}/help${c.reset} for commands.`);
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
	  } catch (err: any) {
		console.log(`\n  ${c.red}Error: ${err?.message || err}${c.reset}\n`);
		rl.prompt();
	  }
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
