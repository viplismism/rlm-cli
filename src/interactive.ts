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
import { fileURLToPath } from "node:url";
import {
	fetchOllamaModels, createOllamaModel, formatOllamaSize, OLLAMA_DEFAULT_BASE_URL,
} from "./ollama.js";
import {
	RESET, BOLD, DIM,
	AMBER, AMBER_DIM, SAGE, ICE, LAVENDER, STONE, ASH, DARK_ASH, ROSE,
	paint, printPanel,
} from "./colors.js";
import {
	buildStatusBar,
	joinColumns,
	padAnsiRight,
	renderCard,
	visibleWidth,
} from "./ui.js";
// Global error handlers — prevent raw stack traces from leaking to terminal
process.on("uncaughtException", (err) => {
	cleanupShell();
	console.error(`\n  \x1b[31mUnexpected error: ${err.message}\x1b[0m\n`);
	process.exit(1);
});
process.on("unhandledRejection", (err: any) => {
	cleanupShell();
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
	reset:     RESET,
	bold:      BOLD,
	dim:       DIM,
	italic:    "\x1b[3m",
	underline: "\x1b[4m",
	// ── rlm identity: Electric Amber (true RGB) ────────────────────
	accent:    AMBER,        // electric amber — primary
	accentDim: AMBER_DIM,    // deep amber — secondary
	result:    SAGE,         // soft green — success / result
	code:      ICE,          // ice blue — code blocks
	subquery:  LAVENDER,     // soft lavender — sub-queries
	// ── semantic aliases ───────────────────────────────────────────
	cyan:      AMBER,        // alias → accent (amber) for backward compat
	green:     SAGE,         // alias → result
	magenta:   LAVENDER,     // alias → subquery
	// ── stable colors ─────────────────────────────────────────────
	red:       ROSE,
	yellow:    STONE,
	blue:      ICE,
	white:     "\x1b[37m",
	gray:      ASH,
	clearLine: "\x1b[2K\r",
	bgPanel:   "\x1b[48;5;236m",
	bgPanelAlt:"\x1b[48;5;234m",
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
const SESSIONS_DIR = path.join(RLM_HOME, "sessions");

// Per-session folder — all trajectories for this run go here
const SESSION_ID = `session-${new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19)}`;
const SESSION_DIR = path.join(SESSIONS_DIR, SESSION_ID);

let W = Math.min(process.stdout.columns || 80, 100);

// ── Session state ───────────────────────────────────────────────────────────

let currentModelId = DEFAULT_MODEL;
let currentModel: Model<Api> | undefined;
let currentProviderName = "";
let contextText = "";
let contextSource = "";
let contextIsDefault = true;
let queryCount = 0;
let isRunning = false;

// Exposed so the readline SIGINT handler can abort the running query
let activeAc: AbortController | null = null;
let activeRepl: InstanceType<typeof PythonRepl> | null = null;
let activeSpinner: Spinner | null = null;

// ── Ollama local model registry ──────────────────────────────────────────────

// Maps "ollama:<name>" (and bare name) → Model for Ollama-hosted models
const ollamaModelMap = new Map<string, Model<"openai-completions">>();
const ollamaBaseUrl = process.env.OLLAMA_BASE_URL || OLLAMA_DEFAULT_BASE_URL;
let ollamaSizes: Map<string, number> = new Map(); // name → bytes

async function loadOllamaModels(): Promise<void> {
	const models = await fetchOllamaModels(ollamaBaseUrl);
	ollamaModelMap.clear();
	ollamaSizes.clear();
	for (const m of models) {
		const mdl = createOllamaModel(m.name, ollamaBaseUrl);
		ollamaModelMap.set(`ollama:${m.name}`, mdl);
		ollamaModelMap.set(m.name, mdl); // bare name also works (colons are unique to Ollama)
		ollamaSizes.set(m.name, m.size);
	}
}

// ── Resolve model ───────────────────────────────────────────────────────────

function resolveModel(modelId: string): Model<Api> | undefined {
	const result = resolveModelWithProvider(modelId);
	return result?.model;
}

// Provider → env var mapping for well-known providers
const PROVIDER_KEYS: Record<string, string> = {
	anthropic: "ANTHROPIC_API_KEY",
	openai: "OPENAI_API_KEY",
	google: "GEMINI_API_KEY",
	openrouter: "OPENROUTER_API_KEY",
};

// User-facing provider list for setup & /provider command
const SETUP_PROVIDERS = [
	{ name: "Anthropic", label: "Claude", env: "ANTHROPIC_API_KEY", piProvider: "anthropic" },
	{ name: "OpenAI", label: "GPT", env: "OPENAI_API_KEY", piProvider: "openai" },
	{ name: "Google", label: "Gemini", env: "GEMINI_API_KEY", piProvider: "google" },
	{ name: "OpenRouter", label: "Multi-provider", env: "OPENROUTER_API_KEY", piProvider: "openrouter" },
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
	return detectProvider() !== "unknown" || ollamaModelMap.size > 0;
}

/** Returns the pi-ai provider name + model for a given model ID, searching all providers.
 *  Prioritises SETUP_PROVIDERS (with API key) so e.g. "gpt-4o" resolves to "openai" not "azure-openai-responses".
 *  Checks local Ollama models first for `ollama:` prefix or bare Ollama model names. */
function resolveModelWithProvider(modelId: string): { model: Model<Api>; provider: string } | undefined {
	// Ollama: explicit prefix "ollama:<name>" or bare Ollama model name (e.g. "llama3.1:8b")
	const ollamaKey = modelId.startsWith("ollama:") ? modelId : (ollamaModelMap.has(modelId) ? modelId : null);
	if (ollamaKey) {
		const m = ollamaModelMap.get(ollamaKey);
		if (m) return { model: m as unknown as Model<Api>, provider: "ollama" };
	}
	const knownNames = new Set(SETUP_PROVIDERS.map((p) => p.piProvider));
	let firstMatch: { model: Model<Api>; provider: string } | undefined;

	// First pass: well-known providers that have an API key set (best match)
	for (const provider of getProviders()) {
		if (!knownNames.has(provider)) continue;
		for (const m of getModels(provider)) {
			if (m.id === modelId) {
				if (process.env[providerEnvKey(provider)]) {
					return { model: m, provider };
				}
				if (!firstMatch) firstMatch = { model: m, provider };
			}
		}
	}
	// Second pass: well-known providers without key (user may enter it later)
	if (firstMatch) return firstMatch;

	// Third pass: all remaining providers
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
	openrouter: "auto",
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

/** Wrap rl.question with ESC-to-cancel. Returns user input, empty string, or null on ESC. */
function questionWithEsc(rlInstance: readline.Interface, promptText: string): Promise<string | null> {
	return new Promise((resolve) => {
		let escaped = false;

		const onKeypress = (_str: string | undefined, key: { name?: string } | undefined) => {
			if (key?.name === "escape" && !escaped) {
				escaped = true;
				stdin.removeListener("keypress", onKeypress);
				process.stdout.write("\r\x1b[2K");
				rlInstance.write("\n");
			}
		};
		stdin.on("keypress", onKeypress);
		rlInstance.question(promptText, (answer) => {
			stdin.removeListener("keypress", onKeypress);
			resolve(escaped ? null : answer.trim());
		});
	});
}

/** Prompt user for a provider's API key (only if not already set).
 *  Returns true (got key / already set), false (empty input), or null (ESC pressed). */
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

	// Save to ~/.rlm/credentials (persistent across sessions, replaces existing key)
	const credPath = path.join(RLM_HOME, "credentials");
	try {
		if (!fs.existsSync(RLM_HOME)) fs.mkdirSync(RLM_HOME, { recursive: true });
		// Remove existing entry for this key to avoid duplicates
		if (fs.existsSync(credPath)) {
			const existing = fs.readFileSync(credPath, "utf-8");
			const filtered = existing.split("\n").filter((l) => {
				const t = l.trim();
				if (t.startsWith("export ")) return !t.slice(7).startsWith(providerInfo.env + "=");
				return !t.startsWith(providerInfo.env + "=");
			}).join("\n");
			fs.writeFileSync(credPath, filtered.endsWith("\n") ? filtered : filtered + "\n");
		}
		fs.appendFileSync(credPath, `${providerInfo.env}=${key}\n`);
		// Restrict permissions (owner-only read/write)
		try { fs.chmodSync(credPath, 0o600); } catch { /* Windows etc. */ }
		console.log(`\n  ${c.green}✓${c.reset} ${providerInfo.name} key saved to ${c.dim}~/.rlm/credentials${c.reset}`);
	} catch {
		console.log(`\n  ${c.yellow}!${c.reset} Could not save key. Add manually:`);
		console.log(`    ${c.yellow}export ${providerInfo.env}=<your-key>${c.reset}`);
	}
	return true;
}

/** Persist the user's model choice to ~/.rlm/credentials so it survives restarts. */
function saveModelPreference(modelId: string): void {
	const credPath = path.join(RLM_HOME, "credentials");
	try {
		if (!fs.existsSync(RLM_HOME)) fs.mkdirSync(RLM_HOME, { recursive: true });
		// Remove existing RLM_MODEL entry
		if (fs.existsSync(credPath)) {
			const existing = fs.readFileSync(credPath, "utf-8");
			const filtered = existing.split("\n").filter((l) => {
				const t = l.trim();
				if (t.startsWith("export ")) return !t.slice(7).startsWith("RLM_MODEL=");
				return !t.startsWith("RLM_MODEL=");
			}).join("\n");
			fs.writeFileSync(credPath, filtered.endsWith("\n") ? filtered : filtered + "\n");
		}
		fs.appendFileSync(credPath, `RLM_MODEL=${modelId}\n`);
		try { fs.chmodSync(credPath, 0o600); } catch {}
	} catch { /* best-effort */ }
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
	// Wide pixel-art RLM logo — amber on black
	const a = `${AMBER}${BOLD}`;
	const r = RESET;
	console.log(`
${a}    ██████╗ ██╗     ███╗   ███╗${r}
${a}    ██╔══██╗██║     ████╗ ████║${r}
${a}    ██████╔╝██║     ██╔████╔██║${r}
${a}    ██╔══██╗██║     ██║╚██╔╝██║${r}
${a}    ██║  ██║███████╗██║ ╚═╝ ██║${r}
${a}    ╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝${r}
${paint("    recursive language models · arXiv:2512.24601", DIM)}
`);
}

// ── Version ──────────────────────────────────────────────────────────────────

function loadVersion(): string {
	try {
		const __dir = path.dirname(fileURLToPath(import.meta.url));
		const pkgPath = path.join(__dir, "..", "package.json");
		const pkg = JSON.parse(fs.readFileSync(pkgPath, "utf-8")) as { version?: string };
		return pkg.version ? `v${pkg.version}` : "";
	} catch { return ""; }
}

// ── Welcome panel (Feynman-style two-column) ──────────────────────────────────

function printWelcomePanel(): void {
	const termW = Math.min(process.stdout.columns || 100, 110);
	const version = loadVersion();

	// Panel dimensions
	const PANEL_W = Math.min(termW - 4, 96);  // total inner width (excl. borders)
	const LEFT_W  = 36;                        // left column content width
	const RIGHT_W = PANEL_W - LEFT_W - 3;     // right column (│ sep + padding)

	const provider    = currentProviderName === "ollama"
		? "Ollama (local)"
		: (currentProviderName || detectProvider());
	const modelShort  = currentModelId.length > LEFT_W
		? currentModelId.slice(0, LEFT_W - 1) + "…"
		: currentModelId;
	const cwdShort    = process.cwd().replace(os.homedir(), "~").slice(0, LEFT_W);
	const ctxInfo     = contextText
		? `${(contextText.length / 1024).toFixed(1)}KB${contextSource ? ` · ${contextSource}` : ""}`
		: "none";
	const ctxDisplay  = ctxInfo.length > LEFT_W ? ctxInfo.slice(0, LEFT_W - 1) + "…" : ctxInfo;

	// Left column rows  [label, value]
	const subModelDisplay = config.sub_model
		? (config.sub_model.length > LEFT_W ? config.sub_model.slice(0, LEFT_W - 1) + "…" : config.sub_model)
		: null;

	const leftRows: [string, string][] = [
		["model",     modelShort],
		...(subModelDisplay ? [["sub-model", subModelDisplay] as [string, string]] : []),
		["provider",  provider],
		["directory", cwdShort],
		["context",   ctxDisplay],
		["",          ""],
		["max iters", String(config.max_iterations)],
		["sub-queries", String(config.max_sub_queries)],
		["queries run", String(queryCount)],
	];

	// Right column: slash command reference
	const rightRows: [string, string][] = [
		["/file <path>",   "load file / dir / glob"],
		["/url <url>",     "fetch URL as context"],
		["/paste",         "multi-line paste"],
		["@file <query>",  "inline load + query"],
		["",               ""],
		["/model",         "list / switch model"],
		["/provider",      "switch provider"],
		["/key",           "update API key"],
		["",               ""],
		["/trajectories",  "browse saved runs"],
		["/clear",         "clear screen"],
		["/help",          "all commands"],
		["/quit",          "exit"],
	];

	const border = "─".repeat(PANEL_W + 2);

	// Version tag centered in top border
	const vTag    = version ? ` ${version} ` : "";
	const vTagLen = vTag.length;
	const lDash   = Math.floor((PANEL_W + 2 - vTagLen) / 2);
	const rDash   = PANEL_W + 2 - vTagLen - lDash;
	const topBorder = `${DARK_ASH}${BOLD}┌${"─".repeat(lDash)}${RESET}${DIM}${vTag}${RESET}${DARK_ASH}${BOLD}${"─".repeat(rDash)}┐${RESET}`;

	// Column header
	const renderHeader = (left: string, right: string): string => {
		const lPad = Math.max(0, LEFT_W - left.length);
		const rPad = Math.max(0, RIGHT_W - right.length);
		return `${DARK_ASH}${BOLD}│${RESET} ${AMBER}${BOLD}${left}${" ".repeat(lPad)}${RESET} ${DARK_ASH}${BOLD}│${RESET} ${AMBER}${BOLD}${right}${" ".repeat(rPad)}${RESET} ${DARK_ASH}${BOLD}│${RESET}`;
	};

	// Separator row (├──┤)
	const sepBorder = `${DARK_ASH}${BOLD}├${"─".repeat(LEFT_W + 2)}┼${"─".repeat(RIGHT_W + 2)}┤${RESET}`;

	// Content row
	const renderRow = (leftLabel: string, leftVal: string, rightCmd: string, rightDesc: string): string => {
		// Left cell
		let leftCell: string;
		if (!leftLabel && !leftVal) {
			leftCell = " ".repeat(LEFT_W);
		} else {
			const lbl = paint(leftLabel.padEnd(11), DIM);
			const val = paint(leftVal.slice(0, LEFT_W - 12), STONE);
			const rawLen = leftLabel.padEnd(11).length + leftVal.slice(0, LEFT_W - 12).length;
			const lPad = " ".repeat(Math.max(0, LEFT_W - rawLen));
			leftCell = `${lbl} ${val}${lPad}`;
		}

		// Right cell
		let rightCell: string;
		if (!rightCmd && !rightDesc) {
			rightCell = " ".repeat(RIGHT_W);
		} else {
			const cmd  = paint(rightCmd.padEnd(18), SAGE);
			const desc = paint(rightDesc.slice(0, RIGHT_W - 19), ASH);
			const rawLen = 18 + rightDesc.slice(0, RIGHT_W - 19).length;
			const rPad = " ".repeat(Math.max(0, RIGHT_W - rawLen));
			rightCell = `${cmd} ${desc}${rPad}`;
		}

		return `${DARK_ASH}${BOLD}│${RESET} ${leftCell} ${DARK_ASH}${BOLD}│${RESET} ${rightCell} ${DARK_ASH}${BOLD}│${RESET}`;
	};

	const rows = Math.max(leftRows.length, rightRows.length);

	// Print
	console.log(topBorder);
	console.log(renderHeader("Session", "Slash Commands"));
	console.log(sepBorder);
	for (let i = 0; i < rows; i++) {
		const [ll, lv] = leftRows[i] ?? ["", ""];
		const [rc, rd] = rightRows[i] ?? ["", ""];
		console.log(renderRow(ll, lv, rc, rd));
	}
	console.log(`${DARK_ASH}${BOLD}└${"─".repeat(LEFT_W + 2)}┴${"─".repeat(RIGHT_W + 2)}┘${RESET}`);
}

// ── Status line (compact — used after queries) ───────────────────────────────

function printStatusLine(): void {
	const provider    = currentProviderName === "ollama"
		? "Ollama (local)"
		: (currentProviderName || detectProvider());
	const modelShort  = currentModelId.length > 40
		? currentModelId.slice(0, 37) + "…"
		: currentModelId;
	const ctxInfo = contextText
		? `${paint("●", SAGE)} ${paint((contextText.length / 1024).toFixed(1) + "KB", STONE)}${contextSource ? paint(` (${contextSource})`, DIM) : ""}`
		: paint("○ no context", DIM);

	console.log(
		`  ${paint(modelShort, AMBER)}  ${paint(provider, DIM)}  ${ctxInfo}  ${paint(`Q:${queryCount}`, DIM)}`
	);
}

// ── Welcome ──────────────────────────────────────────────────────────────────

function printWelcome(): void {
	console.clear();
	printBanner();
	printWelcomePanel();
	console.log(paint(`\n  Type your query or /help for commands\n`, DIM));
}

/** Generate a concise directory tree string (like `tree -L 2`). */
function generateDirTree(dir: string, prefix = "", depth = 0, maxDepth = 2): string {
	if (depth > maxDepth) return "";
	let entries: fs.Dirent[];
	try { entries = fs.readdirSync(dir, { withFileTypes: true }); }
	catch { return ""; }

	// Filter and sort: dirs first, skip hidden/ignored
	const filtered = entries.filter(e => {
		if (e.name.startsWith(".") && e.name !== ".env") return false;
		if (e.isDirectory() && SKIP_DIRS.has(e.name)) return false;
		if (e.isSymbolicLink()) return false;
		if (e.isFile() && isBinaryFile(path.join(dir, e.name))) return false;
		return true;
	}).sort((a, b) => {
		if (a.isDirectory() !== b.isDirectory()) return a.isDirectory() ? -1 : 1;
		return a.name.localeCompare(b.name);
	});

	// Cap entries per level to keep it concise
	const MAX_PER_LEVEL = 25;
	const shown = filtered.slice(0, MAX_PER_LEVEL);
	const omitted = filtered.length - shown.length;
	const lines: string[] = [];

	for (let i = 0; i < shown.length; i++) {
		const entry = shown[i];
		const isLast = i === shown.length - 1 && omitted === 0;
		const connector = isLast ? "└── " : "├── ";
		const childPrefix = isLast ? "    " : "│   ";
		const suffix = entry.isDirectory() ? "/" : "";
		lines.push(`${prefix}${connector}${entry.name}${suffix}`);
		if (entry.isDirectory()) {
			const sub = generateDirTree(path.join(dir, entry.name), prefix + childPrefix, depth + 1, maxDepth);
			if (sub) lines.push(sub);
		}
	}
	if (omitted > 0) lines.push(`${prefix}└── ... ${omitted} more`);
	return lines.join("\n");
}

/** Build a cwd context string with directory tree. */
function buildCwdContext(): string {
	const cwd = process.cwd();
	const tree = generateDirTree(cwd);
	const parts = [`Working directory: ${cwd}\n`];
	if (tree) parts.push(`File tree:\n${tree}`);
	return parts.join("\n");
}

// ── Help ────────────────────────────────────────────────────────────────────

function printCommandHelp(): void {
	const cmd = (s: string) => paint(s, AMBER, BOLD);
	const kw  = (s: string) => paint(s, ICE);
	const dim = (s: string) => paint(s, DIM);
	const sec = (s: string) => `\n${paint(`  ◆ ${s}`, ICE, BOLD)}`;

	console.log(`
${sec("Loading Context")}
  ${cmd("/file")} <path>              Load a single file
  ${cmd("/file")} <p1> <p2> …        Load multiple files
  ${cmd("/file")} <dir>/             Load all files in a directory (recursive)
  ${cmd("/file")} src/**/*.ts        Load files matching a glob pattern
  ${cmd("/url")} <url>               Fetch URL as context
  ${cmd("/paste")}                   Multi-line paste mode (type ${kw("EOF")} to finish)
  ${cmd("/clear-context")}           Unload context

${sec("@ Shorthand")}  ${dim("(inline file loading)")}
  ${cmd("@file.ts")} <query>          Load file and ask in one shot
  ${cmd("@a.ts @b.ts")} <query>       Load multiple files + query
  ${cmd("@src/")} <query>             Load directory + query
  ${cmd("@src/**/*.ts")} <query>      Load glob + query

${sec("Model & Provider")}
  ${cmd("/model")}                   List models for current provider
  ${cmd("/model")} <#|id>            Switch model by number or ID
  ${cmd("/provider")}                Switch provider (Anthropic · OpenAI · Google · OpenRouter)
  ${cmd("/key")}                     Update an API key

${sec("Tools")}
  ${cmd("/trajectories")}            List saved runs

${sec("General")}
  ${cmd("/clear")}                   Clear screen
  ${cmd("/help")}                    Show this help
  ${cmd("/quit")}                    Exit

${sec("Tips")}
  ${dim("◇")} Just type a question — no context needed for general queries
  ${dim("◇")} Paste a URL directly to fetch it as context
  ${dim("◇")} Paste 4+ lines of text to set it as context
  ${dim("◇")} ${paint("Ctrl+C", BOLD)} stops a running query  ·  ${paint("Ctrl+C twice", BOLD)} exits
  ${dim("◇")} Directories skip node_modules, .git, dist, binaries, etc.
  ${dim("◇")} Limits: ${MAX_FILES} files max, ${MAX_TOTAL_BYTES / 1024 / 1024}MB total
`);
}

// ── Slash command handlers ──────────────────────────────────────────────────

/** Check if a token looks like a file path (vs. plain query text). */
function looksLikePath(token: string): boolean {
	if (token.includes("/") || token.includes("\\")) return true;
	if (token.includes("*") || token.includes("?")) return true;
	if (token.startsWith("~")) return true;
	if (token.startsWith(".")) return true;
	if (/\.\w{1,6}$/.test(token)) return true; // has file extension
	return false;
}

async function handleFile(arg: string): Promise<string | void> {
	if (!arg) {
		console.log(`  ${c.red}Usage: /file <path> [query]${c.reset}`);
		console.log(`  ${c.dim}Examples: /file src/main.ts  |  /file src/  |  /file src/**/*.ts${c.reset}`);
		return;
	}
	const tokens = arg.split(/\s+/).filter(Boolean);

	// Separate paths from query text
	const pathTokens: string[] = [];
	const queryTokens: string[] = [];
	let pastPaths = false;
	for (const t of tokens) {
		if (!pastPaths && looksLikePath(t)) {
			pathTokens.push(t);
		} else {
			pastPaths = true;
			queryTokens.push(t);
		}
	}

	if (pathTokens.length === 0) {
		console.log(`  ${c.red}No file path found in: ${arg}${c.reset}`);
		return;
	}

	const filePaths = resolveFileArgs(pathTokens);

	if (filePaths.length === 0) {
		console.log(`  ${c.red}No files found.${c.reset}`);
		return;
	}

	if (filePaths.length === 1) {
		try {
			contextText = fs.readFileSync(filePaths[0], "utf-8");
			contextSource = path.relative(process.cwd(), filePaths[0]) || filePaths[0];
			contextIsDefault = false;
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

	// Return query text if provided after paths
	if (queryTokens.length > 0) return queryTokens.join(" ");
}

async function handleUrl(arg: string): Promise<void> {
	if (!arg) {
		console.log(`  ${c.red}Usage: /url <url>${c.reset}`);
		return;
	}
	console.log(`  ${c.dim}Fetching ${arg}...${c.reset}`);
	try {
		contextText = await safeFetch(arg);
		contextSource = arg;
		contextIsDefault = false;
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
				contextIsDefault = false;
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
	if (!fs.existsSync(SESSIONS_DIR)) {
		console.log(`  ${c.dim}No sessions yet.${c.reset}`);
		return;
	}

	const sessions = fs
		.readdirSync(SESSIONS_DIR)
		.filter((s) => fs.statSync(path.join(SESSIONS_DIR, s)).isDirectory())
		.sort()
		.reverse();

	if (sessions.length === 0) {
		console.log(`  ${c.dim}No sessions yet.${c.reset}`);
		return;
	}

	console.log(`\n  ${c.bold}Saved sessions${c.reset} ${c.dim}(~/.rlm/sessions/)${c.reset}\n`);

	const shown = sessions.slice(0, 10);
	for (const s of shown) {
		const dir = path.join(SESSIONS_DIR, s);
		const files = fs.readdirSync(dir).filter((f) => f.endsWith(".json"));
		const isCurrent = s === SESSION_ID;
		const dot = isCurrent ? `${c.green}●${c.reset}` : `${c.dim}·${c.reset}`;
		const label = isCurrent
			? `${c.accent}${s}${c.reset} ${c.dim}(current)${c.reset}`
			: `${c.dim}${s}${c.reset}`;
		console.log(`  ${dot} ${label} ${c.dim}${files.length} query${files.length !== 1 ? "s" : ""}${c.reset}`);
	}

	if (sessions.length > 10) {
		console.log(`  ${c.dim}... and ${sessions.length - 10} older sessions${c.reset}`);
	}
	console.log();
}

// ── Display helpers ─────────────────────────────────────────────────────────

let BOX_W = Math.min(process.stdout.columns || 80, 96) - 4;
let MAX_CONTENT_W = BOX_W - 4;

// Update dimensions on terminal resize
process.stdout.on("resize", () => {
	W = Math.min(process.stdout.columns || 80, 100);
	BOX_W = Math.min(process.stdout.columns || 80, 96) - 4;
	MAX_CONTENT_W = BOX_W - 4;
});

/** Strip ANSI escape codes to get visible string length. */
function stripAnsi(str: string): string {
	return str.replace(/\x1b\[[0-9;]*m/g, "");
}

/** Wrap a raw text line into chunks that fit within maxWidth. */
function wrapText(text: string, maxWidth: number): string[] {
	if (text.length <= maxWidth) return [text];
	const result: string[] = [];
	for (let i = 0; i < text.length; i += maxWidth) {
		result.push(text.slice(i, i + maxWidth));
	}
	return result;
}

// ── Box-drawing helpers ─────────────────────────────────────────────────────

function boxTop(label: string, color: string = c.dim): string {
	const visLen = stripAnsi(label).length;
	const right = Math.max(0, BOX_W - visLen - 5);
	return `    ${color}╭─ ${c.reset}${label}${color} ${"─".repeat(right)}╮${c.reset}`;
}

function boxLine(text: string, color: string = c.dim): string {
	return `    ${color}│${c.reset} ${text}`;
}

function boxBottom(color: string = c.dim): string {
	return `    ${color}╰${"─".repeat(BOX_W - 2)}╯${c.reset}`;
}

function stepRule(): void {
	console.log(`    ${c.dim}${"─".repeat(BOX_W - 2)}${c.reset}`);
}

function displayUserPrompt(query: string): void {
	const width = Math.max(24, Math.min(process.stdout.columns || 80, 110) - 8);
	const contentWidth = Math.max(8, width - 4);
	console.log();
	for (const rawLine of query.split("\n")) {
		const chunks = wrapText(rawLine || " ", contentWidth);
		for (const chunk of chunks) {
			const pad = " ".repeat(Math.max(0, contentWidth - stripAnsi(chunk).length));
			console.log(`  ${c.bgPanelAlt || ""}${STONE}${BOLD}›${RESET}${c.bgPanelAlt || ""} ${chunk}${pad} ${RESET}`);
		}
	}
}

// ── Display functions ───────────────────────────────────────────────────────

function displayCode(code: string): void {
	const lines = code.split("\n");
	const lineNumWidth = String(lines.length).length;
	const codeMaxW = MAX_CONTENT_W - lineNumWidth - 1;

	console.log(boxTop("Code", c.code));
	for (let i = 0; i < lines.length; i++) {
		const wrapped = wrapText(lines[i], codeMaxW);
		for (let j = 0; j < wrapped.length; j++) {
			const prefix = j === 0
				? `${c.dim}${String(i + 1).padStart(lineNumWidth)}${c.reset}`
				: " ".repeat(lineNumWidth);
			console.log(`    ${c.code}\u258e${c.reset} ${prefix} ${c.code}${wrapped[j]}${c.reset}`);
		}
	}
	console.log(boxBottom(c.code));
}

function displayOutput(output: string): void {
	const lines = output.split("\n").filter(l => l.trim() !== "");
	if (lines.length === 0) return;

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
	if (lines.length === 0) return;

	console.log(boxTop("Error", c.red));
	for (const line of lines) {
		for (const chunk of wrapText(line, MAX_CONTENT_W)) {
			console.log(boxLine(`${c.red}${chunk}${c.reset}`, c.red));
		}
	}
	console.log(boxBottom(c.red));
}

function showErrorMsg(msg: string): void {
	const lines = msg.split(/\n/).filter(l => l.trim());
	console.log(boxTop("Error", c.red));
	for (const line of lines) {
		for (const chunk of wrapText(line, MAX_CONTENT_W)) {
			console.log(boxLine(chunk, c.red));
		}
	}
	console.log(boxBottom(c.red));
}

function formatSize(chars: number): string {
	return chars >= 1000 ? `${(chars / 1000).toFixed(1)}K` : `${chars}`;
}

function displaySubQueryStart(_info: SubQueryStartInfo): void {
	// Nothing printed here — we wait for the result and show a single compact line
}

function displaySubQueryResult(info: SubQueryInfo): void {
	const elapsed = (info.elapsedMs / 1000).toFixed(1);
	const instrPreview = truncateStr(info.instruction.replace(/\n/g, " "), 55);
	const resultPreview = truncateStr(info.resultPreview.replace(/\n/g, " "), 45);
	console.log(
		`    ${c.subquery}↳${c.reset} ${c.dim}#${info.index}${c.reset}  ${instrPreview}  ${c.dim}→${c.reset}  ${c.result}${resultPreview}${c.reset}  ${c.dim}${elapsed}s${c.reset}`
	);
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
	// ── Dated snapshots / previews ──
	/preview-\d{2}-\d{2}$/,      // e.g. preview-04-17
	/preview-\d{2}-\d{4}$/,      // e.g. preview-09-2025
	/^labs-/,
	/-customtools$/,
	/deep-research$/,
];

function isModelExcluded(modelId: string): boolean {
	return EXCLUDED_MODEL_PATTERNS.some((p) => p.test(modelId));
}

/** Collect models from providers that have an API key set. */
function getAvailableModels(): { id: string; provider: string }[] {
	const items: { id: string; provider: string }[] = [];
	for (const provider of getProviders()) {
		if (!process.env[providerEnvKey(provider)]) continue;
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
const FETCH_TIMEOUT_MS = 30_000;
const MAX_RESPONSE_BYTES = 50 * 1024 * 1024; // 50MB

/** Fetch a URL with timeout and size limits. */
async function safeFetch(url: string): Promise<string> {
	const resp = await fetch(url, { signal: AbortSignal.timeout(FETCH_TIMEOUT_MS) });
	if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`);
	const contentLength = resp.headers.get("content-length");
	if (contentLength && parseInt(contentLength, 10) > MAX_RESPONSE_BYTES) {
		throw new Error(`Response too large (${(parseInt(contentLength, 10) / 1024 / 1024).toFixed(1)}MB)`);
	}
	const text = await resp.text();
	if (text.length > MAX_RESPONSE_BYTES) {
		throw new Error(`Response too large (${(text.length / 1024 / 1024).toFixed(1)}MB)`);
	}
	return text;
}

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
	let fd: number | undefined;
	try {
		fd = fs.openSync(filePath, "r");
		const buf = Buffer.alloc(512);
		const bytesRead = fs.readSync(fd, buf, 0, 512, 0);
		for (let i = 0; i < bytesRead; i++) {
			if (buf[i] === 0) return true;
		}
	} catch { /* unreadable → skip */ return true; }
	finally { if (fd !== undefined) try { fs.closeSync(fd); } catch {} }
	return false;
}

const MAX_DIR_DEPTH = 30;

function walkDir(dir: string, depth = 0): string[] {
	if (depth > MAX_DIR_DEPTH) return [];
	const results: string[] = [];
	let entries: fs.Dirent[];
	try {
		entries = fs.readdirSync(dir, { withFileTypes: true });
	} catch { return results; }

	for (const entry of entries) {
		if (results.length >= MAX_FILES) break;
		if (entry.name.startsWith(".") && entry.name !== ".env") continue;
		if (entry.isSymbolicLink()) continue;
		const full = path.join(dir, entry.name);
		if (entry.isDirectory()) {
			if (SKIP_DIRS.has(entry.name)) continue;
			const sub = walkDir(full, depth + 1);
			const remaining = MAX_FILES - results.length;
			results.push(...sub.slice(0, remaining));
		} else if (entry.isFile()) {
			if (!isBinaryFile(full)) results.push(full);
		}
	}
	return results;
}

/** Normalize path separators to forward slash for consistent matching. */
function toForwardSlash(p: string): string {
	return p.replace(/\\/g, "/");
}

function simpleGlobMatch(pattern: string, filePath: string, _braceDepth = 0): boolean {
	// Normalize both to forward slashes for cross-platform matching
	pattern = toForwardSlash(pattern);
	filePath = toForwardSlash(filePath);

	// Expand {a,b,c} braces into alternatives (with depth limit)
	const braceMatch = pattern.match(/\{([^}]+)\}/);
	if (braceMatch && _braceDepth < 5) {
		const alternatives = braceMatch[1].split(",").slice(0, 50);
		return alternatives.some((alt) =>
			simpleGlobMatch(pattern.replace(braceMatch[0], alt.trim()), filePath, _braceDepth + 1)
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

/** Expand ~ to home directory (shell doesn't do this for us). */
function expandTilde(p: string): string {
	if (p === "~") return os.homedir();
	if (p.startsWith("~/") || p.startsWith("~\\")) return path.join(os.homedir(), p.slice(2));
	return p;
}

function resolveFileArgs(args: string[]): string[] {
	const files: string[] = [];
	for (const rawArg of args) {
		const arg = expandTilde(rawArg);
		const resolved = path.resolve(arg);

		// Glob pattern (contains * or ?)
		if (arg.includes("*") || arg.includes("?")) {
			// Find the base directory (portion before the first glob char)
			const normalized = toForwardSlash(arg);
			const firstGlob = normalized.search(/[*?{]/);
			const baseDir = firstGlob > 0 ? path.resolve(normalized.slice(0, normalized.lastIndexOf("/", firstGlob) + 1) || ".") : process.cwd();
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

// ── Codex-style shell state ────────────────────────────────────────────────

type TranscriptEntry =
	| { kind: "user"; text: string }
	| { kind: "assistant"; text: string }
	| { kind: "stats"; text: string }
	| { kind: "event"; text: string; tone?: "info" | "success" | "error" }
	| { kind: "queued"; text: string };

type TraceEntry =
	| {
		kind: "iteration";
		title: string;
		status: string;
		startedAt: number;
		iteration: number;
		userMessage: string;
		systemPrompt: string;
		code: string;
		rawResponse: string;
		stdout: string;
		stderr: string;
		elapsedMs: number;
	  }
	| {
		kind: "subquery";
		title: string;
		status: string;
		startedAt: number;
		index: number;
		contextLength: number;
		instruction: string;
		result: string;
		elapsedMs: number;
	  };

type IterationTrace = Extract<TraceEntry, { kind: "iteration" }>;
type SubqueryTrace = Extract<TraceEntry, { kind: "subquery" }>;

interface SessionInfo {
	id: string;
	queries: number;
	model: string;
	isCurrent: boolean;
}

interface SessionQuery {
	query: string;
	model: string;
	answerPreview: string;
	fullAnswer: string;
	elapsedMs: number;
	iterations: number;
	subQueries: number;
}

type ShellModal =
	| { kind: "help"; scroll: number }
	| { kind: "model"; selected: number; items: ModelChoice[] }
	| { kind: "provider"; selected: number; items: ProviderChoice[] }
	| { kind: "trace"; selected: number; scroll: number; expanded: boolean; querySelected: number; queryExpanded: boolean }
	| { kind: "trajectories"; selected: number; scroll: number; expanded: boolean; sessions: SessionInfo[]; queries: SessionQuery[]; querySelected: number; queryExpanded: boolean };

interface ModelChoice {
	id: string;
	provider: string;
	label: string;
	meta: string;
}

interface ProviderChoice {
	id: string;
	label: string;
	meta: string;
}

let shellInput = "";
let shellCursor = 0;
let shellScroll = 0;
let shellRenderPending = false;
let shellActive = false;
let shellDestroyed = false;
let introVisible = true;
let transcript: TranscriptEntry[] = [];
let queuedSubmissions: string[] = [];
let activeRunStatus = "";
let activeRunStartedAt = 0;
let traceEntries: TraceEntry[] = [];
let traceHistory: { query: string; entries: TraceEntry[] }[] = [];
let shellModal: ShellModal | null = null;
let traceMessage = "";
let liveRenderTimer: ReturnType<typeof setInterval> | null = null;
let slashMenuIndex = 0;
let transcriptRevision = 0;
let previousFrame: string[] = [];
let previousFrameWidth = 0;
let previousFrameHeight = 0;
let transcriptChunkCacheWidth = 0;
let transcriptChunkCacheIntroVisible = true;
let transcriptStaticRowsCache: string[] = [];
let transcriptChunkRowsCache: string[][] = [];

const SLASH_COMMANDS: { cmd: string; desc: string; needsArg?: boolean }[] = [
	{ cmd: "/help", desc: "show command reference" },
	{ cmd: "/model", desc: "choose model and reasoning effort" },
	{ cmd: "/provider", desc: "switch provider" },
	{ cmd: "/file", desc: "load file, directory, or glob as context", needsArg: true },
	{ cmd: "/url", desc: "fetch a URL into context", needsArg: true },
	{ cmd: "/clear", desc: "clear transcript" },
	{ cmd: "/trace", desc: "open RLM trace window" },
	{ cmd: "/trajectories", desc: "list saved runs" },
	{ cmd: "/quit", desc: "exit the shell" },
];

const CURSOR_BG = "\x1b[48;5;118m\x1b[30m";
const TRANSCRIPT_PAD = 0;
const TURN_TEXT_INDENT = 0;
const SHORTCUTS = [
	`${paint("/help", STONE)} commands`,
	`${paint("/model", STONE)} models`,
	`${paint("/file", STONE)} context`,
	`${paint("/provider", STONE)} provider`,
	`${paint("/trace", STONE)} trace`,
	`${paint("Ctrl+C", STONE)} stop`,
].join(` ${paint("·", DARK_ASH)} `);

function cleanupShell(): void {
	if (!shellActive || shellDestroyed) return;
	shellDestroyed = true;
	if (liveRenderTimer) {
		clearInterval(liveRenderTimer);
		liveRenderTimer = null;
	}
	previousFrame = [];
	previousFrameWidth = 0;
	previousFrameHeight = 0;
	transcriptChunkCacheWidth = 0;
	transcriptChunkCacheIntroVisible = true;
	transcriptStaticRowsCache = [];
	transcriptChunkRowsCache = [];
	try { stdout.write("\x1b[?25h"); } catch {}
	try { stdout.write("\x1b[?1049l"); } catch {}
	try { if (stdin.isTTY) stdin.setRawMode(false); } catch {}
	try { stdin.pause(); } catch {}
}

function startLiveRenderTicker(): void {
	if (liveRenderTimer) return;
	liveRenderTimer = setInterval(() => {
		if (isRunning) requestBackgroundRender();
	}, 1000);
}

function stopLiveRenderTicker(): void {
	if (!liveRenderTimer) return;
	clearInterval(liveRenderTimer);
	liveRenderTimer = null;
}

function requestRender(): void {
	if (!shellActive || shellDestroyed) return;
	if (shellRenderPending) return;
	shellRenderPending = true;
	setImmediate(() => {
		shellRenderPending = false;
		renderShell();
	});
}

function renderNow(): void {
	if (!shellActive || shellDestroyed) return;
	shellRenderPending = false;
	renderShell();
}

function requestBackgroundRender(): void {
	if (!shellActive || shellDestroyed) return;
	if (!shellModal && shellScroll > 0) return;
	requestRender();
}

function shellWidth(): number {
	return Math.max(60, process.stdout.columns || 80);
}

function shellHeight(): number {
	return Math.max(20, process.stdout.rows || 24);
}

function lineScrollStep(): number {
	return Math.max(6, Math.floor(shellHeight() / 5));
}

function contentWidth(width = shellWidth()): number {
	return Math.max(24, width - TRANSCRIPT_PAD * 2);
}

function arraysEqual(a: string[], b: string[]): boolean {
	if (a.length !== b.length) return false;
	for (let i = 0; i < a.length; i++) {
		if (a[i] !== b[i]) return false;
	}
	return true;
}

function statusItems(): string[] {
	const provider = currentProviderName === "ollama" ? "ollama" : (currentProviderName || detectProvider());
	const items = [
		paint(currentModelId, AMBER),
		paint(provider, DIM),
		paint("rlm-cli", DIM),
	];
	if (queuedSubmissions.length > 0) items.push(paint(`queue ${queuedSubmissions.length}`, ICE));
	return items;
}

function wordWrap(text: string, maxWidth: number): string[] {
	if (!text) return [""];
	const rawLines = text.split("\n");
	const out: string[] = [];
	for (const raw of rawLines) {
		if (!raw.trim()) {
			out.push("");
			continue;
		}
		let line = raw;
		while (line.length > maxWidth) {
			let split = line.lastIndexOf(" ", maxWidth);
			if (split <= 0 || split < Math.floor(maxWidth / 3)) split = maxWidth;
			out.push(line.slice(0, split).trimEnd());
			line = line.slice(split).trimStart();
		}
		out.push(line);
	}
	return out;
}

function formatAssistantText(text: string): string {
	return text
		.replace(/\*\*(.+?)\*\*/g, "$1")
		.replace(/__(.+?)__/g, "$1")
		.replace(/`([^`]+)`/g, "$1")
		.replace(/^#{1,6}\s+/gm, "")
		.replace(/^\s*[-*]\s+/gm, "• ")
		.replace(/^\s*\d+\.\s+/gm, (m) => m.trimStart());
}

/** Apply inline markdown styling with ANSI codes to a single line. */
function styleInline(line: string): string {
	return line
		.replace(/\*\*(.+?)\*\*/g, `${BOLD}$1${RESET}`)
		.replace(/__(.+?)__/g, `${BOLD}$1${RESET}`)
		.replace(/`([^`]+)`/g, `${ICE}$1${RESET}`);
}

/**
 * Parse and render a markdown table block into aligned, styled lines.
 * Input: array of raw table lines (|col|col|...).
 * Returns styled output lines.
 */
function renderMarkdownTable(tableLines: string[], maxWidth: number): string[] {
	// Parse rows
	const rows: string[][] = [];
	let alignRow = -1;
	for (let i = 0; i < tableLines.length; i++) {
		const cells = tableLines[i].split("|").map((c) => c.trim()).filter((_, idx, arr) => idx > 0 && idx < arr.length);
		// Detect separator row (e.g. |---|---|)
		if (cells.every((c) => /^[-:]+$/.test(c))) {
			alignRow = i;
			continue;
		}
		rows.push(cells);
	}
	if (rows.length === 0) return [];
	// Compute column widths
	const numCols = Math.max(...rows.map((r) => r.length));
	const colWidths: number[] = [];
	for (let c = 0; c < numCols; c++) {
		colWidths[c] = Math.min(
			Math.max(4, ...rows.map((r) => (r[c] || "").length)),
			Math.max(8, Math.floor((maxWidth - numCols * 3 - 1) / numCols))
		);
	}
	const out: string[] = [];
	// Top border
	const topBorder = `${DARK_ASH}┌${colWidths.map((w) => "─".repeat(w + 2)).join("┬")}┐${RESET}`;
	out.push(topBorder);
	for (let r = 0; r < rows.length; r++) {
		const cells = rows[r];
		const line = cells.map((cell, c) => {
			const w = colWidths[c] || 8;
			const truncated = cell.length > w ? cell.slice(0, w - 1) + "…" : cell;
			const padded = truncated.padEnd(w);
			return r === 0 ? ` ${BOLD}${padded}${RESET} ` : ` ${padded} `;
		}).join(`${DARK_ASH}│${RESET}`);
		out.push(`${DARK_ASH}│${RESET}${line}${DARK_ASH}│${RESET}`);
		// After header row, add separator
		if (r === 0) {
			const sep = `${DARK_ASH}├${colWidths.map((w) => "─".repeat(w + 2)).join("┼")}┤${RESET}`;
			out.push(sep);
		}
	}
	// Bottom border
	const bottomBorder = `${DARK_ASH}└${colWidths.map((w) => "─".repeat(w + 2)).join("┴")}┘${RESET}`;
	out.push(bottomBorder);
	return out;
}

/**
 * Render markdown text into ANSI-styled lines.
 * Handles: headers, bold, inline code, code blocks, tables, hr, lists.
 */
function renderMarkdownLines(text: string, maxWidth: number): string[] {
	const rawLines = text.split("\n");
	const out: string[] = [];
	let i = 0;
	while (i < rawLines.length) {
		const line = rawLines[i];

		// Fenced code block
		if (line.trimStart().startsWith("```")) {
			i++;
			const codeLines: string[] = [];
			while (i < rawLines.length && !rawLines[i].trimStart().startsWith("```")) {
				codeLines.push(rawLines[i]);
				i++;
			}
			if (i < rawLines.length) i++; // skip closing ```
			for (const cl of codeLines) {
				const truncated = cl.length > maxWidth ? cl.slice(0, maxWidth - 1) + "…" : cl;
				out.push(`${ICE}${truncated}${RESET}`);
			}
			continue;
		}

		// Table block (consecutive lines starting with |)
		if (line.trimStart().startsWith("|")) {
			const tableLines: string[] = [];
			while (i < rawLines.length && rawLines[i].trimStart().startsWith("|")) {
				tableLines.push(rawLines[i]);
				i++;
			}
			out.push(...renderMarkdownTable(tableLines, maxWidth));
			continue;
		}

		// Horizontal rule
		if (/^\s*[-*_]{3,}\s*$/.test(line)) {
			out.push(`${DARK_ASH}${"─".repeat(Math.min(maxWidth, 40))}${RESET}`);
			i++;
			continue;
		}

		// Header
		const headerMatch = line.match(/^(#{1,6})\s+(.*)/);
		if (headerMatch) {
			const content = headerMatch[2];
			for (const wl of wordWrap(content, maxWidth)) {
				out.push(`${AMBER}${BOLD}${wl}${RESET}`);
			}
			out.push("");
			i++;
			continue;
		}

		// Blank line
		if (!line.trim()) {
			out.push("");
			i++;
			continue;
		}

		// List items
		const listMatch = line.match(/^(\s*)([-*])\s+(.*)/);
		if (listMatch) {
			const content = `• ${listMatch[3]}`;
			for (const wl of wordWrap(content, maxWidth)) {
				out.push(styleInline(wl));
			}
			i++;
			continue;
		}
		const numListMatch = line.match(/^(\s*)(\d+)\.\s+(.*)/);
		if (numListMatch) {
			const content = `${numListMatch[2]}. ${numListMatch[3]}`;
			for (const wl of wordWrap(content, maxWidth)) {
				out.push(styleInline(wl));
			}
			i++;
			continue;
		}

		// Regular paragraph text — wrap then style inline
		for (const wl of wordWrap(line, maxWidth)) {
			out.push(styleInline(wl));
		}
		i++;
	}
	return out;
}

function appendTranscript(entry: TranscriptEntry): void {
	transcript.push(entry);
	introVisible = false;
	transcriptRevision++;
	transcriptChunkCacheWidth = 0;
	transcriptStaticRowsCache = [];
	transcriptChunkRowsCache = [];
	if (shellScroll === 0) requestRender();
}

function renderPromptBlock(text: string, width: number): string[] {
	const outer = Math.max(20, width);
	const contentW = Math.max(8, outer - 6); // │ › text pad │
	const wrapped = wordWrap(text, contentW);
	const lines: string[] = [];
	lines.push(`${DARK_ASH}╭${"─".repeat(outer - 2)}╮${RESET}`);
	for (const raw of wrapped) {
		const plain = raw || " ";
		const pad = " ".repeat(Math.max(0, contentW - plain.length));
		lines.push(
			`${DARK_ASH}│${RESET} ${c.bgPanelAlt}${STONE}${BOLD}›${RESET}${c.bgPanelAlt} ${plain}${pad}${RESET} ${DARK_ASH}│${RESET}`
		);
	}
	lines.push(`${DARK_ASH}╰${"─".repeat(outer - 2)}╯${RESET}`);
	return lines;
}

function formatLiveElapsed(startedAt: number, elapsedMs: number, status: string): string {
	if (status === "completed" && elapsedMs > 0) return `${(elapsedMs / 1000).toFixed(1)}s`;
	return `${((Date.now() - startedAt) / 1000).toFixed(1)}s`;
}

function renderLiveTraceCards(width: number): string[] {
	if (!isRunning || traceEntries.length === 0) return [];
	const panelWidth = Math.max(24, width - TRANSCRIPT_PAD * 2);
	const recent = traceEntries.slice(-6);
	const out: string[] = [];
	let activeIndex = -1;
	for (let i = recent.length - 1; i >= 0; i--) {
		if (recent[i].status !== "completed") {
			activeIndex = i;
			break;
		}
	}
	for (let i = 0; i < recent.length; i++) {
		const entry = recent[i];
		const isActive = i === (activeIndex >= 0 ? activeIndex : recent.length - 1);
		const tone = entry.status === "completed" ? SAGE : STONE;
		const marker = entry.status === "completed" ? paint("✓", SAGE) : paint("•", tone);
		const title = entry.status === "completed"
			? paint(entry.title, STONE)
			: paint(entry.title, AMBER, BOLD);
		const summary = paint(`${entry.status} · ${formatLiveElapsed(entry.startedAt, entry.elapsedMs, entry.status)}`, DIM);
		out.push(`${marker} ${title} ${paint("·", DARK_ASH)} ${summary}`);
		if (!isActive) continue;
		const detailSource = entry.kind === "iteration"
			? (entry.code ? entry.code : entry.userMessage || "(waiting for code)")
			: `${entry.instruction}${entry.result ? `\n\n${entry.result}` : ""}`;
		const preview = wordWrap(formatAssistantText(detailSource), Math.max(20, panelWidth - 4))
			.map((line) => line.trim())
			.filter((line) => line.length > 0)
			.slice(0, 3);
		for (const line of preview) {
			out.push(`  ${paint("│", DARK_ASH)} ${paint(line || "", DIM)}`);
		}
	}
	return out;
}

function renderAnswerBlock(text: string, width: number, statsText?: string): string[] {
	const outer = Math.max(20, width);
	const contentW = Math.max(8, outer - 4); // │ text pad │
	const styledLines = renderMarkdownLines(text, contentW);
	const lines: string[] = [];
	const label = statsText ? ` ${statsText} ` : "";
	const dashCount = Math.max(0, outer - 3 - label.length);
	lines.push(`${DARK_ASH}╭─${RESET}${paint(label, DIM)}${DARK_ASH}${"─".repeat(dashCount)}╮${RESET}`);
	for (const raw of styledLines) {
		const vw = visibleWidth(raw);
		const padLen = Math.max(0, contentW - vw);
		lines.push(
			`${DARK_ASH}│${RESET} ${raw}${RESET}${" ".repeat(padLen)} ${DARK_ASH}│${RESET}`
		);
	}
	lines.push(`${DARK_ASH}╰${"─".repeat(outer - 2)}╯${RESET}`);
	return lines;
}

function renderTranscriptEntryRows(index: number, width: number): string[] {
	const entry = transcript[index];
	if (!entry) return [];
	if (entry.kind === "user") {
		return renderPromptBlock(entry.text, width);
	}
	const prefix = " ".repeat(TURN_TEXT_INDENT);
	if (entry.kind === "stats") {
		if (index + 1 < transcript.length && transcript[index + 1].kind === "assistant") return [];
		return [`${prefix}${paint(entry.text, DIM)}`];
	}
	if (entry.kind === "assistant") {
		const prev = index > 0 ? transcript[index - 1] : null;
		const statsLabel = prev?.kind === "stats" ? prev.text : undefined;
		return renderAnswerBlock(entry.text, width, statsLabel);
	}
	if (entry.kind === "queued") {
		return [`${prefix}${paint("Queued", ICE, BOLD)} ${paint(entry.text, DIM)}`];
	}
	const tone = entry.tone === "error" ? ROSE : entry.tone === "success" ? SAGE : STONE;
	return wordWrap(entry.text, Math.max(20, width - TURN_TEXT_INDENT - 2)).map(
		(line) => `${prefix}${paint("•", tone)} ${paint(line, entry.tone === "error" ? ROSE : DIM)}`
	);
}

function rebuildTranscriptStaticCache(width: number): void {
	transcriptChunkCacheWidth = width;
	transcriptChunkCacheIntroVisible = introVisible;
	transcriptChunkRowsCache = [];
	transcriptStaticRowsCache = [];
	if (introVisible) {
		transcriptStaticRowsCache.push(`${paint("Recursive language model shell for long-context work.", AMBER, BOLD)}`);
		transcriptStaticRowsCache.push(`${paint("Type a task, load files with /file or @path, and inspect runs with /trajectories.", DIM)}`);
		transcriptStaticRowsCache.push("");
	}
	for (let i = 0; i < transcript.length; i++) {
		const rows = renderTranscriptEntryRows(i, width);
		transcriptChunkRowsCache.push(rows);
		transcriptStaticRowsCache.push(...rows);
	}
}

function ensureTranscriptStaticCache(width: number): void {
	if (
		transcriptChunkCacheWidth !== width ||
		transcriptChunkCacheIntroVisible !== introVisible ||
		transcriptChunkRowsCache.length !== transcript.length
	) {
		rebuildTranscriptStaticCache(width);
	}
}

function renderTranscript(width: number, liveUpdates = true): string[] {
	ensureTranscriptStaticCache(width);
	const staticLines = transcriptStaticRowsCache;
	if (!isRunning) return staticLines;
	// Append dynamic "Working" line (not cached — changes every tick)
	const out = [...staticLines];
	if (!liveUpdates) {
		out.push(`${paint("•", STONE)} ${paint("Working", STONE, BOLD)} ${paint("(live updates paused while scrolled · /trace for details)", DIM)}`);
		return out;
	}
	const elapsed = activeRunStartedAt > 0 ? `${((Date.now() - activeRunStartedAt) / 1000).toFixed(1)}s` : "";
	out.push(`${" ".repeat(TURN_TEXT_INDENT)}${paint("•", STONE)} ${paint("Working", STONE, BOLD)} ${paint(`(${activeRunStatus || "reasoning"}${elapsed ? ` · ${elapsed}` : ""} · /trace for details · Ctrl+C to interrupt)`, DIM)}`);
	out.push(...renderLiveTraceCards(width));
	return out;
}

function renderComposer(width: number): string[] {
	const inner = Math.max(20, width - 2);
	const prefix = "› ";
	const maxText = Math.max(4, inner - prefix.length - 1);
	const start = Math.max(0, shellCursor - maxText + 1);
	const visibleText = shellInput.slice(start, start + maxText);
	const cursorPos = Math.max(0, Math.min(visibleText.length, shellCursor - start));
	const before = visibleText.slice(0, cursorPos);
	const at = visibleText[cursorPos] ?? " ";
	const after = visibleText.slice(Math.min(cursorPos + 1, visibleText.length));
	const shown = `${before}${CURSOR_BG}${at}${RESET}${c.bgPanelAlt}${after}`;
	const visibleLen = before.length + 1 + after.length;
	const pad = " ".repeat(Math.max(0, maxText - visibleLen));
	return [
		`${c.bgPanelAlt}${STONE}${BOLD}›${RESET}${c.bgPanelAlt} ${shown}${pad}${RESET}`,
	];
}

function getSlashMatches(): { cmd: string; desc: string; needsArg?: boolean }[] {
	if (!shellInput.startsWith("/")) return [];
	return SLASH_COMMANDS.filter((c) => c.cmd.startsWith(shellInput));
}

function renderSlashMenu(width: number): string[] {
	const matches = getSlashMatches();
	if (matches.length === 0) return [];
	const lines: string[] = [];
	lines.push("");
	for (let i = 0; i < matches.length; i++) {
		const m = matches[i];
		const active = i === slashMenuIndex;
		const marker = active ? paint("›", SAGE, BOLD) : " ";
		const label = active ? paint(m.cmd.padEnd(18), AMBER, BOLD) : paint(m.cmd.padEnd(18), STONE);
		const desc = paint(m.desc, DIM);
		lines.push(`${marker} ${label}${desc}`);
	}
	return lines;
}

function helpLines(width: number): string[] {
	const items = [
		["/help", "show command reference"],
		["/model", "open model switcher"],
		["/provider", "open provider switcher"],
		["/file <path> [query]", "load file, directory, or glob as context"],
		["/url <url>", "fetch a URL into context"],
		["/clear", "clear transcript"],
		["/trace", "open RLM trace window"],
		["/trajectories", "list saved runs"],
		["/quit", "exit the shell"],
		["Ctrl+O", "open trace window"],
		["PgUp/PgDn", "scroll transcript"],
	];
	const left = items.map(([cmd]) => paint(cmd.padEnd(18), AMBER));
	const right = items.map(([, desc]) => paint(desc, DIM));
	const lines = joinColumns(left, right, 2);
	return lines.map((line) => padAnsiRight(line, Math.max(20, width)));
}

function buildModelChoices(): ModelChoice[] {
	const choices: ModelChoice[] = [];
	const seen = new Set<string>();
	for (const provider of getProviders()) {
		if (!process.env[providerEnvKey(provider)]) continue;
		for (const model of getModels(provider)) {
			if (isModelExcluded(model.id) || seen.has(model.id)) continue;
			seen.add(model.id);
			choices.push({
				id: model.id,
				provider,
				label: model.id,
				meta: provider,
			});
		}
	}
	for (const [key] of ollamaModelMap) {
		if (!key.startsWith("ollama:")) continue;
		const name = key.slice("ollama:".length);
		if (seen.has(name)) continue;
		seen.add(name);
		choices.push({
			id: name,
			provider: "ollama",
			label: name,
			meta: `ollama${ollamaSizes.get(name) ? ` · ${formatOllamaSize(ollamaSizes.get(name) || 0)}` : ""}`,
		});
	}
	return choices;
}

function buildProviderChoices(): ProviderChoice[] {
	const items: ProviderChoice[] = SETUP_PROVIDERS
		.filter((p) => process.env[p.env])
		.map((p) => ({
			id: p.piProvider,
			label: p.name,
			meta: p.label,
		}));
	if (ollamaModelMap.size > 0) {
		items.push({
			id: "ollama",
			label: "Ollama",
			meta: "local",
		});
	}
	return items;
}

function openHelpModal(): void {
	shellModal = { kind: "help", scroll: 0 };
	requestRender();
}

function openModelModal(): void {
	const items = buildModelChoices();
	const selected = Math.max(0, items.findIndex((item) => item.id === currentModelId));
	shellModal = { kind: "model", items, selected };
	requestRender();
}

function openProviderModal(): void {
	const items = buildProviderChoices();
	const selected = Math.max(0, items.findIndex((item) => item.id === currentProviderName));
	shellModal = { kind: "provider", items, selected };
	requestRender();
}

function openTraceModal(): void {
	const allQueries = [...traceHistory];
	if (traceEntries.length > 0) {
		// Include current in-progress/last query
		allQueries.push({ query: "(current)", entries: traceEntries });
	}
	shellModal = {
		kind: "trace",
		selected: Math.max(0, allQueries.length - 1),
		scroll: 0,
		expanded: false,
		querySelected: 0,
		queryExpanded: false,
	};
	requestRender();
}

function loadSessionList(): SessionInfo[] {
	if (!fs.existsSync(SESSIONS_DIR)) return [];
	return fs.readdirSync(SESSIONS_DIR)
		.filter((s) => { try { return fs.statSync(path.join(SESSIONS_DIR, s)).isDirectory(); } catch { return false; } })
		.sort()
		.reverse()
		.map((s) => {
			const dir = path.join(SESSIONS_DIR, s);
			const files = fs.readdirSync(dir).filter((f) => f.endsWith(".json"));
			let model = "";
			if (files.length > 0) {
				try {
					const first = JSON.parse(fs.readFileSync(path.join(dir, files[0]), "utf-8"));
					model = first.model || "";
				} catch {}
			}
			return { id: s, queries: files.length, model, isCurrent: s === SESSION_ID };
		});
}

function loadSessionQueries(sessionId: string): SessionQuery[] {
	const dir = path.join(SESSIONS_DIR, sessionId);
	if (!fs.existsSync(dir)) return [];
	return fs.readdirSync(dir)
		.filter((f) => f.endsWith(".json"))
		.sort()
		.map((f) => {
			try {
				const d = JSON.parse(fs.readFileSync(path.join(dir, f), "utf-8"));
				const answer = d.result?.answer || "";
				return {
					query: d.query || "(no query)",
					model: d.model || "",
					answerPreview: answer.replace(/\n/g, " ").slice(0, 120),
					fullAnswer: answer,
					elapsedMs: d.totalElapsedMs || 0,
					iterations: d.iterations?.length || d.result?.iterations || 0,
					subQueries: d.result?.totalSubQueries || 0,
				};
			} catch {
				return { query: "(unreadable)", model: "", answerPreview: "", fullAnswer: "", elapsedMs: 0, iterations: 0, subQueries: 0 };
			}
		});
}

function openTrajectoriesModal(): void {
	const sessions = loadSessionList();
	shellModal = { kind: "trajectories", selected: 0, scroll: 0, expanded: false, sessions, queries: [], querySelected: 0, queryExpanded: false };
	requestRender();
}

function closeModal(): void {
	shellModal = null;
	requestRender();
}

let _traceDetailCache: { key: string; lines: string[] } | null = null;

function buildTraceDetail(entry: TraceEntry, width: number): string[] {
	const cacheKey = `${entry.kind}:${entry.title}:${entry.status}:${width}`;
	if (_traceDetailCache && _traceDetailCache.key === cacheKey) return _traceDetailCache.lines;
	const inner = Math.max(20, width - 2);
	if (entry.kind === "iteration") {
		const sections = [
			...renderCard("Prompt", wordWrap(entry.userMessage || "(none)", inner - 3), inner),
			...renderCard("System Prompt", wordWrap(entry.systemPrompt || "(none)", inner - 3), inner),
			...renderCard("Code", wordWrap(entry.code || "(none)", inner - 3), inner),
		];
		if (entry.stdout.trim()) sections.push(...renderCard("Output", wordWrap(entry.stdout, inner - 3), inner));
		if (entry.stderr.trim()) sections.push(...renderCard("Error", wordWrap(entry.stderr, inner - 3), inner));
		if (entry.rawResponse.trim()) sections.push(...renderCard("Model Response", wordWrap(entry.rawResponse, inner - 3), inner));
		_traceDetailCache = { key: cacheKey, lines: sections };
		return sections;
	}
	const result = [
		...renderCard("Instruction", wordWrap(entry.instruction || "(none)", inner - 3), inner),
		...renderCard("Context", wordWrap(`Context length: ${entry.contextLength.toLocaleString()} chars`, inner - 3), inner),
		...renderCard("Result", wordWrap(entry.result || "(pending)", inner - 3), inner),
	];
	_traceDetailCache = { key: cacheKey, lines: result };
	return result;
}

function renderModalView(width: number, height: number): string[] {
	if (!shellModal) return [];
	const lines: string[] = [];
	if (shellModal.kind === "help") {
		lines.push(paint("Commands", AMBER, BOLD));
		lines.push(paint("Esc closes this window.", DIM));
		lines.push("");
		lines.push(...helpLines(width));
		return lines;
	}
	if (shellModal.kind === "model") {
		lines.push(paint("Select Model", AMBER, BOLD));
		lines.push(paint("↑/↓ navigate · Enter select · Esc cancel", DIM));
		lines.push("");
		for (let i = 0; i < shellModal.items.length; i++) {
			const item = shellModal.items[i];
			const active = i === shellModal.selected;
			const marker = active ? paint("›", SAGE, BOLD) : paint(" ", DIM);
			const label = active ? paint(item.label, STONE, BOLD) : item.label;
			lines.push(`${marker} ${padAnsiRight(label, Math.max(20, width - 26))} ${paint(item.meta, DIM)}`);
		}
		return lines;
	}
	if (shellModal.kind === "provider") {
		lines.push(paint("Select Provider", AMBER, BOLD));
		lines.push(paint("↑/↓ navigate · Enter select · Esc cancel", DIM));
		lines.push("");
		for (let i = 0; i < shellModal.items.length; i++) {
			const item = shellModal.items[i];
			const active = i === shellModal.selected;
			const marker = active ? paint("›", SAGE, BOLD) : paint(" ", DIM);
			const label = active ? paint(item.label, STONE, BOLD) : item.label;
			lines.push(`${marker} ${padAnsiRight(label, Math.max(20, width - 24))} ${paint(item.meta, DIM)}`);
		}
		return lines;
	}
	if (shellModal.kind === "trajectories") {
		const trajModal = shellModal;
		if (trajModal.expanded && trajModal.queryExpanded) {
			// Level 3: Full answer view for a single query
			const q = trajModal.queries[Math.min(trajModal.querySelected, trajModal.queries.length - 1)];
			const secs = (q.elapsedMs / 1000).toFixed(1);
			lines.push(paint(`Query: ${q.query.slice(0, width - 10)}`, AMBER, BOLD));
			lines.push(paint(`${q.model} · ${q.iterations} iter · ${q.subQueries} sub · ${secs}s`, DIM));
			lines.push(paint("↑/↓ scroll · Esc back", DIM));
			lines.push("");
			const answerLines = q.fullAnswer
				? renderMarkdownLines(q.fullAnswer, Math.max(20, width - 4))
				: [paint("(no answer)", DIM)];
			const detailHeight = Math.max(4, height - lines.length);
			const maxScroll = Math.max(0, answerLines.length - detailHeight);
			if (trajModal.scroll > maxScroll) trajModal.scroll = maxScroll;
			for (let j = 0; j < detailHeight; j++) {
				lines.push(answerLines[trajModal.scroll + j] ?? "");
			}
			return lines;
		}
		if (trajModal.expanded) {
			// Level 2: Queries list within a session
			const session = trajModal.sessions[Math.min(trajModal.selected, trajModal.sessions.length - 1)];
			lines.push(paint(`Session: ${session.id}`, AMBER, BOLD));
			if (trajModal.queries.length === 0) {
				lines.push(paint("No queries in this session.", DIM));
				return lines;
			}
			lines.push(paint("↑/↓ select · Enter view answer · Esc back", DIM));
			lines.push("");
			const inner = Math.max(20, width - 6);
			for (let i = 0; i < trajModal.queries.length; i++) {
				const q = trajModal.queries[i];
				const active = i === trajModal.querySelected;
				const marker = active ? paint("›", SAGE, BOLD) : paint(" ", DIM);
				const secs = (q.elapsedMs / 1000).toFixed(1);
				const qLabel = active ? paint(q.query.slice(0, inner), STONE, BOLD) : paint(q.query.slice(0, inner), DIM);
				lines.push(`${marker} ${qLabel}`);
				lines.push(`  ${paint(q.model, DIM)}  ${paint(`${q.iterations} iter · ${q.subQueries} sub · ${secs}s`, DIM)}`);
				if (q.answerPreview) {
					lines.push(`  ${paint(q.answerPreview.slice(0, inner), DIM)}`);
				}
				lines.push("");
			}
			return lines;
		}
		// Level 1: Sessions list
		lines.push(paint("Trajectories", AMBER, BOLD));
		if (trajModal.sessions.length === 0) {
			lines.push(paint("No saved sessions yet.", DIM));
			return lines;
		}
		lines.push(paint("↑/↓ select · Enter expand · Esc close", DIM));
		lines.push("");
		for (let i = 0; i < trajModal.sessions.length; i++) {
			const s = trajModal.sessions[i];
			const active = i === trajModal.selected;
			const marker = active ? paint("›", SAGE, BOLD) : paint(" ", DIM);
			const dot = s.isCurrent ? paint("●", SAGE) : paint("·", DIM);
			const label = active ? paint(s.id, STONE, BOLD) : paint(s.id, DIM);
			const meta = paint(`${s.queries} quer${s.queries !== 1 ? "ies" : "y"} · ${s.model}`, DIM);
			lines.push(`${marker} ${dot} ${label}  ${meta}`);
		}
		return lines;
	}

	lines.push(paint("RLM Trace", AMBER, BOLD));
	const allQueries = [...traceHistory];
	if (traceEntries.length > 0) {
		allQueries.push({ query: "(current)", entries: traceEntries });
	}
	if (allQueries.length === 0) {
		lines.push(paint("No iterations or sub-queries yet.", DIM));
		return lines;
	}
	const traceModal = shellModal;
	if (!traceModal || traceModal.kind !== "trace") return lines;

	if (traceModal.queryExpanded) {
		// Level 3: expanded detail for a single trace entry
		const q = allQueries[Math.min(traceModal.selected, allQueries.length - 1)];
		const entries = q?.entries || [];
		const entryIdx = Math.min(traceModal.querySelected, entries.length - 1);
		const entry = entries[entryIdx];
		if (!entry) return lines;
		lines.push(paint(`${entry.title}  ← Esc back · ↑/↓ scroll`, DIM));
		lines.push("");
		const detailHeight = Math.max(4, height - lines.length);
		const fullDetail = buildTraceDetail(entry, Math.max(24, width - 2));
		const maxScroll = Math.max(0, fullDetail.length - detailHeight);
		if (traceModal.scroll > maxScroll) traceModal.scroll = maxScroll;
		for (let j = 0; j < detailHeight; j++) {
			lines.push(fullDetail[traceModal.scroll + j] ?? "");
		}
		return lines;
	}

	if (traceModal.expanded) {
		// Level 2: trace entries for a specific query
		const q = allQueries[Math.min(traceModal.selected, allQueries.length - 1)];
		const entries = q?.entries || [];
		lines.push(paint(`Query: ${q?.query || "unknown"}`, AMBER, BOLD));
		if (entries.length === 0) {
			lines.push(paint("No trace entries for this query.", DIM));
			return lines;
		}
		lines.push(paint("↑/↓ select · Enter expand · Esc back", DIM));
		lines.push("");
		for (let idx = 0; idx < entries.length; idx++) {
			const entry = entries[idx];
			const active = idx === traceModal.querySelected;
			const marker = active ? paint("›", SAGE, BOLD) : paint(" ", DIM);
			const title = active ? paint(entry.title, STONE, BOLD) : paint(entry.title, DIM);
			const status = entry.status === "completed" ? paint("✓", SAGE) : paint("…", STONE);
			lines.push(`${marker} ${status} ${title}`);
		}
		return lines;
	}

	// Level 1: query list
	lines.push(paint("↑/↓ select · Enter expand · Esc close", DIM));
	lines.push("");
	for (let idx = 0; idx < allQueries.length; idx++) {
		const q = allQueries[idx];
		const active = idx === traceModal.selected;
		const marker = active ? paint("›", SAGE, BOLD) : paint(" ", DIM);
		const label = active ? paint(q.query, STONE, BOLD) : paint(q.query, DIM);
		const count = paint(`${q.entries.length} step${q.entries.length !== 1 ? "s" : ""}`, DIM);
		lines.push(`${marker} ${label}  ${count}`);
	}
	return lines;
}

function renderShell(): void {
	if (!shellActive || shellDestroyed) return;
	const width = shellWidth();
	const height = shellHeight();
	const slashMenu = (!shellModal && shellInput.startsWith("/")) ? renderSlashMenu(width) : [];
	const footerLines = [
		...slashMenu,
		buildStatusBar(statusItems(), width),
		...renderComposer(width),
	];
	const transcriptHeight = Math.max(4, height - footerLines.length - 2);
	const liveUpdates = !(isRunning && !shellModal && shellScroll > 0);
	const bodyLines = shellModal ? renderModalView(width, transcriptHeight) : renderTranscript(width, liveUpdates);
	const total = bodyLines.length;
	// Modals handle their own internal scroll — skip global scroll for them
	const activeScroll = shellModal ? 0 : shellScroll;
	const start = Math.max(0, total - transcriptHeight - activeScroll);
	const end = Math.max(start, Math.min(total, start + transcriptHeight));
	const visible = bodyLines.slice(start, end);
	const padded: string[] = [...visible];
	while (padded.length < transcriptHeight) padded.push("");
	const finalLines = [...padded, "", ...footerLines];
	while (finalLines.length < height) finalLines.push("");
	if (finalLines.length > height) finalLines.length = height;

	const fullFrame = finalLines.map((line) => padAnsiRight(line, width));
	const resetFrame =
		previousFrameWidth !== width ||
		previousFrameHeight !== height ||
		previousFrame.length !== fullFrame.length;

	const buf: string[] = [];
	if (resetFrame) {
		buf.push("\x1b[H\x1b[2J");
		for (let i = 0; i < fullFrame.length; i++) {
			buf.push(`\x1b[${i + 1};1H`);
			buf.push(fullFrame[i]);
			buf.push("\x1b[K");
		}
	} else {
		for (let i = 0; i < fullFrame.length; i++) {
			if (fullFrame[i] === previousFrame[i]) continue;
			buf.push(`\x1b[${i + 1};1H`);
			buf.push(fullFrame[i]);
			buf.push("\x1b[K");
		}
	}
	buf.push(`\x1b[${height};${Math.max(1, width)}H`);
	previousFrame = fullFrame;
	previousFrameWidth = width;
	previousFrameHeight = height;
	stdout.write(buf.join(""));
}

function insertInput(text: string): void {
	shellInput = shellInput.slice(0, shellCursor) + text + shellInput.slice(shellCursor);
	shellCursor += text.length;
	requestRender();
}

function deleteBackward(): void {
	if (shellCursor === 0) return;
	shellInput = shellInput.slice(0, shellCursor - 1) + shellInput.slice(shellCursor);
	shellCursor--;
	requestRender();
}

function saveTrajectory(query: string, effectiveContext: string, trajectory: any): void {
	try {
		if (!fs.existsSync(SESSION_DIR)) fs.mkdirSync(SESSION_DIR, { recursive: true });
		const trajFile = `query-${String(queryCount).padStart(3, "0")}.json`;
		fs.writeFileSync(path.join(SESSION_DIR, trajFile), JSON.stringify({
			...trajectory,
			model: currentModelId,
			query,
			contextLength: effectiveContext.length,
			contextLines: effectiveContext.split("\n").length,
		}, null, 2), "utf-8");
	} catch {}
}

async function runQuery(query: string): Promise<void> {
	const effectiveContext = contextText || query;
	if (!currentModel) {
		const resolved = resolveModelWithProvider(currentModelId);
		if (resolved) {
			currentModel = resolved.model;
			currentProviderName = resolved.provider;
		}
	}
	if (!currentModel) {
		appendTranscript({ kind: "event", tone: "error", text: `Model "${currentModelId}" not found.` });
		requestRender();
		return;
	}

	isRunning = true;
	queryCount++;
	shellScroll = 0;
	appendTranscript({ kind: "user", text: query });
	// Save previous trace to history before clearing
	if (traceEntries.length > 0) {
		const prevQuery = transcript.filter(t => t.kind === "user");
		const label = prevQuery.length >= 2 ? prevQuery[prevQuery.length - 2].text.slice(0, 60) : "query";
		traceHistory.push({ query: label, entries: traceEntries });
	}
	traceEntries = [];
	traceMessage = "";
	activeRunStatus = "reasoning";
	activeRunStartedAt = Date.now();
	startLiveRenderTicker();
	requestRender();

	const startTime = Date.now();
	const trajectory: any = {
		startTime: new Date().toISOString(),
		iterations: [],
		result: null,
		totalElapsedMs: 0,
	};
	let currentIteration: IterationTrace | null = null;
	const repl = new PythonRepl();
	const ac = new AbortController();
	activeAc = ac;
	activeRepl = repl;

	try {
		await repl.start(ac.signal);
		const result = await runRlmLoop({
			context: effectiveContext,
			query,
			model: currentModel,
			subModel: config.sub_model ? resolveModel(config.sub_model) : undefined,
			repl,
			signal: ac.signal,
			onProgress: (info: RlmProgress) => {
				if (info.phase === "generating_code") {
					currentIteration = {
						kind: "iteration",
						title: `Iteration ${info.iteration}`,
						status: "reasoning",
						startedAt: Date.now(),
						iteration: info.iteration,
						userMessage: info.userMessage || "",
						systemPrompt: info.systemPrompt || "",
						code: "",
						rawResponse: "",
						stdout: "",
						stderr: "",
						elapsedMs: 0,
					};
					traceEntries.push(currentIteration);
					activeRunStatus = `iteration ${info.iteration} · reasoning`;
					requestBackgroundRender();
				}
				if (info.phase === "executing" && currentIteration) {
					currentIteration.code = info.code || "";
					currentIteration.rawResponse = info.rawResponse || "";
					currentIteration.status = "executing";
					activeRunStatus = `iteration ${info.iteration} · executing`;
					requestBackgroundRender();
				}
				if (info.phase === "checking_final" && currentIteration) {
					currentIteration.stdout = info.stdout || "";
					currentIteration.stderr = info.stderr || "";
					currentIteration.elapsedMs = Date.now() - currentIteration.startedAt;
					currentIteration.status = "completed";
					trajectory.iterations.push({ ...currentIteration });
					activeRunStatus = `iteration ${info.iteration} · finalizing`;
					requestBackgroundRender();
				}
			},
			onSubQueryStart: (info: SubQueryStartInfo) => {
				traceEntries.push({
					kind: "subquery",
					title: `Sub-query ${info.index}`,
					status: "running",
					startedAt: Date.now(),
					index: info.index,
					contextLength: info.contextLength,
					instruction: info.instruction,
					result: "",
					elapsedMs: 0,
				});
				activeRunStatus = `sub-query ${info.index}`;
				traceMessage = `sub-query ${info.index}`;
				requestBackgroundRender();
			},
			onSubQuery: (info: SubQueryInfo) => {
				const item = traceEntries.find((entry): entry is SubqueryTrace => entry.kind === "subquery" && entry.index === info.index);
				if (item) {
					item.result = info.resultPreview;
					item.elapsedMs = info.elapsedMs;
					item.status = "completed";
				}
				activeRunStatus = `sub-query ${info.index} finished`;
				requestBackgroundRender();
			},
		});

		trajectory.result = result;
		trajectory.totalElapsedMs = Date.now() - startTime;
		saveTrajectory(query, effectiveContext, trajectory);
		appendTranscript({
			kind: "stats",
			text: `${result.iterations} step${result.iterations !== 1 ? "s" : ""} · ${result.totalSubQueries} sub-quer${result.totalSubQueries !== 1 ? "ies" : "y"} · ${((Date.now() - startTime) / 1000).toFixed(1)}s`,
		});
		appendTranscript({ kind: "assistant", text: result.answer });
	} catch (err: any) {
		const msg = err?.message || String(err);
		if (
			err?.name !== "AbortError" &&
			!msg.includes("Aborted") &&
			!msg.includes("not running") &&
			!msg.includes("REPL subprocess") &&
			!msg.includes("REPL shut down")
		) {
			appendTranscript({ kind: "event", tone: "error", text: msg });
		}
	} finally {
		activeRunStatus = "";
		activeRunStartedAt = 0;
		activeAc = null;
		activeRepl = null;
		try { repl.shutdown(); } catch {}
		isRunning = false;
		stopLiveRenderTicker();
		requestRender();
		void processQueuedSubmission();
	}
}

function queueSubmission(query: string): void {
	queuedSubmissions.push(query);
	appendTranscript({ kind: "queued", text: query });
	requestRender();
}

async function processQueuedSubmission(): Promise<void> {
	if (isRunning) return;
	const next = queuedSubmissions.shift();
	if (!next) return;
	await runQuery(next);
}

function expandAtFiles(input: string): string {
	const tokens: string[] = [];
	const remaining: string[] = [];
	for (const part of input.split(/\s+/)) {
		if (part.startsWith("@") && part.length > 1) tokens.push(expandTilde(part.slice(1)));
		else remaining.push(part);
	}
	if (tokens.length === 0) return input;
	const filePaths = resolveFileArgs(tokens);
	if (filePaths.length === 0) {
		appendTranscript({ kind: "event", tone: "error", text: `No files found for: ${tokens.join(", ")}` });
		return "";
	}
	if (filePaths.length === 1) {
		contextText = fs.readFileSync(filePaths[0], "utf-8");
		contextSource = path.relative(process.cwd(), filePaths[0]) || filePaths[0];
		contextIsDefault = false;
		appendTranscript({ kind: "event", tone: "success", text: `Loaded ${contextText.length.toLocaleString()} chars from ${contextSource}` });
	} else {
		const { text, count, totalBytes } = loadMultipleFiles(filePaths);
		contextText = text;
		contextSource = `${count} files`;
		contextIsDefault = false;
		appendTranscript({ kind: "event", tone: "success", text: `Loaded ${count} files (${(totalBytes / 1024).toFixed(1)}KB)` });
	}
	return remaining.join(" ");
}

async function detectAndLoadUrl(input: string): Promise<boolean> {
	const urlMatch = input.match(/^https?:\/\/\S+$/);
	if (!urlMatch) return false;
	const url = urlMatch[0];
	try {
		contextText = await safeFetch(url);
		contextSource = url;
		contextIsDefault = false;
		appendTranscript({ kind: "event", tone: "success", text: `Loaded ${(contextText.length / 1024).toFixed(1)}KB from ${url}` });
		return true;
	} catch (err: any) {
		appendTranscript({ kind: "event", tone: "error", text: `Failed to fetch ${url}: ${err.message}` });
		return false;
	}
}

async function switchModelById(modelId: string): Promise<void> {
	const resolved = resolveModelWithProvider(modelId);
	if (!resolved) {
		appendTranscript({ kind: "event", tone: "error", text: `Model "${modelId}" not found.` });
		return;
	}
	currentModelId = modelId;
	currentModel = resolved.model;
	currentProviderName = resolved.provider;
	saveModelPreference(currentModelId);
	appendTranscript({ kind: "event", tone: "success", text: `Switched to ${currentModelId}` });
}

async function switchProvider(providerId: string): Promise<void> {
	if (providerId === "ollama") {
		const first = [...ollamaModelMap.keys()].find((key) => key.startsWith("ollama:"))?.slice("ollama:".length);
		if (!first) {
			appendTranscript({ kind: "event", tone: "error", text: "No Ollama models available." });
			return;
		}
		await switchModelById(first);
		return;
	}
	const defaultModel = getDefaultModelForProvider(providerId);
	if (!defaultModel) {
		appendTranscript({ kind: "event", tone: "error", text: `No models available for ${providerId}.` });
		return;
	}
	await switchModelById(defaultModel);
}

async function handleSlashCommand(line: string): Promise<void> {
	const [cmd, ...rest] = line.slice(1).split(/\s+/);
	const arg = rest.join(" ").trim();
	switch (cmd) {
		case "help":
		case "h":
			openHelpModal();
			return;
		case "model":
		case "m":
			if (arg) await switchModelById(arg);
			else openModelModal();
			return;
		case "provider":
		case "prov":
			openProviderModal();
			return;
		case "trace":
			openTraceModal();
			return;
		case "file":
		case "f": {
			if (!arg) {
				appendTranscript({ kind: "event", tone: "error", text: "Usage: /file <path> [query]" });
				return;
			}
			const tokens = arg.split(/\s+/).filter(Boolean);
			const pathTokens: string[] = [];
			const queryTokens: string[] = [];
			let pastPaths = false;
			for (const token of tokens) {
				if (!pastPaths && looksLikePath(token)) pathTokens.push(token);
				else {
					pastPaths = true;
					queryTokens.push(token);
				}
			}
			const filePaths = resolveFileArgs(pathTokens);
			if (filePaths.length === 0) {
				appendTranscript({ kind: "event", tone: "error", text: "No files found." });
				return;
			}
			if (filePaths.length === 1) {
				contextText = fs.readFileSync(filePaths[0], "utf-8");
				contextSource = path.relative(process.cwd(), filePaths[0]) || filePaths[0];
				contextIsDefault = false;
				appendTranscript({ kind: "event", tone: "success", text: `Loaded ${contextText.length.toLocaleString()} chars from ${contextSource}` });
			} else {
				const loaded = loadMultipleFiles(filePaths);
				contextText = loaded.text;
				contextSource = `${loaded.count} files`;
				contextIsDefault = false;
				appendTranscript({ kind: "event", tone: "success", text: `Loaded ${loaded.count} files (${(loaded.totalBytes / 1024).toFixed(1)}KB)` });
			}
			const fileQuery = queryTokens.join(" ").trim();
			if (fileQuery) {
				if (isRunning) queueSubmission(fileQuery);
				else await runQuery(fileQuery);
			}
			return;
		}
		case "url":
		case "u":
			if (!arg) {
				appendTranscript({ kind: "event", tone: "error", text: "Usage: /url <url>" });
				return;
			}
			await detectAndLoadUrl(arg);
			return;
		case "clear":
			transcript = [];
			introVisible = true;
			transcriptRevision++;
			transcriptChunkCacheWidth = 0;
			transcriptStaticRowsCache = [];
			transcriptChunkRowsCache = [];
			requestRender();
			return;
		case "trajectories":
		case "traj":
			openTrajectoriesModal();
			return;
		case "quit":
		case "q":
		case "exit":
			cleanupShell();
			process.exit(0);
			return;
		default:
			appendTranscript({ kind: "event", tone: "error", text: `Unknown command: /${cmd}` });
	}
}

async function submitCurrentInput(): Promise<void> {
	const raw = shellInput.trim();
	if (!raw) return;
	shellInput = "";
	shellCursor = 0;
	requestRender();

	if (raw.startsWith("/")) {
		await handleSlashCommand(raw);
		return;
	}
	if (await detectAndLoadUrl(raw)) return;

	let query = expandAtFiles(raw);
	if (!query && raw.startsWith("@")) return;
	if (!query) query = raw;

	if (isRunning) {
		queueSubmission(query);
		return;
	}
	await runQuery(query);
}

async function handleShellKey(str: string, key: readline.Key): Promise<void> {
	// Ctrl+C always works — even inside modals
	if (key.ctrl && key.name === "c") {
		if (shellModal) { closeModal(); return; }
		if (isRunning && activeAc) {
			activeAc.abort();
			try { activeRepl?.shutdown(); } catch {}
			appendTranscript({ kind: "event", tone: "error", text: "Stopped" });
			requestRender();
			return;
		}
		cleanupShell();
		process.exit(0);
	}

	if (shellModal) {
		if (key.name === "escape" || (key.ctrl && key.name === "o")) {
			// In expanded view, Esc goes back one level instead of closing
			if (shellModal.kind === "trace") {
				if (shellModal.queryExpanded) {
					shellModal.queryExpanded = false;
					shellModal.scroll = 0;
					requestRender();
					return;
				}
				if (shellModal.expanded) {
					shellModal.expanded = false;
					shellModal.scroll = 0;
					shellModal.querySelected = 0;
					requestRender();
					return;
				}
			}
			if (shellModal.kind === "trajectories") {
				if (shellModal.queryExpanded) {
					// Level 3 → Level 2
					shellModal.queryExpanded = false;
					shellModal.scroll = 0;
					requestRender();
					return;
				}
				if (shellModal.expanded) {
					// Level 2 → Level 1
					shellModal.expanded = false;
					shellModal.scroll = 0;
					shellModal.querySelected = 0;
					requestRender();
					return;
				}
			}
			closeModal();
			return;
		}
		if (shellModal.kind === "model") {
			if (key.name === "up") shellModal.selected = Math.max(0, shellModal.selected - 1);
			if (key.name === "down") shellModal.selected = Math.min(shellModal.items.length - 1, shellModal.selected + 1);
			if (key.name === "return") {
				const item = shellModal.items[shellModal.selected];
				closeModal();
				await switchModelById(item.id);
			}
			requestRender();
			return;
		}
		if (shellModal.kind === "provider") {
			if (key.name === "up") shellModal.selected = Math.max(0, shellModal.selected - 1);
			if (key.name === "down") shellModal.selected = Math.min(shellModal.items.length - 1, shellModal.selected + 1);
			if (key.name === "return") {
				const item = shellModal.items[shellModal.selected];
				closeModal();
				await switchProvider(item.id);
			}
			requestRender();
			return;
		}
		if (shellModal.kind === "trajectories") {
			if (shellModal.queryExpanded) {
				// Level 3: scrolling full answer
				const lineStep = lineScrollStep();
				if (key.name === "up") shellModal.scroll = Math.max(0, shellModal.scroll - lineStep);
				if (key.name === "down") shellModal.scroll += lineStep;
				const step = Math.max(4, Math.floor(shellHeight() / 3));
				if (key.name === "pageup") shellModal.scroll = Math.max(0, shellModal.scroll - step);
				if (key.name === "pagedown") shellModal.scroll += step;
				renderNow();
				return;
			}
			if (shellModal.expanded) {
				// Level 2: queries list — ↑/↓ selects, Enter expands
				if (key.name === "up") shellModal.querySelected = Math.max(0, shellModal.querySelected - 1);
				if (key.name === "down") shellModal.querySelected = Math.min(shellModal.queries.length - 1, shellModal.querySelected + 1);
				if (key.name === "return" && shellModal.queries.length > 0) {
					shellModal.queryExpanded = true;
					shellModal.scroll = 0;
				}
				renderNow();
				return;
			}
			// Level 1: sessions list — ↑/↓ selects, Enter expands into session
			if (key.name === "up") shellModal.selected = Math.max(0, shellModal.selected - 1);
			if (key.name === "down") shellModal.selected = Math.min(shellModal.sessions.length - 1, shellModal.selected + 1);
			if (key.name === "return" && shellModal.sessions.length > 0) {
				const session = shellModal.sessions[Math.min(shellModal.selected, shellModal.sessions.length - 1)];
				shellModal.queries = loadSessionQueries(session.id);
				shellModal.expanded = true;
				shellModal.querySelected = 0;
				shellModal.scroll = 0;
			}
			renderNow();
			return;
		}
		if (shellModal.kind === "trace") {
			const allQueries = [...traceHistory];
			if (traceEntries.length > 0) allQueries.push({ query: "(current)", entries: traceEntries });
			if (shellModal.queryExpanded) {
				// Level 3: scrolling trace detail
				const lineStep = lineScrollStep();
				if (key.name === "up") shellModal.scroll = Math.max(0, shellModal.scroll - lineStep);
				if (key.name === "down") shellModal.scroll += lineStep;
				const step = Math.max(4, Math.floor(shellHeight() / 3));
				if (key.name === "pageup") shellModal.scroll = Math.max(0, shellModal.scroll - step);
				if (key.name === "pagedown") shellModal.scroll += step;
				renderNow();
				return;
			}
			if (shellModal.expanded) {
				// Level 2: trace entries for a query — ↑/↓ selects, Enter expands
				const q = allQueries[Math.min(shellModal.selected, allQueries.length - 1)];
				const entries = q?.entries || [];
				if (key.name === "up") shellModal.querySelected = Math.max(0, shellModal.querySelected - 1);
				if (key.name === "down") shellModal.querySelected = Math.min(entries.length - 1, shellModal.querySelected + 1);
				if (key.name === "return" && entries.length > 0) {
					shellModal.queryExpanded = true;
					shellModal.scroll = 0;
				}
				renderNow();
				return;
			}
			// Level 1: query list — ↑/↓ selects, Enter expands
			if (key.name === "up") shellModal.selected = Math.max(0, shellModal.selected - 1);
			if (key.name === "down") shellModal.selected = Math.min(allQueries.length - 1, shellModal.selected + 1);
			if (key.name === "return" && allQueries.length > 0) {
				shellModal.expanded = true;
				shellModal.querySelected = 0;
				shellModal.scroll = 0;
			}
			renderNow();
			return;
		}
		if (shellModal.kind === "help") {
			if (key.name === "pageup") shellModal.scroll += Math.max(4, Math.floor(shellHeight() / 3));
			if (key.name === "pagedown") shellModal.scroll = Math.max(0, shellModal.scroll - Math.max(4, Math.floor(shellHeight() / 3)));
			renderNow();
			return;
		}
	}

	if (key.ctrl && key.name === "o") {
		openTraceModal();
		return;
	}

	if (key.name === "pageup") {
		shellScroll += Math.max(4, Math.floor(shellHeight() / 2));
		requestRender();
		return;
	}
	if (key.name === "pagedown") {
		shellScroll = Math.max(0, shellScroll - Math.max(4, Math.floor(shellHeight() / 2)));
		requestRender();
		return;
	}
	// Slash menu navigation when / is typed
	if (shellInput.startsWith("/") && getSlashMatches().length > 0) {
		const matches = getSlashMatches();
		if (key.name === "escape") {
			shellInput = "";
			shellCursor = 0;
			slashMenuIndex = 0;
			requestRender();
			return;
		}
		if (key.name === "up") {
			slashMenuIndex = Math.max(0, slashMenuIndex - 1);
			requestRender();
			return;
		}
		if (key.name === "down") {
			slashMenuIndex = Math.min(matches.length - 1, slashMenuIndex + 1);
			requestRender();
			return;
		}
		if (key.name === "return" || key.name === "tab") {
			const selected = matches[Math.min(slashMenuIndex, matches.length - 1)];
			if (selected.needsArg) {
				// Needs argument — fill into composer
				shellInput = `${selected.cmd} `;
				shellCursor = shellInput.length;
				slashMenuIndex = 0;
				requestRender();
				return;
			}
			// No argument needed — execute directly
			shellInput = selected.cmd;
			shellCursor = shellInput.length;
			slashMenuIndex = 0;
			await submitCurrentInput();
			return;
		}
	}
	if (key.name === "up" && shellInput.length === 0) {
		shellScroll += lineScrollStep();
		requestRender();
		return;
	}
	if (key.name === "down" && shellInput.length === 0) {
		shellScroll = Math.max(0, shellScroll - lineScrollStep());
		requestRender();
		return;
	}
	if (key.name === "left") {
		shellCursor = Math.max(0, shellCursor - 1);
		requestRender();
		return;
	}
	if (key.name === "right") {
		shellCursor = Math.min(shellInput.length, shellCursor + 1);
		requestRender();
		return;
	}
	if (key.name === "home") {
		shellCursor = 0;
		requestRender();
		return;
	}
	if (key.name === "end") {
		shellCursor = shellInput.length;
		requestRender();
		return;
	}
	if (key.name === "backspace") {
		deleteBackward();
		slashMenuIndex = 0;
		return;
	}
	if (key.name === "delete") {
		if (shellCursor < shellInput.length) {
			shellInput = shellInput.slice(0, shellCursor) + shellInput.slice(shellCursor + 1);
			requestRender();
		}
		return;
	}
	if (key.name === "return") {
		await submitCurrentInput();
		return;
	}
	if (!key.ctrl && !key.meta && str) {
		insertInput(str);
		slashMenuIndex = 0;
	}
}

async function startShell(): Promise<void> {
	shellActive = true;
	shellDestroyed = false;
	readline.emitKeypressEvents(stdin);
	if (stdin.isTTY) stdin.setRawMode(true);
	stdout.write("\x1b[?1049h\x1b[?25l");
	stdin.resume();
	process.stdout.on("resize", requestRender);

	// Fast-path Escape: bypass readline's ESCDELAY (~200-500ms).
	// In local terminals, bare \x1b always arrives as a single byte.
	let _fastEscHandled = false;
	stdin.on("data", (data: Buffer) => {
		if (data.length === 1 && data[0] === 0x1b) {
			_fastEscHandled = true;
			void handleShellKey("\x1b", { name: "escape", sequence: "\x1b" } as readline.Key);
		}
	});
	stdin.on("keypress", (str, key) => {
		if (key.name === "escape" && _fastEscHandled) {
			_fastEscHandled = false;
			return; // already handled by fast path
		}
		void handleShellKey(str, key as readline.Key);
	});
	requestRender();
}

async function interactive(): Promise<void> {
	await loadOllamaModels().catch(() => {});

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
			if (gotKey === null) continue;
			if (!gotKey) {
				setupRl.close();
				process.exit(0);
			}
			currentProviderName = provider.piProvider;
			const defaultModel = getDefaultModelForProvider(provider.piProvider);
			if (defaultModel) {
				currentModelId = defaultModel;
				saveModelPreference(currentModelId);
			}
			setupDone = true;
		}
		setupRl.close();
		if (stdin.isPaused()) stdin.resume();
	}

	const initialResolved = resolveModelWithProvider(currentModelId);
	if (initialResolved) {
		currentModel = initialResolved.model;
		currentProviderName = initialResolved.provider;
	}
	if (!currentModel) {
		const activeProvider = detectProvider();
		const fallbackModel = getDefaultModelForProvider(activeProvider);
		if (fallbackModel) {
			const resolved = resolveModelWithProvider(fallbackModel);
			currentModelId = fallbackModel;
			currentModel = resolved?.model;
			currentProviderName = resolved?.provider || activeProvider;
		}
	}
	if (!currentModel) {
		console.log(`\n  ${c.red}Model "${currentModelId}" not found.${c.reset}\n`);
		process.exit(1);
	}

	contextText = buildCwdContext();
	contextSource = path.basename(process.cwd());
	await startShell();
}

interactive().catch((err) => {
	cleanupShell();
	console.error(`${c.red}Fatal error:${c.reset}`, err);
	process.exit(1);
});
