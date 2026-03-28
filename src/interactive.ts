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
	RESET, BOLD, DIM,
	AMBER, AMBER_DIM, SAGE, ICE, LAVENDER, STONE, ASH, DARK_ASH, ROSE,
	paint, printPanel,
} from "./colors.js";
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
let W = Math.min(process.stdout.columns || 80, 100);

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
	return detectProvider() !== "unknown";
}

/** Returns the pi-ai provider name + model for a given model ID, searching all providers.
 *  Prioritises SETUP_PROVIDERS (with API key) so e.g. "gpt-4o" resolves to "openai" not "azure-openai-responses". */
function resolveModelWithProvider(modelId: string): { model: Model<Api>; provider: string } | undefined {
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

	const provider    = currentProviderName || detectProvider();
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
		["/context",       "show loaded context"],
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
	const provider    = currentProviderName || detectProvider();
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
  ${cmd("/context")}                 Show loaded context info + file list
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

// ── Run RLM query ───────────────────────────────────────────────────────────

async function runQuery(query: string): Promise<void> {
	const effectiveContext = contextText || query;

	if (!currentModel) {
		const resolved = resolveModelWithProvider(currentModelId);
		if (resolved) {
			currentModel = resolved.model;
			currentProviderName = resolved.provider;
		}
	}
	// Safety: verify the current provider still has an API key — re-resolve if not
	if (currentModel && currentProviderName) {
		const key = providerEnvKey(currentProviderName);
		if (!process.env[key]) {
			const resolved = resolveModelWithProvider(currentModelId);
			if (resolved && process.env[providerEnvKey(resolved.provider)]) {
				currentModel = resolved.model;
				currentProviderName = resolved.provider;
			}
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
	const spinner = new Spinner();

	// ── RLM mode ─────────────────────────────────────────────────────────
	let subQueryCount = 0;
	console.log(`\n  ${c.dim}${currentModelId} · ${(effectiveContext.length / 1024).toFixed(1)}KB context${c.reset}`);

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
			subModel: config.sub_model ? resolveModel(config.sub_model) : undefined,
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
					console.log(`\n  ${paint(`◆ Iteration ${info.iteration}`, AMBER, BOLD)}  ${paint(`${elapsed}s elapsed`, DIM)}`);
					stepRule();
					spinner.start("reasoning\u2026");
				}

				if (info.phase === "executing" && info.code) {
					spinner.stop();

					if (currentStep) {
						currentStep.code = info.code;
						currentStep.rawResponse = info.rawResponse || "";
					}

					displayCode(info.code);
					spinner.start("executing\u2026");
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
					const sqLabel = info.subQueries > 0 ? `  ·  ${info.subQueries} sub-quer${info.subQueries !== 1 ? "ies" : "y"}` : "";
					console.log(`\n    ${paint(`✓ ${iterElapsed}s${sqLabel}`, DIM)}`);
				}
			},
			onSubQueryStart: (info: SubQueryStartInfo) => {
				spinner.stop();
				displaySubQueryStart(info);
				spinner.start("querying\u2026");
			},
			onSubQuery: (info: SubQueryInfo) => {
				subQueryCount++;
				if (currentStep) {
					currentStep.subQueries.push(info);
				}

				spinner.stop();
				displaySubQueryResult(info);
				spinner.start("executing\u2026");
			},
		});

		trajectory.result = result;
		trajectory.totalElapsedMs = Date.now() - startTime;

		if (result.completed && trajectory.iterations.length > 0) {
			trajectory.iterations[trajectory.iterations.length - 1].hasFinal = true;
		}

		// Final answer
		const totalSec = ((Date.now() - startTime) / 1000).toFixed(1);
		const stats = `${result.iterations} step${result.iterations !== 1 ? "s" : ""}  ·  ${result.totalSubQueries} sub-quer${result.totalSubQueries !== 1 ? "ies" : "y"}  ·  ${totalSec}s`;

		const answerLines = result.answer.split("\n");
		console.log();
		console.log(boxTop(`${paint("✓ Result", SAGE, BOLD)}  ${paint(stats, DIM)}`, c.green));
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
			err?.name !== "AbortError" &&
			!msg.includes("Aborted") &&
			!msg.includes("not running") &&
			!msg.includes("REPL subprocess") &&
			!msg.includes("REPL shut down")
		) {
			showErrorMsg(msg);
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


function expandAtFiles(input: string): string {
	// Extract all @tokens from input
	const tokens: string[] = [];
	const remaining: string[] = [];

	for (const part of input.split(/\s+/)) {
		if (part.startsWith("@") && part.length > 1) {
			tokens.push(expandTilde(part.slice(1)));
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
			contextText = await safeFetch(url);
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
			currentProviderName = provider.piProvider;
			const defaultModel = getDefaultModelForProvider(provider.piProvider);
			if (defaultModel) {
				currentModelId = defaultModel;
				saveModelPreference(currentModelId);
				console.log(`  ${c.green}✓${c.reset} Default model: ${c.bold}${currentModelId}${c.reset}`);
			}
			console.log();
			setupDone = true;
		}
		setupRl.close();
		// Ensure stdin is resumed after closing the setup readline — some
		// platforms/terminals pause stdin on close, preventing the main REPL
		// readline from receiving input.
		if (stdin.isPaused()) stdin.resume();
	}

	// Resolve model — ensure the resolved provider actually has an API key
	const initialResolved = resolveModelWithProvider(currentModelId);
	if (initialResolved) {
		const resolvedKey = providerEnvKey(initialResolved.provider);
		if (process.env[resolvedKey]) {
			// Provider has a key — use it
			currentModel = initialResolved.model;
			currentProviderName = initialResolved.provider;
		}
	}

	// If default model's provider has no key, fall back to a provider that does
	if (!currentModel) {
		const activeProvider = detectProvider();
		if (activeProvider !== "unknown") {
			const fallbackModel = getDefaultModelForProvider(activeProvider);
			if (fallbackModel) {
				const fallbackResolved = resolveModelWithProvider(fallbackModel);
				if (fallbackResolved) {
					currentModelId = fallbackModel;
					currentModel = fallbackResolved.model;
					currentProviderName = fallbackResolved.provider;
					const label = findSetupProvider(activeProvider)?.name || activeProvider;
					console.log(`  ${c.dim}Using ${label} (${currentModelId})${c.reset}`);
				}
			}
		}
	}

	if (!currentModel) {
		console.log(`\n  ${c.red}Model "${currentModelId}" not found.${c.reset}`);
		console.log(`  Check ${c.bold}RLM_MODEL${c.reset} in your .env file.\n`);
		process.exit(1);
	}

	// Auto-load cwd context so the LLM knows the project structure
	contextText = buildCwdContext();
	contextSource = path.basename(process.cwd());

	printWelcome();

	const rl = readline.createInterface({
		input: stdin,
		output: stdout,
		prompt: `${c.cyan}❯${c.reset} `,
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
				case "f": {
					const fileQuery = await handleFile(arg);
					if (fileQuery && contextText) {
						await runQuery(fileQuery);
						printStatusLine();
					}
					break;
				}
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
					const curProvider = currentProviderName || detectProvider();
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
						saveModelPreference(currentModelId);
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
					const curProvider = currentProviderName || detectProvider();
					const curLabel = findSetupProvider(curProvider)?.name || curProvider;
					console.log(`\n  ${c.bold}Current provider:${c.reset} ${c.cyan}${curLabel}${c.reset}\n`);

					for (let i = 0; i < SETUP_PROVIDERS.length; i++) {
						const p = SETUP_PROVIDERS[i];
						const isCurrent = p.piProvider === curProvider;
						const dot = isCurrent ? `${c.green}●${c.reset}` : ` `;
						const label = isCurrent
							? `${c.cyan}${p.name}${c.reset} ${c.dim}(${p.label})${c.reset}`
							: `${p.name} ${c.dim}(${p.label})${c.reset}`;
						console.log(`  ${c.dim}${i + 1}${c.reset} ${dot} ${label}`);
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
						currentProviderName = provResolved?.provider || chosen.piProvider;
						saveModelPreference(currentModelId);
						console.log(`  ${c.green}✓${c.reset} ${chosen.name} · ${c.bold}${currentModelId}${c.reset}`);
						printStatusLine();
					} else {
						console.log(`  ${c.red}No models available for ${chosen.name}.${c.reset}`);
					}
					break;
				}
				case "key": {
					// Update API key for a provider
					const curProvider = currentProviderName || detectProvider();
					console.log();
					for (let i = 0; i < SETUP_PROVIDERS.length; i++) {
						const p = SETUP_PROVIDERS[i];
						const hasKey = process.env[p.env] ? `${c.green}✓${c.reset}` : `${c.dim}○${c.reset}`;
						console.log(`  ${c.dim}${i + 1}${c.reset} ${hasKey} ${p.name} ${c.dim}(${p.label})${c.reset}`);
					}
					console.log();
					const keyChoice = await questionWithEsc(rl, `  ${c.cyan}Update key for [1-${SETUP_PROVIDERS.length}]:${c.reset} ${c.dim}(ESC to cancel)${c.reset} `);
					if (keyChoice === null || !keyChoice) break;
					const keyIdx = parseInt(keyChoice, 10) - 1;
					if (isNaN(keyIdx) || keyIdx < 0 || keyIdx >= SETUP_PROVIDERS.length) {
						console.log(`  ${c.dim}Cancelled.${c.reset}`);
						break;
					}
					const keyProvider = SETUP_PROVIDERS[keyIdx];
					const newKey = await questionWithEsc(rl, `  ${c.cyan}${keyProvider.env}:${c.reset} `);
					if (newKey === null || !newKey) break;
					const sanitized = newKey.replace(/[\r\n\x00-\x1f]/g, "").trim();
					if (!sanitized) break;
					process.env[keyProvider.env] = sanitized;
					const credPath = path.join(RLM_HOME, "credentials");
					try {
						if (!fs.existsSync(RLM_HOME)) fs.mkdirSync(RLM_HOME, { recursive: true });
						if (fs.existsSync(credPath)) {
							const content = fs.readFileSync(credPath, "utf-8");
							const filtered = content.split("\n").filter((l) => {
								const t = l.trim();
								if (t.startsWith("export ")) return !t.slice(7).startsWith(keyProvider.env + "=");
								return !t.startsWith(keyProvider.env + "=");
							}).join("\n");
							fs.writeFileSync(credPath, filtered.endsWith("\n") ? filtered : filtered + "\n");
						}
						fs.appendFileSync(credPath, `${keyProvider.env}=${sanitized}\n`);
						try { fs.chmodSync(credPath, 0o600); } catch {}
						console.log(`  ${c.green}✓${c.reset} ${keyProvider.name} key updated`);
					} catch {
						console.log(`  ${c.yellow}!${c.reset} Could not save key.`);
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
					contextText = await safeFetch(url);
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

		// Auto-detect bare file/directory paths (tilde, absolute, relative)
		if (!contextText) {
			const tokens = query.split(/\s+/);
			const pathTokens: string[] = [];
			for (const t of tokens) {
				if (looksLikePath(t)) pathTokens.push(t);
				else break;
			}
			if (pathTokens.length > 0) {
				const existing = pathTokens.filter((t) => {
					const p = path.resolve(expandTilde(t));
					return fs.existsSync(p);
				});
				if (existing.length > 0) {
					const filePaths = resolveFileArgs(existing);
					if (filePaths.length === 1) {
						try {
							contextText = fs.readFileSync(filePaths[0], "utf-8");
							contextSource = path.relative(process.cwd(), filePaths[0]) || filePaths[0];
							const lines = contextText.split("\n").length;
							console.log(
								`  ${c.green}✓${c.reset} Loaded ${c.bold}${contextText.length.toLocaleString()}${c.reset} chars (${lines} lines) from ${c.underline}${contextSource}${c.reset}`
							);
						} catch (err: any) {
							console.log(`  ${c.red}Could not read file: ${err.message}${c.reset}`);
						}
					} else if (filePaths.length > 1) {
						const { text, count, totalBytes } = loadMultipleFiles(filePaths);
						contextText = text;
						contextSource = `${count} files`;
						console.log(
							`  ${c.green}✓${c.reset} Loaded ${c.bold}${count}${c.reset} files (${(totalBytes / 1024).toFixed(1)}KB total)`
						);
					}
					if (contextText) {
						query = tokens.slice(pathTokens.length).join(" ") || query;
					}
				}
			}
		}

		// Run query
		await runQuery(query);
		printStatusLine();
		rl.prompt();
	  } catch (err: any) {
		showErrorMsg(String(err?.message || err));
		rl.prompt();
	  }
	});

	// Ctrl+C: abort running query, or double-tap to exit
	let lastSigint = 0;
	rl.on("SIGINT", () => {
		if (isRunning && activeAc) {
			activeSpinner?.stop();
			console.log(`\n  ${c.red}Stopped${c.reset}`);
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
