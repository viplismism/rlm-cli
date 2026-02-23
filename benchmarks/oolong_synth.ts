#!/usr/bin/env tsx
/**
 * Oolong Synth Benchmark — oolongbench/oolong-synth
 *
 * Synthetic long-context tasks: timeline ordering, user tracking, counting.
 * Compares direct LLM vs RLM on the same query.
 *
 * Usage:
 *   npx tsx benchmarks/oolong_synth.ts [--idx 42]
 *
 * Requires: python3 -m venv .venv && .venv/bin/pip install -r benchmarks/requirements.txt
 */

import "../src/env.js";
import { execSync } from "node:child_process";
import { completeSimple, getModels, getProviders } from "@mariozechner/pi-ai";
import { PythonRepl } from "../src/repl.js";
import { runRlmLoop } from "../src/rlm.js";
import * as fs from "node:fs";
import * as path from "node:path";
import type { Api, Model, TextContent } from "@mariozechner/pi-ai";
import type { RlmProgress, SubQueryStartInfo, SubQueryInfo } from "../src/rlm.js";

// Resolve Python: prefer .venv if it exists
const venvPython = path.resolve(process.cwd(), ".venv", "bin", "python3");
const PYTHON = fs.existsSync(venvPython) ? venvPython : "python3";

// ── ANSI + display helpers (matching interactive.ts style) ─────────────────

const c = {
	reset: "\x1b[0m",
	bold: "\x1b[1m",
	dim: "\x1b[2m",
	red: "\x1b[31m",
	green: "\x1b[32m",
	yellow: "\x1b[33m",
	blue: "\x1b[34m",
	magenta: "\x1b[35m",
	cyan: "\x1b[36m",
	gray: "\x1b[90m",
	clearLine: "\x1b[2K\r",
};

const BOX_W = Math.min(process.stdout.columns || 80, 96) - 6;
const MAX_CONTENT_W = BOX_W - 4;

function wrapText(text: string, maxWidth: number): string[] {
	if (text.length <= maxWidth) return [text];
	const result: string[] = [];
	for (let i = 0; i < text.length; i += maxWidth) {
		result.push(text.slice(i, i + maxWidth));
	}
	return result;
}

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

// ── Parse args ─────────────────────────────────────────────────────────────

const args = process.argv.slice(2);
const idxFlag = args.indexOf("--idx");
const idx = idxFlag !== -1 ? parseInt(args[idxFlag + 1], 10) : 42;

console.log(`\n  ${c.cyan}${c.bold}Oolong Synth Benchmark${c.reset}`);
console.log(`  ${c.dim}Dataset: oolongbench/oolong-synth | Index: ${idx}${c.reset}\n`);

// ── Load dataset ───────────────────────────────────────────────────────────

console.log(`  ${c.dim}Loading dataset...${c.reset}`);
const pythonScript = `
import json
from datasets import load_dataset
ds = load_dataset("oolongbench/oolong-synth", split="test")
example = ds[${idx}]
print(json.dumps({
    "context": example["context_window_text_with_labels"],
    "question": example["question"],
    "answer": example["answer"],
    "task_group": example.get("task_group", "unknown"),
}))
`;

let example: { context: string; question: string; answer: string; task_group: string };
try {
	const output = execSync(`${PYTHON} -`, {
		input: pythonScript,
		encoding: "utf-8",
		maxBuffer: 50 * 1024 * 1024,
	});
	example = JSON.parse(output.trim());
} catch (err: any) {
	console.error(`${c.red}Failed to load dataset. Make sure 'datasets' is installed:${c.reset}`);
	console.error(`  python3 -m venv .venv && .venv/bin/pip install -r benchmarks/requirements.txt`);
	console.error(err.message);
	process.exit(1);
}

console.log(`  ${c.green}✓${c.reset} Loaded: task=${c.bold}${example.task_group}${c.reset}, context=${(example.context.length / 1024).toFixed(1)}KB`);
console.log(`  ${c.dim}Question: ${example.question.slice(0, 80)}${example.question.length > 80 ? "..." : ""}${c.reset}`);
console.log(`  ${c.dim}Expected: ${example.answer}${c.reset}\n`);

// ── Resolve model ──────────────────────────────────────────────────────────

const modelId = process.env.RLM_MODEL || "claude-sonnet-4-5-20250929";
let model: Model<Api> | undefined;
for (const provider of getProviders()) {
	for (const m of getModels(provider)) {
		if (m.id === modelId) model = m;
	}
}
if (!model) {
	console.error(`${c.red}Model "${modelId}" not found.${c.reset}`);
	process.exit(1);
}

console.log(`  ${c.dim}Model: ${modelId}${c.reset}\n`);

const fullContext = `${example.question}\n\n${example.context}`;
const ac = new AbortController();

process.on("SIGINT", () => {
	console.log(`\n  ${c.red}Aborted${c.reset}`);
	ac.abort();
	process.exit(1);
});

// ── Run 1: Direct LLM ─────────────────────────────────────────────────────

const directBar = "─".repeat(BOX_W - 2);
console.log(`  ${c.magenta}${directBar}${c.reset}`);
console.log(`  ${c.magenta}${c.bold} Direct LLM${c.reset} ${c.dim}(no RLM)${c.reset}`);
console.log(`  ${c.magenta}${directBar}${c.reset}`);

const directSpinner = new Spinner();
directSpinner.start("Generating response");

const t1 = Date.now();
const directResponse = await completeSimple(model, {
	messages: [
		{
			role: "user",
			content: `Context:\n${example.context}\n\nQuestion: ${example.question}\n\nAnswer the question based on the context above. Be concise.`,
			timestamp: Date.now(),
		},
	],
});
directSpinner.stop();

const directTime = ((Date.now() - t1) / 1000).toFixed(1);
const directText = directResponse.content
	.filter((b): b is TextContent => b.type === "text")
	.map((b) => b.text)
	.join("\n");

console.log(boxTop(`✔ Direct Result  ${c.dim}${directTime}s`, c.magenta));
for (const line of directText.split("\n")) {
	for (const chunk of wrapText(line, MAX_CONTENT_W)) {
		console.log(boxLine(chunk, c.magenta));
	}
}
console.log(boxBottom(c.magenta));
console.log();

// ── Run 2: With RLM ───────────────────────────────────────────────────────

const repl = new PythonRepl();
const spinner = new Spinner();
const t2 = Date.now();
let iterStart = Date.now();

try {
	await repl.start(ac.signal);

	const result = await runRlmLoop({
		context: fullContext,
		query: example.question,
		model,
		repl,
		signal: ac.signal,
		onProgress: (info: RlmProgress) => {
			if (info.phase === "generating_code") {
				iterStart = Date.now();
				const elapsed = ((Date.now() - t2) / 1000).toFixed(1);
				const bar = "─".repeat(BOX_W - 2);
				console.log(`  ${c.blue}${bar}${c.reset}`);
				console.log(`  ${c.blue}${c.bold} Step ${info.iteration}${c.reset}${c.dim}/${info.maxIterations}${c.reset}  ${c.dim}${elapsed}s elapsed${c.reset}`);
				console.log(`  ${c.blue}${bar}${c.reset}`);
				spinner.start("Generating code");
			}

			if (info.phase === "executing" && info.code) {
				spinner.stop();
				displayCode(info.code);
				spinner.start("Executing");
			}

			if (info.phase === "checking_final") {
				spinner.stop();

				if (info.stdout) displayOutput(info.stdout);
				if (info.stderr) displayError(info.stderr);

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
			spinner.stop();
			displaySubQueryResult(info);
			spinner.start("Executing");
		},
	});

	spinner.stop();

	const rlmTime = ((Date.now() - t2) / 1000).toFixed(1);
	const stats = `${result.iterations} step${result.iterations !== 1 ? "s" : ""} · ${result.totalSubQueries} sub-quer${result.totalSubQueries !== 1 ? "ies" : "y"} · ${rlmTime}s`;

	console.log(boxTop(`✔ RLM Result  ${c.dim}${stats}`, c.green));
	for (const line of result.answer.split("\n")) {
		for (const chunk of wrapText(line, MAX_CONTENT_W)) {
			console.log(boxLine(chunk, c.green));
		}
	}
	console.log(boxBottom(c.green));
	console.log();

	// ── Comparison summary ─────────────────────────────────────────────────
	const sumBar = "═".repeat(BOX_W + 2);
	console.log(`  ${c.bold}${sumBar}${c.reset}`);
	console.log(`  ${c.yellow}${c.bold}Expected:${c.reset} ${example.answer}`);
	console.log(`  ${c.bold}${"─".repeat(BOX_W + 2)}${c.reset}`);
	const sumMaxW = (process.stdout.columns || 80) - 4;
	console.log(`  ${c.magenta}${c.bold}Direct LLM${c.reset} ${c.dim}(${directTime}s)${c.reset}`);
	for (const line of directText.split("\n")) {
		for (const chunk of wrapText(line, sumMaxW)) {
			console.log(`  ${chunk}`);
		}
	}
	console.log(`  ${c.bold}${"─".repeat(BOX_W + 2)}${c.reset}`);
	console.log(`  ${c.green}${c.bold}RLM${c.reset} ${c.dim}(${rlmTime}s, ${result.iterations} iters, ${result.totalSubQueries} subs)${c.reset}`);
	for (const line of result.answer.split("\n")) {
		for (const chunk of wrapText(line, sumMaxW)) {
			console.log(`  ${chunk}`);
		}
	}
	console.log(`  ${c.bold}${sumBar}${c.reset}`);

	// Save trajectory
	const trajDir = path.resolve(process.cwd(), "trajectories");
	if (!fs.existsSync(trajDir)) fs.mkdirSync(trajDir, { recursive: true });
	const ts = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
	const trajFile = `benchmark-oolong-idx${idx}-${ts}.json`;
	fs.writeFileSync(
		path.join(trajDir, trajFile),
		JSON.stringify({
			benchmark: "oolong_synth",
			idx,
			expected: example.answer,
			directLlm: { answer: directText, elapsedMs: Date.now() - t1 - (Date.now() - t2) },
			rlm: { answer: result.answer, iterations: result.iterations, subQueries: result.totalSubQueries, elapsedMs: Date.now() - t2 },
		}, null, 2),
		"utf-8"
	);
	console.log(`\n  ${c.dim}Saved: ${trajFile}${c.reset}\n`);
} finally {
	spinner.stop();
	repl.shutdown();
}
