#!/usr/bin/env tsx
/**
 * RLM Trajectory Viewer — interactive TUI for browsing saved trajectory JSON files.
 *
 * Navigate through iterations with arrow keys, view code, REPL output,
 * sub-queries, and the final answer in a beautifully formatted display.
 *
 * Usage:
 *   rlm viewer                              # pick from list
 *   rlm viewer trajectories/file.json       # open specific file
 */

import * as fs from "node:fs";
import * as path from "node:path";
import * as readline from "node:readline";

// ── Types ───────────────────────────────────────────────────────────────────

interface SubQueryEntry {
	index: number;
	contextLength: number;
	instruction: string;
	resultLength: number;
	resultPreview: string;
	elapsedMs?: number;
}

interface TrajectoryStep {
	iteration: number;
	code: string | null;
	stdout: string;
	stderr: string;
	subQueries: SubQueryEntry[];
	hasFinal: boolean;
	elapsedMs: number;
	userMessage?: string;
	rawResponse?: string;
	systemPrompt?: string;
}

interface TrajectoryData {
	model: string;
	query: string;
	contextLength: number;
	contextLines: number;
	startTime: string;
	iterations: TrajectoryStep[];
	result: { answer: string; iterations: number; totalSubQueries: number; completed: boolean } | null;
	totalElapsedMs: number;
}

// ── ANSI helpers ────────────────────────────────────────────────────────────

const c = {
	reset: "\x1b[0m",
	bold: "\x1b[1m",
	dim: "\x1b[2m",
	italic: "\x1b[3m",
	underline: "\x1b[4m",
	inverse: "\x1b[7m",
	red: "\x1b[31m",
	green: "\x1b[32m",
	yellow: "\x1b[33m",
	blue: "\x1b[34m",
	magenta: "\x1b[35m",
	cyan: "\x1b[36m",
	white: "\x1b[37m",
	gray: "\x1b[90m",
	bgBlue: "\x1b[44m",
	bgCyan: "\x1b[46m",
	bgGray: "\x1b[100m",
	clearScreen: "\x1b[2J",
	cursorHome: "\x1b[H",
	hideCursor: "\x1b[?25l",
	showCursor: "\x1b[?25h",
	altScreenOn: "\x1b[?1049h",
	altScreenOff: "\x1b[?1049l",
};

function W(...args: string[]): void {
	process.stdout.write(args.join(""));
}

// ── Layout helpers ──────────────────────────────────────────────────────────

function getWidth(): number {
	return Math.min(process.stdout.columns || 80, 120);
}

function getHeight(): number {
	return process.stdout.rows || 24;
}

function hline(ch = "━", color = c.cyan): string {
	return `${color}${ch.repeat(getWidth())}${c.reset}`;
}

function centeredHeader(text: string, color = c.cyan): string {
	const w = getWidth();
	const stripped = text.replace(/\x1b\[[0-9;]*m/g, "");
	const pad = Math.max(0, w - stripped.length - 4);
	const left = Math.floor(pad / 2);
	const right = pad - left;
	return `${color}${"━".repeat(left)} ${text}${color} ${"━".repeat(right)}${c.reset}`;
}

function boxed(
	title: string,
	content: string,
	color: string,
): void {
	const w = getWidth() - 4;
	const display = content;
	W(`  ${color}${c.bold}${title}${c.reset}\n`);
	W(`  ${color}┌${"─".repeat(w)}┐${c.reset}\n`);
	for (const line of display.split("\n")) {
		const stripped = line.replace(/\x1b\[[0-9;]*m/g, "");
		const padding = Math.max(0, w - stripped.length - 1);
		W(`  ${color}│${c.reset} ${line}${" ".repeat(padding)}${color}│${c.reset}\n`);
	}
	W(`  ${color}└${"─".repeat(w)}┘${c.reset}\n`);
}

function kvLine(key: string, value: string): void {
	W(`  ${c.gray}${key}:${c.reset} ${value}\n`);
}

function formatSize(chars: number): string {
	if (chars >= 1_000_000) return `${(chars / 1_000_000).toFixed(1)}M`;
	if (chars >= 1000) return `${(chars / 1000).toFixed(1)}K`;
	return `${chars}`;
}

// ── File picker ─────────────────────────────────────────────────────────────

interface FileEntry {
	name: string;
	path: string;
	size: number;
	mtime: Date;
	traj?: TrajectoryData;
}

function listTrajectories(): FileEntry[] {
	const dir = path.resolve(process.cwd(), "trajectories");
	if (!fs.existsSync(dir)) return [];
	return fs
		.readdirSync(dir)
		.filter((f) => f.endsWith(".json"))
		.map((f) => {
			const full = path.join(dir, f);
			const stat = fs.statSync(full);
			let traj: TrajectoryData | undefined;
			try {
				traj = JSON.parse(fs.readFileSync(full, "utf-8"));
			} catch { /* skip */ }
			return { name: f, path: full, size: stat.size, mtime: stat.mtime, traj };
		})
		.sort((a, b) => b.mtime.getTime() - a.mtime.getTime()); // newest first
}

async function pickFile(files: FileEntry[]): Promise<string> {
	return new Promise((resolve) => {
		let selected = 0;
		const maxVisible = Math.min(files.length, getHeight() - 10);

		function render(): void {
			W(c.cursorHome, c.clearScreen, c.hideCursor);
			W(`\n${hline()}\n`);
			W(`${centeredHeader(`${c.bold}${c.white}RLM Trajectory Viewer${c.reset}`)}\n`);
			W(`${hline()}\n\n`);
			W(`  ${c.bold}Select a trajectory:${c.reset}  ${c.dim}(up/down navigate, enter select, q quit)${c.reset}\n\n`);

			const scrollStart = Math.max(0, selected - Math.floor(maxVisible / 2));
			const scrollEnd = Math.min(files.length, scrollStart + maxVisible);

			for (let i = scrollStart; i < scrollEnd; i++) {
				const f = files[i];
				const isSel = i === selected;
				const sizeKB = (f.size / 1024).toFixed(1);

				// Extract info from trajectory data
				const steps = f.traj?.iterations?.length ?? 0;
				const completed = f.traj?.result?.completed;
				const status = completed === true ? `${c.green}done${c.reset}` : completed === false ? `${c.yellow}partial${c.reset}` : "";
				const queryPreview = f.traj?.query
					? (f.traj.query.length > 40 ? f.traj.query.slice(0, 37) + "..." : f.traj.query)
					: "";

				// Date from filename
				const dateMatch = f.name.match(/(\d{4}-\d{2}-\d{2})T(\d{2})-(\d{2})/);
				const dateStr = dateMatch ? `${dateMatch[1]} ${dateMatch[2]}:${dateMatch[3]}` : f.name;

				const prefix = isSel ? `${c.cyan}${c.bold}  > ` : `    `;
				const nameColor = isSel ? `${c.cyan}${c.bold}` : c.white;

				W(`${prefix}${nameColor}${dateStr}${c.reset}`);
				W(`  ${c.dim}${sizeKB}KB${c.reset}`);
				W(`  ${c.dim}${steps} step${steps !== 1 ? "s" : ""}${c.reset}`);
				if (status) W(`  ${status}`);
				if (queryPreview) W(`  ${c.dim}${queryPreview}${c.reset}`);
				W(`\n`);
			}

			if (files.length > maxVisible) {
				W(`\n  ${c.dim}${scrollStart > 0 ? "^ more above" : ""}  ${scrollEnd < files.length ? "v more below" : ""}${c.reset}\n`);
			}
			W(`\n`);
		}

		render();

		process.stdin.setRawMode(true);
		process.stdin.resume();
		process.stdin.setEncoding("utf8");

		function onKey(key: string): void {
			if (key === "\x1b[A") {
				selected = Math.max(0, selected - 1);
				render();
			} else if (key === "\x1b[B") {
				selected = Math.min(files.length - 1, selected + 1);
				render();
			} else if (key === "\r" || key === "\n") {
				process.stdin.removeListener("data", onKey);
				process.stdin.setRawMode(false);
				process.stdin.pause();
				resolve(files[selected].path);
			} else if (key === "q" || key === "\x03") {
				W(c.showCursor, c.altScreenOff);
				process.exit(0);
			}
		}

		process.stdin.on("data", onKey);
	});
}

// ── Rendering views ─────────────────────────────────────────────────────────

type ViewMode = "overview" | "iteration" | "result" | "subqueries" | "subqueryDetail" | "llmInput" | "llmResponse" | "systemPrompt";

interface ViewState {
	mode: ViewMode;
	iterIdx: number; // 0-based index into iterations
	subQueryIdx: number; // 0-based index into sub-queries for current iteration
	scrollY: number;
	traj: TrajectoryData;
}

function buildIterLine(step: TrajectoryStep, isSelected: boolean): string {
	const isFinal = step.hasFinal;
	const elapsed = (step.elapsedMs / 1000).toFixed(1);
	const sqCount = step.subQueries.length;

	const bullet = isFinal ? `${c.green}${c.bold}*${c.reset}` : `${c.blue}o${c.reset}`;
	const sel = isSelected ? `${c.inverse}${c.cyan}` : "";
	const codeLen = step.code ? step.code.split("\n").length : 0;
	const outLen = step.stdout ? step.stdout.split("\n").length : 0;
	const sqInfo = sqCount > 0 ? ` | ${c.magenta}${sqCount} sub-quer${sqCount !== 1 ? "ies" : "y"}${c.reset}` : "";
	const errInfo = step.stderr ? ` | ${c.red}stderr${c.reset}` : "";

	let line = `  ${sel} ${bullet} ${c.bold}Iteration ${step.iteration}${c.reset}`;
	line += `${sel ? c.reset : ""} ${c.dim}${elapsed}s${c.reset}`;
	line += ` | ${c.green}${codeLen}L code${c.reset} | ${c.yellow}${outLen}L output${c.reset}${sqInfo}${errInfo}`;
	if (isFinal) line += ` | ${c.green}${c.bold}FINAL${c.reset}`;
	return line;
}

function renderOverview(state: ViewState): void {
	const { traj } = state;
	const w = getWidth();
	const h = getHeight();

	// Build all lines into a buffer
	const buf: string[] = [];

	// Header
	buf.push(``);
	buf.push(hline());
	buf.push(centeredHeader(`${c.bold}${c.white}RLM Trajectory Viewer${c.reset}`));
	buf.push(hline());
	buf.push(``);
	buf.push(`  ${c.gray}Model   :${c.reset} ${c.bold}${traj.model}${c.reset}`);
	buf.push(`  ${c.gray}Query   :${c.reset} ${c.yellow}${traj.query}${c.reset}`);
	buf.push(`  ${c.gray}Context :${c.reset} ${traj.contextLength.toLocaleString()} chars | ${traj.contextLines.toLocaleString()} lines`);
	buf.push(`  ${c.gray}Duration:${c.reset} ${(traj.totalElapsedMs / 1000).toFixed(1)}s  ${c.gray}|${c.reset}  ${traj.result?.completed ? `${c.green}Completed${c.reset}` : `${c.red}Incomplete${c.reset}`}`);
	buf.push(``);
	buf.push(`  ${c.bold}Iterations${c.reset}  ${c.dim}(${traj.iterations.length} total)${c.reset}`);
	buf.push(``);

	const headerSize = buf.length;
	const footerSize = 2;
	const answerSize = traj.result ? 4 : 0;
	const iterBudget = h - headerSize - footerSize - answerSize;

	// Build iteration lines (each iteration = summary + separator)
	const flatLines: string[] = [];
	const iterStartOffsets: number[] = [];
	for (let i = 0; i < traj.iterations.length; i++) {
		const step = traj.iterations[i];
		const isSel = i === state.iterIdx;
		iterStartOffsets.push(flatLines.length);
		flatLines.push(buildIterLine(step, isSel));
		if (i < traj.iterations.length - 1) {
			flatLines.push(`  ${c.dim}  |${c.reset}`);
		}
	}

	// Scroll so selected iteration is visible
	const selStart = iterStartOffsets[state.iterIdx] ?? 0;
	let scrollY = Math.max(0, selStart - 2);

	// If everything fits, no scroll needed
	if (flatLines.length <= iterBudget) {
		scrollY = 0;
	}

	const showFrom = scrollY;
	const showTo = Math.min(flatLines.length, scrollY + iterBudget);

	if (showFrom > 0) {
		buf.push(`  ${c.dim}  ^ more above${c.reset}`);
	}

	for (let i = showFrom; i < showTo; i++) {
		buf.push(flatLines[i]);
	}

	if (showTo < flatLines.length) {
		buf.push(`  ${c.dim}  | more below${c.reset}`);
	}

	// Answer preview
	if (traj.result) {
		buf.push(`${c.green}${"─".repeat(w)}${c.reset}`);
		buf.push(`  ${c.green}${c.bold}Answer Preview:${c.reset}`);
		const preview = traj.result.answer.split("\n")[0] || "";
		buf.push(`  ${c.white}${preview}${c.reset}`);
		if (traj.result.answer.split("\n").length > 1) {
			buf.push(`  ${c.dim}... (press 'r' to see full result)${c.reset}`);
		}
	}

	// Render
	W(c.cursorHome, c.clearScreen, c.hideCursor);
	for (const l of buf) W(l + "\n");

	// Footer
	W(hline("─", c.gray) + "\n");
	W(`  ${c.dim}up/down${c.reset} select  ${c.dim}enter${c.reset} view  ${c.dim}r${c.reset} result  ${c.dim}q${c.reset} quit\n`);
}

function buildIterationContent(step: TrajectoryStep, traj: TrajectoryData): string[] {
	const lines: string[] = [];
	const w = getWidth() - 4;

	// Title
	lines.push(``);
	lines.push(hline());
	const finalTag = step.hasFinal ? `  ${c.green}${c.bold}FINAL${c.reset}` : "";
	lines.push(centeredHeader(`${c.bold}${c.white}Iteration ${step.iteration} / ${traj.iterations.length}${c.reset}${finalTag}`));
	lines.push(hline());
	lines.push(``);

	// Metadata
	const elapsed = (step.elapsedMs / 1000).toFixed(1);
	lines.push(`  ${c.gray}Elapsed    :${c.reset} ${elapsed}s`);
	lines.push(`  ${c.gray}Sub-queries:${c.reset} ${step.subQueries.length}`);
	lines.push(`  ${c.gray}Has Final  :${c.reset} ${step.hasFinal ? `${c.green}yes${c.reset}` : `${c.gray}no${c.reset}`}`);
	lines.push(``);

	// Code
	if (step.code) {
		lines.push(`  ${c.green}${c.bold}Generated Code${c.reset}`);
		lines.push(`  ${c.green}┌${"─".repeat(w)}┐${c.reset}`);
		for (const cl of syntaxHighlight(step.code).split("\n")) {
			const stripped = cl.replace(/\x1b\[[0-9;]*m/g, "");
			const padding = Math.max(0, w - stripped.length - 1);
			lines.push(`  ${c.green}│${c.reset} ${cl}${" ".repeat(padding)}${c.green}│${c.reset}`);
		}
		lines.push(`  ${c.green}└${"─".repeat(w)}┘${c.reset}`);
		lines.push(``);
	}

	// REPL Output
	if (step.stdout) {
		lines.push(`  ${c.yellow}${c.bold}REPL Output${c.reset}`);
		lines.push(`  ${c.yellow}┌${"─".repeat(w)}┐${c.reset}`);
		for (const ol of step.stdout.split("\n")) {
			const stripped = ol.replace(/\x1b\[[0-9;]*m/g, "");
			const padding = Math.max(0, w - stripped.length - 1);
			lines.push(`  ${c.yellow}│${c.reset} ${ol}${" ".repeat(padding)}${c.yellow}│${c.reset}`);
		}
		lines.push(`  ${c.yellow}└${"─".repeat(w)}┘${c.reset}`);
		lines.push(``);
	}

	// Stderr
	if (step.stderr) {
		lines.push(`  ${c.red}${c.bold}Stderr${c.reset}`);
		lines.push(`  ${c.red}┌${"─".repeat(w)}┐${c.reset}`);
		for (const el of step.stderr.split("\n")) {
			const stripped = el.replace(/\x1b\[[0-9;]*m/g, "");
			const padding = Math.max(0, w - stripped.length - 1);
			lines.push(`  ${c.red}│${c.reset} ${el}${" ".repeat(padding)}${c.red}│${c.reset}`);
		}
		lines.push(`  ${c.red}└${"─".repeat(w)}┘${c.reset}`);
		lines.push(``);
	}

	// Sub-queries
	if (step.subQueries.length > 0) {
		lines.push(`  ${c.magenta}${c.bold}Sub-queries (${step.subQueries.length})${c.reset}  ${c.dim}press 's' for details${c.reset}`);
		for (const sq of step.subQueries) {
			const instrPreview = sq.instruction.length > 60 ? sq.instruction.slice(0, 57) + "..." : sq.instruction;
			const sqElapsed = sq.elapsedMs ? `  ${c.dim}${(sq.elapsedMs / 1000).toFixed(1)}s${c.reset}` : "";
			lines.push(`    ${c.magenta}#${sq.index}${c.reset} ${c.dim}(${formatSize(sq.contextLength)})${c.reset}${sqElapsed} ${instrPreview}`);
		}
		lines.push(``);
	}

	return lines;
}

function renderIteration(state: ViewState): void {
	const { traj, iterIdx } = state;
	const step = traj.iterations[iterIdx];
	if (!step) return;

	const allLines = buildIterationContent(step, traj);

	const h = getHeight();
	const footerSize = 2;
	const viewable = h - footerSize;

	// Clamp scrollY
	const maxScroll = Math.max(0, allLines.length - viewable);
	if (state.scrollY > maxScroll) state.scrollY = maxScroll;
	if (state.scrollY < 0) state.scrollY = 0;

	const from = state.scrollY;
	const to = Math.min(allLines.length, from + viewable);

	W(c.cursorHome, c.clearScreen, c.hideCursor);

	// Scroll indicator at top
	if (from > 0) {
		W(`  ${c.dim}^ scroll up (${from} lines above)${c.reset}\n`);
		for (let i = from + 1; i < to; i++) W(allLines[i] + "\n");
	} else {
		for (let i = from; i < to; i++) W(allLines[i] + "\n");
	}

	if (to < allLines.length) {
		// Replace last visible line with scroll indicator
		W(`  ${c.dim}v scroll down (${allLines.length - to} lines below)${c.reset}\n`);
	}

	// Footer
	const hints: string[] = [];
	if (step.userMessage) hints.push(`${c.dim}i${c.reset} input`);
	if (step.rawResponse) hints.push(`${c.dim}l${c.reset} response`);
	if (step.systemPrompt || traj.iterations[0]?.systemPrompt) hints.push(`${c.dim}p${c.reset} prompt`);

	W(hline("─", c.gray) + "\n");
	W(`  ${c.dim}esc${c.reset} back  `);
	W(`${c.dim}up/down${c.reset} scroll  `);
	W(`${c.dim}n/p${c.reset} next/prev  `);
	if (step.subQueries.length > 0) W(`${c.dim}s${c.reset} sub-queries  `);
	for (const hint of hints) W(`${hint}  `);
	W(`${c.dim}r${c.reset} result  `);
	W(`${c.dim}q${c.reset} quit\n`);
}

function renderResult(state: ViewState): void {
	const { traj } = state;
	const result = traj.result;

	W(c.cursorHome, c.clearScreen, c.hideCursor);

	W(`\n${hline("━", c.green)}\n`);
	W(`${centeredHeader(`${c.bold}${c.white}Final Result${c.reset}`, c.green)}\n`);
	W(`${hline("━", c.green)}\n\n`);

	if (!result) {
		W(`  ${c.red}${c.bold}No result available${c.reset} — the run may have been interrupted.\n`);
	} else {
		kvLine("Completed   ", result.completed ? `${c.green}yes${c.reset}` : `${c.red}no${c.reset}`);
		kvLine("Iterations  ", `${result.iterations}`);
		kvLine("Sub-queries ", `${result.totalSubQueries}`);
		kvLine("Duration    ", `${(traj.totalElapsedMs / 1000).toFixed(1)}s`);
		W(`\n`);

		boxed("Answer", result.answer, c.green);
	}

	W(`\n${hline("─", c.gray)}\n`);
	W(`  ${c.dim}esc${c.reset} back  `);
	W(`${c.dim}q${c.reset} quit\n`);
}

function renderSubQueries(state: ViewState): void {
	const { traj, iterIdx } = state;
	const step = traj.iterations[iterIdx];
	if (!step) return;

	const h = getHeight();

	// Build buffer
	const buf: string[] = [];

	buf.push(``);
	buf.push(hline("━", c.magenta));
	buf.push(centeredHeader(`${c.bold}${c.white}Sub-queries — Iteration ${step.iteration}${c.reset}`, c.magenta));
	buf.push(hline("━", c.magenta));
	buf.push(``);

	if (step.subQueries.length === 0) {
		buf.push(`  ${c.dim}No sub-queries in this iteration.${c.reset}`);
	} else {
		// Clamp subQueryIdx
		if (state.subQueryIdx >= step.subQueries.length) state.subQueryIdx = step.subQueries.length - 1;
		if (state.subQueryIdx < 0) state.subQueryIdx = 0;

		const headerSize = buf.length;
		const footerSize = 2;
		const listBudget = h - headerSize - footerSize;

		// Build list lines (each sub-query = 2 lines: summary + separator)
		const listLines: string[] = [];
		const sqStartOffsets: number[] = [];
		for (let i = 0; i < step.subQueries.length; i++) {
			const sq = step.subQueries[i];
			const isSel = i === state.subQueryIdx;
			const sqElapsed = sq.elapsedMs ? `${(sq.elapsedMs / 1000).toFixed(1)}s` : "";
			const instrPreview = sq.instruction.length > 50 ? sq.instruction.slice(0, 47) + "..." : sq.instruction;

			sqStartOffsets.push(listLines.length);

			const sel = isSel ? `${c.inverse}${c.magenta}` : "";
			const prefix = isSel ? `${c.magenta}${c.bold}  > ` : `    `;
			let line = `${prefix}${sel}#${sq.index}${c.reset}`;
			line += `  ${c.dim}${sqElapsed}${c.reset}`;
			line += `  ${c.dim}${formatSize(sq.contextLength)} in, ${formatSize(sq.resultLength)} out${c.reset}`;
			line += `  ${instrPreview}`;
			listLines.push(line);

			if (i < step.subQueries.length - 1) {
				listLines.push(`  ${c.dim}  |${c.reset}`);
			}
		}

		// Scroll so selected sub-query is visible
		const selStart = sqStartOffsets[state.subQueryIdx] ?? 0;
		let scrollY = Math.max(0, selStart - 2);
		if (listLines.length <= listBudget) scrollY = 0;

		const showFrom = scrollY;
		const showTo = Math.min(listLines.length, scrollY + listBudget);

		if (showFrom > 0) {
			buf.push(`  ${c.dim}  ^ more above${c.reset}`);
		}
		for (let i = showFrom; i < showTo; i++) {
			buf.push(listLines[i]);
		}
		if (showTo < listLines.length) {
			buf.push(`  ${c.dim}  | more below${c.reset}`);
		}
	}

	// Render
	W(c.cursorHome, c.clearScreen, c.hideCursor);
	for (const l of buf) W(l + "\n");

	// Footer
	W(hline("─", c.gray) + "\n");
	W(`  ${c.dim}up/down${c.reset} select  ${c.dim}enter${c.reset} view  ${c.dim}esc${c.reset} back  ${c.dim}q${c.reset} quit\n`);
}

function renderSubQueryDetail(state: ViewState): void {
	const { traj, iterIdx } = state;
	const step = traj.iterations[iterIdx];
	if (!step) return;

	// Clamp subQueryIdx
	if (state.subQueryIdx >= step.subQueries.length) state.subQueryIdx = step.subQueries.length - 1;
	if (state.subQueryIdx < 0) state.subQueryIdx = 0;
	const sq = step.subQueries[state.subQueryIdx];
	if (!sq) return;

	const w = getWidth() - 4;
	const h = getHeight();

	// Build all content lines
	const allLines: string[] = [];

	allLines.push(``);
	allLines.push(hline("━", c.magenta));
	allLines.push(centeredHeader(
		`${c.bold}${c.white}Sub-query #${sq.index} — Iteration ${step.iteration}${c.reset}`,
		c.magenta,
	));
	allLines.push(hline("━", c.magenta));
	allLines.push(``);

	// Metadata
	const sqElapsed = sq.elapsedMs ? `${(sq.elapsedMs / 1000).toFixed(1)}s` : "n/a";
	allLines.push(`  ${c.gray}Elapsed       :${c.reset} ${sqElapsed}`);
	allLines.push(`  ${c.gray}Context length:${c.reset} ${formatSize(sq.contextLength)} chars`);
	allLines.push(`  ${c.gray}Result length :${c.reset} ${formatSize(sq.resultLength)} chars`);
	allLines.push(`  ${c.gray}Position      :${c.reset} ${state.subQueryIdx + 1} of ${step.subQueries.length}`);
	allLines.push(``);

	// Full instruction (boxed, no truncation)
	allLines.push(`  ${c.magenta}${c.bold}Instruction${c.reset}`);
	allLines.push(`  ${c.magenta}┌${"─".repeat(w)}┐${c.reset}`);
	for (const line of sq.instruction.split("\n")) {
		const stripped = line.replace(/\x1b\[[0-9;]*m/g, "");
		const padding = Math.max(0, w - stripped.length - 1);
		allLines.push(`  ${c.magenta}│${c.reset} ${line}${" ".repeat(padding)}${c.magenta}│${c.reset}`);
	}
	allLines.push(`  ${c.magenta}└${"─".repeat(w)}┘${c.reset}`);
	allLines.push(``);

	// Full result preview (boxed, no truncation)
	allLines.push(`  ${c.cyan}${c.bold}Result Preview${c.reset}`);
	allLines.push(`  ${c.cyan}┌${"─".repeat(w)}┐${c.reset}`);
	for (const line of sq.resultPreview.split("\n")) {
		const stripped = line.replace(/\x1b\[[0-9;]*m/g, "");
		const padding = Math.max(0, w - stripped.length - 1);
		allLines.push(`  ${c.cyan}│${c.reset} ${line}${" ".repeat(padding)}${c.cyan}│${c.reset}`);
	}
	allLines.push(`  ${c.cyan}└${"─".repeat(w)}┘${c.reset}`);
	allLines.push(``);

	// Scrollable rendering
	const footerSize = 2;
	const viewable = h - footerSize;

	const maxScroll = Math.max(0, allLines.length - viewable);
	if (state.scrollY > maxScroll) state.scrollY = maxScroll;
	if (state.scrollY < 0) state.scrollY = 0;

	const from = state.scrollY;
	const to = Math.min(allLines.length, from + viewable);

	W(c.cursorHome, c.clearScreen, c.hideCursor);

	if (from > 0) {
		W(`  ${c.dim}^ scroll up (${from} lines above)${c.reset}\n`);
		for (let i = from + 1; i < to; i++) W(allLines[i] + "\n");
	} else {
		for (let i = from; i < to; i++) W(allLines[i] + "\n");
	}

	if (to < allLines.length) {
		W(`  ${c.dim}v scroll down (${allLines.length - to} lines below)${c.reset}\n`);
	}

	// Footer
	W(hline("─", c.gray) + "\n");
	W(`  ${c.dim}up/down${c.reset} scroll  ${c.dim}n/p${c.reset} next/prev  ${c.dim}esc${c.reset} back  ${c.dim}q${c.reset} quit\n`);
}

function renderLlmInput(state: ViewState): void {
	const { traj, iterIdx } = state;
	const step = traj.iterations[iterIdx];
	if (!step) return;

	W(c.cursorHome, c.clearScreen, c.hideCursor);

	W(`\n${hline("━", c.blue)}\n`);
	W(`${centeredHeader(`${c.bold}${c.white}LLM Input — Iteration ${step.iteration}${c.reset}`, c.blue)}\n`);
	W(`${hline("━", c.blue)}\n\n`);

	if (step.userMessage) {
		kvLine("Length", `${step.userMessage.length.toLocaleString()} chars`);
		W(`\n`);
		boxed("User Message", step.userMessage, c.blue);
	} else {
		W(`  ${c.dim}No user message recorded for this iteration.${c.reset}\n`);
	}

	W(`\n${hline("─", c.gray)}\n`);
	W(`  ${c.dim}esc${c.reset} back  `);
	W(`${c.dim}q${c.reset} quit\n`);
}

function renderLlmResponse(state: ViewState): void {
	const { traj, iterIdx } = state;
	const step = traj.iterations[iterIdx];
	if (!step) return;

	W(c.cursorHome, c.clearScreen, c.hideCursor);

	W(`\n${hline("━", c.green)}\n`);
	W(`${centeredHeader(`${c.bold}${c.white}LLM Response — Iteration ${step.iteration}${c.reset}`, c.green)}\n`);
	W(`${hline("━", c.green)}\n\n`);

	if (step.rawResponse) {
		kvLine("Length", `${step.rawResponse.length.toLocaleString()} chars`);
		W(`\n`);
		boxed("Full LLM Response", step.rawResponse, c.green);
	} else {
		W(`  ${c.dim}No response recorded for this iteration.${c.reset}\n`);
	}

	W(`\n${hline("─", c.gray)}\n`);
	W(`  ${c.dim}esc${c.reset} back  `);
	W(`${c.dim}q${c.reset} quit\n`);
}

function renderSystemPrompt(state: ViewState): void {
	const { traj, iterIdx } = state;
	const step = traj.iterations[iterIdx];
	if (!step) return;

	W(c.cursorHome, c.clearScreen, c.hideCursor);

	W(`\n${hline("━", c.cyan)}\n`);
	W(`${centeredHeader(`${c.bold}${c.white}System Prompt${c.reset}`, c.cyan)}\n`);
	W(`${hline("━", c.cyan)}\n\n`);

	const sysPrompt = step.systemPrompt || traj.iterations[0]?.systemPrompt;
	if (sysPrompt) {
		boxed("System Prompt", sysPrompt, c.cyan);
	} else {
		W(`  ${c.dim}System prompt not recorded in this trajectory.${c.reset}\n`);
	}

	W(`\n${hline("─", c.gray)}\n`);
	W(`  ${c.dim}esc${c.reset} back  `);
	W(`${c.dim}q${c.reset} quit\n`);
}

// ── Minimal syntax highlighting ─────────────────────────────────────────────

function syntaxHighlight(code: string): string {
	return code
		.replace(
			/\b(import|from|def|class|return|if|elif|else|for|while|in|not|and|or|try|except|finally|with|as|raise|pass|break|continue|yield|lambda|True|False|None|await|async)\b/g,
			`${c.magenta}$1${c.reset}`,
		)
		.replace(
			/\b(print|len|range|enumerate|sorted|set|list|dict|str|int|float|type|isinstance|zip|map|filter)\b/g,
			`${c.cyan}$1${c.reset}`,
		)
		.replace(/(#.*)$/gm, `${c.gray}$1${c.reset}`)
		.replace(/("""[\s\S]*?"""|'''[\s\S]*?'''|"[^"]*"|'[^']*')/g, `${c.yellow}$1${c.reset}`)
		.replace(/\b(llm_query|async_llm_query|context|FINAL|FINAL_VAR)\b/g, `${c.green}${c.bold}$1${c.reset}`);
}

// ── Main interactive loop ───────────────────────────────────────────────────

async function main(): Promise<void> {
	// Enter alternate screen buffer so output never scrolls the main terminal
	W(c.altScreenOn);

	// Ensure we always leave alt screen on exit
	const cleanup = () => W(c.showCursor, c.altScreenOff);
	process.on("exit", cleanup);

	let filePath: string | undefined = process.argv[2];

	if (!filePath) {
		const files = listTrajectories();
		if (files.length === 0) {
			console.error(
				`${c.red}No trajectory files found in ./trajectories/${c.reset}\nRun a query first to generate trajectories.`,
			);
			process.exit(1);
		}
		filePath = await pickFile(files);
	}

	// Load trajectory
	if (!fs.existsSync(filePath)) {
		console.error(`${c.red}File not found: ${filePath}${c.reset}`);
		process.exit(1);
	}
	const traj: TrajectoryData = JSON.parse(fs.readFileSync(filePath, "utf-8"));

	if (!traj.iterations || traj.iterations.length === 0) {
		console.error(`${c.red}Trajectory has no iterations (empty run).${c.reset}`);
		process.exit(1);
	}

	// State
	const state: ViewState = {
		mode: "overview",
		iterIdx: 0,
		subQueryIdx: 0,
		scrollY: 0,
		traj,
	};

	function render(): void {
		switch (state.mode) {
			case "overview":
				renderOverview(state);
				break;
			case "iteration":
				renderIteration(state);
				break;
			case "result":
				renderResult(state);
				break;
			case "subqueries":
				renderSubQueries(state);
				break;
			case "subqueryDetail":
				renderSubQueryDetail(state);
				break;
			case "llmInput":
				renderLlmInput(state);
				break;
			case "llmResponse":
				renderLlmResponse(state);
				break;
			case "systemPrompt":
				renderSystemPrompt(state);
				break;
		}
	}

	render();

	// Key handling
	process.stdin.setRawMode(true);
	process.stdin.resume();
	process.stdin.setEncoding("utf8");

	process.stdin.on("data", (key: string) => {
		const maxIter = traj.iterations.length - 1;

		switch (state.mode) {
			case "overview":
				if (key === "\x1b[A") {
					state.iterIdx = Math.max(0, state.iterIdx - 1);
				} else if (key === "\x1b[B") {
					state.iterIdx = Math.min(maxIter, state.iterIdx + 1);
				} else if (key === "\r" || key === "\n" || key === "\x1b[C") {
					// Drill into iteration detail
					state.mode = "iteration";
					state.scrollY = 0;
				} else if (key === "r") {
					state.mode = "result";
				} else if (key === "q" || key === "\x03") {
					W(c.showCursor, "\n");
					process.exit(0);
				}
				break;

			case "iteration":
				if (key === "\x1b[A") {
					state.scrollY = Math.max(0, state.scrollY - 3);
				} else if (key === "\x1b[B") {
					state.scrollY += 3;
				} else if (key === "n") {
					if (state.iterIdx < maxIter) {
						state.iterIdx++;
						state.scrollY = 0;
					}
				} else if (key === "N") {
					if (state.iterIdx > 0) {
						state.iterIdx--;
						state.scrollY = 0;
					}
				} else if (key === "\x1b[D" || key === "\x1b" || key === "b") {
					state.mode = "overview";
					state.scrollY = 0;
				} else if (key === "s" && traj.iterations[state.iterIdx]?.subQueries.length > 0) {
					state.mode = "subqueries";
					state.subQueryIdx = 0;
				} else if (key === "i") {
					state.mode = "llmInput";
				} else if (key === "l") {
					state.mode = "llmResponse";
				} else if (key === "p") {
					state.mode = "systemPrompt";
				} else if (key === "r") {
					state.mode = "result";
				} else if (key === "q" || key === "\x03") {
					W(c.showCursor, "\n");
					process.exit(0);
				}
				break;

			case "result":
				if (key === "\x1b[D" || key === "\x1b" || key === "b") {
					state.mode = "overview";
				} else if (key === "q" || key === "\x03") {
					W(c.showCursor, "\n");
					process.exit(0);
				}
				break;

			case "subqueries": {
				const sqCount = traj.iterations[state.iterIdx]?.subQueries.length ?? 0;
				if (key === "\x1b[A") {
					state.subQueryIdx = Math.max(0, state.subQueryIdx - 1);
				} else if (key === "\x1b[B") {
					state.subQueryIdx = Math.min(sqCount - 1, state.subQueryIdx + 1);
				} else if (key === "\r" || key === "\n" || key === "\x1b[C") {
					state.mode = "subqueryDetail";
					state.scrollY = 0;
				} else if (key === "\x1b[D" || key === "\x1b" || key === "b") {
					state.mode = "iteration";
				} else if (key === "q" || key === "\x03") {
					W(c.showCursor, "\n");
					process.exit(0);
				}
				break;
			}

			case "subqueryDetail": {
				const sqMax = (traj.iterations[state.iterIdx]?.subQueries.length ?? 1) - 1;
				if (key === "\x1b[A") {
					state.scrollY = Math.max(0, state.scrollY - 3);
				} else if (key === "\x1b[B") {
					state.scrollY += 3;
				} else if (key === "n" || key === "\x1b[C") {
					if (state.subQueryIdx < sqMax) {
						state.subQueryIdx++;
						state.scrollY = 0;
					}
				} else if (key === "p" || key === "N") {
					if (state.subQueryIdx > 0) {
						state.subQueryIdx--;
						state.scrollY = 0;
					}
				} else if (key === "\x1b[D" || key === "\x1b" || key === "b") {
					state.mode = "subqueries";
					state.scrollY = 0;
				} else if (key === "q" || key === "\x03") {
					W(c.showCursor, "\n");
					process.exit(0);
				}
				break;
			}

			case "llmInput":
			case "llmResponse":
			case "systemPrompt":
				if (key === "\x1b[D" || key === "\x1b" || key === "b") {
					state.mode = "iteration";
				} else if (key === "q" || key === "\x03") {
					W(c.showCursor, "\n");
					process.exit(0);
				}
				break;
		}

		render();
	});

	// (cleanup handler already registered at top of main)
}

main().catch((err) => {
	W(c.showCursor, c.altScreenOff);
	console.error(`Fatal: ${err}`);
	process.exit(1);
});
