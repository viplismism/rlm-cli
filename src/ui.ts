import {
	RESET,
	BOLD,
	DIM,
	AMBER,
	ICE,
	STONE,
	ASH,
	DARK_ASH,
	paint,
} from "./colors.js";

export function stripAnsi(text: string): string {
	return text.replace(/\x1b\[[0-9;]*m/g, "");
}

export function visibleWidth(text: string): number {
	return stripAnsi(text).length;
}

export function padAnsiRight(text: string, width: number): string {
	const pad = Math.max(0, width - visibleWidth(text));
	return text + " ".repeat(pad);
}

export function truncateAnsi(text: string, width: number): string {
	const plain = stripAnsi(text);
	if (plain.length <= width) return text;
	if (width <= 1) return plain.slice(0, width);
	return plain.slice(0, width - 1) + "…";
}

export function renderCard(title: string, lines: string[], width: number): string[] {
	const inner = Math.max(12, width - 2);
	const safeTitle = truncateAnsi(title, Math.max(4, inner - 4));
	const titleWidth = visibleWidth(safeTitle) + 2;
	const rightFill = Math.max(0, inner - titleWidth - 1);
	const out: string[] = [
		`${DARK_ASH}╭─${RESET}${AMBER}${BOLD} ${safeTitle} ${RESET}${DARK_ASH}${"─".repeat(rightFill)}╮${RESET}`,
	];

	for (const line of lines) {
		const fitted = truncateAnsi(line, inner - 1);
		out.push(`${DARK_ASH}│${RESET} ${padAnsiRight(fitted, inner - 1)}${DARK_ASH}│${RESET}`);
	}

	out.push(`${DARK_ASH}╰${"─".repeat(inner)}╯${RESET}`);
	return out;
}

export function joinColumns(left: string[], right: string[], gap = 3): string[] {
	const leftWidth = left.reduce((max, line) => Math.max(max, visibleWidth(line)), 0);
	const rows = Math.max(left.length, right.length);
	const out: string[] = [];

	for (let i = 0; i < rows; i++) {
		const l = left[i] ?? "";
		const r = right[i] ?? "";
		out.push(`${padAnsiRight(l, leftWidth)}${" ".repeat(gap)}${r}`);
	}

	return out;
}

export function buildShellHeader(opts: {
	app: string;
	mode: string;
	cwd?: string;
	version?: string;
	width?: number;
}): string[] {
	const width = Math.max(60, opts.width ?? Math.min(process.stdout.columns || 80, 110));
	const app = paint(opts.app, AMBER, BOLD);
	const mode = paint(opts.mode, ICE);
	const cwd = opts.cwd ? paint(opts.cwd, DIM) : "";
	const version = opts.version ? paint(opts.version, ASH) : "";
	const line1 = `${app} ${paint("·", DARK_ASH)} ${mode}`;
	const line2Parts = [cwd, version].filter(Boolean);
	const line2 = line2Parts.join(` ${paint("·", DARK_ASH)} `);

	return [
		paint("─".repeat(width), DARK_ASH, BOLD),
		padAnsiRight(line1, width),
		line2 ? padAnsiRight(line2, width) : "",
		paint("─".repeat(width), DARK_ASH, BOLD),
	].filter(Boolean);
}

export function buildStatusBar(items: string[], width?: number): string {
	const maxWidth = width ?? Math.min(process.stdout.columns || 80, 110);
	const separator = ` ${paint("·", DARK_ASH)} `;
	const filtered = items.filter(Boolean);
	let out = "";

	for (const item of filtered) {
		const next = out ? `${out}${separator}${item}` : item;
		if (visibleWidth(next) > maxWidth) break;
		out = next;
	}

	if (!out && filtered[0]) out = filtered[0];
	return padAnsiRight(out, maxWidth);
}

export function buildMutedList(items: string[]): string[] {
	return items.map((item) => `${paint("•", STONE)} ${paint(item, DIM)}`);
}
