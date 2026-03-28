/**
 * rlm-cli — Shared terminal color & print utilities
 *
 * Electric Amber palette with true RGB (24-bit truecolor) rendering,
 * falling back to xterm-256 on terminals that don't support truecolor.
 *
 * Inspired by Feynman CLI's terminal.ts color system.
 */

const hasTrueColor =
	process.env.COLORTERM === "truecolor" ||
	process.env.COLORTERM === "24bit" ||
	process.env.TERM_PROGRAM === "iTerm.app" ||
	process.env.TERM_PROGRAM === "vscode" ||
	process.env.TERM_PROGRAM === "WezTerm";

const has256Color =
	hasTrueColor ||
	(process.env.TERM || "").includes("256color") ||
	process.stdout.hasColors?.(256) === true;

function rgb(r: number, g: number, b: number, fallback256: number): string {
	if (hasTrueColor) return `\x1b[38;2;${r};${g};${b}m`;
	if (has256Color) return `\x1b[38;5;${fallback256}m`;
	return "";
}

// ── Reset / style codes ──────────────────────────────────────────────────────
export const RESET     = "\x1b[0m";
export const BOLD      = "\x1b[1m";
export const DIM       = "\x1b[2m";
export const ITALIC    = "\x1b[3m";
export const UNDERLINE = "\x1b[4m";

// ── Electric Amber palette ───────────────────────────────────────────────────
export const AMBER      = rgb(214, 163,   0, 214);  // electric amber — primary
export const AMBER_DIM  = rgb(172, 126,   0, 172);  // deep amber — secondary
export const SAGE       = rgb(130, 185, 128, 114);  // soft green — success / result
export const ICE        = rgb(117, 187, 220, 117);  // ice blue — code / info
export const LAVENDER   = rgb(183, 150, 220, 183);  // soft lavender — sub-queries
export const STONE      = rgb(157, 169, 160, 246);  // neutral stone — secondary text
export const ASH        = rgb(133, 146, 137, 244);  // dimmer neutral — info text
export const DARK_ASH   = rgb( 92, 106, 114, 242);  // dark neutral — borders / chrome
export const ROSE       = rgb(230, 126, 128, 203);  // rose — errors / warnings

// ── Utility ──────────────────────────────────────────────────────────────────

export function paint(text: string, ...codes: string[]): string {
	return `${codes.join("")}${text}${RESET}`;
}

// ── Print helpers ─────────────────────────────────────────────────────────────

export function printInfo(text: string): void {
	console.log(paint(`  ${text}`, ASH));
}

export function printSuccess(text: string): void {
	console.log(paint(`  ✓ ${text}`, SAGE, BOLD));
}

export function printWarning(text: string): void {
	console.log(paint(`  ⚠ ${text}`, STONE, BOLD));
}

export function printError(text: string): void {
	console.log(paint(`  ✗ ${text}`, ROSE, BOLD));
}

export function printSection(title: string): void {
	console.log("");
	console.log(paint(`  ◆ ${title}`, ICE, BOLD));
}

/**
 * Render a bordered panel.
 *
 * @param title        Bold header line (rendered in AMBER)
 * @param subtitleLines Optional sub-lines below the divider (rendered in STONE)
 * @param width        Inner content width (default 53)
 */
export function printPanel(
	title: string,
	subtitleLines: string[] = [],
	width = 53,
): void {
	const border = "─".repeat(width + 2);

	const renderLine = (text: string, color: string, bold = false): string => {
		const content = text.length > width ? `${text.slice(0, width - 3)}…` : text;
		const codes = bold ? `${color}${BOLD}` : color;
		return `${DARK_ASH}${BOLD}│${RESET} ${codes}${content.padEnd(width)}${RESET} ${DARK_ASH}${BOLD}│${RESET}`;
	};

	console.log("");
	console.log(paint(`  ┌${border}┐`, DARK_ASH, BOLD));
	console.log(`  ${renderLine(title, AMBER, true)}`);
	if (subtitleLines.length > 0) {
		console.log(paint(`  ├${border}┤`, DARK_ASH, BOLD));
		for (const line of subtitleLines) {
			console.log(`  ${renderLine(line, STONE)}`);
		}
	}
	console.log(paint(`  └${border}┘`, DARK_ASH, BOLD));
	console.log("");
}
