/**
 * Configuration loader for RLM CLI.
 *
 * Reads rlm_config.yaml from the project root (or cwd), with sensible defaults.
 */

import * as fs from "node:fs";
import * as path from "node:path";

export interface RlmConfig {
	max_iterations: number;
	max_depth: number;
	max_sub_queries: number;
	truncate_len: number;
	metadata_preview_lines: number;
}

const DEFAULTS: RlmConfig = {
	max_iterations: 20,
	max_depth: 3,
	max_sub_queries: 50,
	truncate_len: 5000,
	metadata_preview_lines: 20,
};

function parseYaml(text: string): Record<string, unknown> {
	// Minimal YAML parser for flat key:value files (no nested objects, no arrays)
	const result: Record<string, unknown> = {};
	for (const line of text.split("\n")) {
		const trimmed = line.trim();
		if (!trimmed || trimmed.startsWith("#")) continue;
		const colonIdx = trimmed.indexOf(":");
		if (colonIdx === -1) continue;
		const key = trimmed.slice(0, colonIdx).trim();
		const rawVal = trimmed.slice(colonIdx + 1).trim();
		// Strip inline comments
		const val = rawVal.replace(/\s+#.*$/, "");
		// Parse number
		const num = Number(val);
		if (!isNaN(num) && val !== "") {
			result[key] = num;
		} else if (val === "true") {
			result[key] = true;
		} else if (val === "false") {
			result[key] = false;
		} else {
			// Strip quotes
			result[key] = val.replace(/^["']|["']$/g, "");
		}
	}
	return result;
}

export function loadConfig(): RlmConfig {
	// Search order: cwd, then package root
	const candidates = [
		path.resolve(process.cwd(), "rlm_config.yaml"),
		path.resolve(new URL(".", import.meta.url).pathname, "..", "rlm_config.yaml"),
	];

	for (const configPath of candidates) {
		if (fs.existsSync(configPath)) {
			try {
				const raw = fs.readFileSync(configPath, "utf-8");
				const parsed = parseYaml(raw);
				return {
					max_iterations: typeof parsed.max_iterations === "number" ? parsed.max_iterations : DEFAULTS.max_iterations,
					max_depth: typeof parsed.max_depth === "number" ? parsed.max_depth : DEFAULTS.max_depth,
					max_sub_queries: typeof parsed.max_sub_queries === "number" ? parsed.max_sub_queries : DEFAULTS.max_sub_queries,
					truncate_len: typeof parsed.truncate_len === "number" ? parsed.truncate_len : DEFAULTS.truncate_len,
					metadata_preview_lines: typeof parsed.metadata_preview_lines === "number" ? parsed.metadata_preview_lines : DEFAULTS.metadata_preview_lines,
				};
			} catch {
				// Fall through to defaults
			}
		}
	}

	return { ...DEFAULTS };
}
