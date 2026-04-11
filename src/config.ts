/**
 * Configuration loader for RLM CLI.
 *
 * Reads rlm_config.yaml from the project root (or cwd), with sensible defaults.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";

export interface RlmConfig {
	max_iterations: number;
	max_depth: 1;
	max_sub_queries: number;
	truncate_len: number;
	metadata_preview_lines: number;
	sub_model: string;  // model ID for sub-queries (empty = same as root)
}

const DEFAULTS: RlmConfig = {
	max_iterations: 20,
	max_depth: 1,  // current runtime implements paper-style flat sub-calls, not nested RLM recursion
	max_sub_queries: 50,
	truncate_len: 5000,
	metadata_preview_lines: 20,
	sub_model: "",  // empty = same model as root
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
		path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "rlm_config.yaml"),
	];

	for (const configPath of candidates) {
		if (fs.existsSync(configPath)) {
			try {
				const raw = fs.readFileSync(configPath, "utf-8");
				const parsed = parseYaml(raw);
				const clamp = (v: unknown, min: number, max: number, def: number) =>
					typeof v === "number" && isFinite(v) ? Math.max(min, Math.min(max, Math.round(v))) : def;
				return {
					max_iterations: clamp(parsed.max_iterations, 1, 100, DEFAULTS.max_iterations),
					max_depth: 1,
					max_sub_queries: clamp(parsed.max_sub_queries, 1, 500, DEFAULTS.max_sub_queries),
					truncate_len: clamp(parsed.truncate_len, 500, 50000, DEFAULTS.truncate_len),
					metadata_preview_lines: clamp(parsed.metadata_preview_lines, 5, 100, DEFAULTS.metadata_preview_lines),
					sub_model: typeof parsed.sub_model === "string" ? parsed.sub_model.trim() : (process.env.RLM_SUB_MODEL ?? ""),
				};
			} catch {
				// Fall through to defaults
			}
		}
	}

	return { ...DEFAULTS, sub_model: process.env.RLM_SUB_MODEL ?? "" };
}
