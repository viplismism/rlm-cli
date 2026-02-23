/**
 * Load .env file into process.env.
 * Must be imported BEFORE any module that reads env vars (e.g. pi-ai).
 *
 * Supports:
 * - ANTHROPIC_API_KEY
 * - RLM_MODEL (model name, e.g. claude-sonnet-4-5-20250929)
 */

import * as fs from "node:fs";
import * as path from "node:path";

// Load .env file
const envPath = path.resolve(process.cwd(), ".env");
if (fs.existsSync(envPath)) {
	const content = fs.readFileSync(envPath, "utf-8");
	for (const line of content.split("\n")) {
		const trimmed = line.trim();
		if (!trimmed || trimmed.startsWith("#")) continue;
		const eqIndex = trimmed.indexOf("=");
		if (eqIndex === -1) continue;
		const key = trimmed.slice(0, eqIndex).trim();
		const value = trimmed.slice(eqIndex + 1).trim();
		if (key) {
			process.env[key] = value;
		}
	}
}

// Default model
if (!process.env.RLM_MODEL) {
	process.env.RLM_MODEL = "claude-sonnet-4-5-20250929";
}
