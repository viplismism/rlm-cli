/**
 * Load env vars into process.env.
 * Must be imported BEFORE any module that reads env vars (e.g. pi-ai).
 *
 * Priority (highest wins):
 *   1. Shell environment variables (already in process.env)
 *   2. .env in the current working directory, else .env in package root
 *   3. ~/.rlm/credentials — persistent keys saved by first-run setup
 *
 * Supported provider keys include ANTHROPIC_API_KEY, OPENAI_API_KEY,
 * GEMINI_API_KEY, and OPENROUTER_API_KEY.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import * as os from "node:os";
import { fileURLToPath } from "node:url";

function parseEnvFile(filePath: string): Map<string, string> {
	const vars = new Map<string, string>();
	if (!fs.existsSync(filePath)) return vars;
	const content = fs.readFileSync(filePath, "utf-8");
	for (const rawLine of content.split("\n")) {
		let trimmed = rawLine.trim();
		if (!trimmed || trimmed.startsWith("#")) continue;
		// Strip leading "export " (common in .env files)
		if (trimmed.startsWith("export ")) trimmed = trimmed.slice(7);
		const eqIndex = trimmed.indexOf("=");
		if (eqIndex === -1) continue;
		const key = trimmed.slice(0, eqIndex).trim();
		let value = trimmed.slice(eqIndex + 1).trim();
		// Strip matching surrounding quotes ("..." or '...')
		if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
			value = value.slice(1, -1);
		}
		if (key) vars.set(key, value);
	}
	return vars;
}

// Collect file-based vars: credentials first, .env overwrites
const fileVars = new Map<string, string>();

// 1. Load persistent credentials (lowest priority file)
const credVars = parseEnvFile(path.join(os.homedir(), ".rlm", "credentials"));
for (const [k, v] of credVars) fileVars.set(k, v);

// 2. Load .env (overwrites credentials): prefer the current working
// directory, falling back to the package root (matters for global installs)
const __dir = path.dirname(fileURLToPath(import.meta.url));
const cwdDotenv = path.resolve(process.cwd(), ".env");
const dotenvPath = fs.existsSync(cwdDotenv) ? cwdDotenv : path.resolve(__dir, "..", ".env");
const dotenvVars = parseEnvFile(dotenvPath);
for (const [k, v] of dotenvVars) fileVars.set(k, v);

// 3. Apply: only set if NOT already in shell env (shell always wins)
for (const [key, value] of fileVars) {
	if (!process.env[key]) {
		process.env[key] = value;
	}
}

// Alias: GOOGLE_API_KEY → GEMINI_API_KEY (pi-ai expects GEMINI_API_KEY)
// Remove GOOGLE_API_KEY after aliasing to avoid pi-ai "both set" warning
if (process.env.GOOGLE_API_KEY) {
	if (!process.env.GEMINI_API_KEY) {
		process.env.GEMINI_API_KEY = process.env.GOOGLE_API_KEY;
	}
	delete process.env.GOOGLE_API_KEY;
}

// Default model
if (!process.env.RLM_MODEL) {
	process.env.RLM_MODEL = "claude-sonnet-4-6";
}
