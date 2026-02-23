#!/usr/bin/env node

/**
 * rlm — Recursive Language Model CLI
 *
 * This shim boots the CLI entry point. It tries the compiled dist first,
 * then falls back to tsx for development.
 */

import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { existsSync } from "node:fs";

const __dirname = dirname(fileURLToPath(import.meta.url));
const distEntry = join(__dirname, "..", "dist", "main.js");

if (existsSync(distEntry)) {
	// Production: use compiled JS
	await import(distEntry);
} else {
	// Development: use tsx to run TypeScript directly
	const srcEntry = join(__dirname, "..", "src", "main.ts");
	const { register } = await import("node:module");

	// Try to register tsx loader, then import
	try {
		const tsxPath = join(__dirname, "..", "node_modules", "tsx", "dist", "esm", "index.mjs");
		if (existsSync(tsxPath)) {
			register(tsxPath);
		}
		await import(srcEntry);
	} catch {
		// Fallback: spawn tsx as a child process
		const { spawn } = await import("node:child_process");
		const tsxBin = join(__dirname, "..", "node_modules", ".bin", "tsx");
		const child = spawn(tsxBin, [srcEntry, ...process.argv.slice(2)], {
			stdio: "inherit",
		});
		child.on("exit", (code) => process.exit(code ?? 1));
	}
}
