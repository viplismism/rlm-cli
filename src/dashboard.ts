#!/usr/bin/env tsx
/**
 * RLM Dashboard — local web dashboard for running and viewing RLM trajectories.
 *
 * Serves a web UI on http://localhost:3000 where you can:
 *   - Submit queries with context (file upload / URL / paste)
 *   - Watch iterations unfold in real-time via SSE
 *   - Browse and inspect saved trajectory JSON files
 *
 * Usage:
 *   npx tsx src/dashboard.ts
 *   RLM_PORT=4000 npx tsx src/dashboard.ts
 */

import "./env.js";
import * as fs from "node:fs";
import * as path from "node:path";
import * as http from "node:http";
import { fileURLToPath } from "node:url";

const { getModels, getProviders } = await import("@mariozechner/pi-ai");
const { PythonRepl } = await import("./repl.js");
const { runRlmLoop } = await import("./rlm.js");

import type { Api, Model } from "@mariozechner/pi-ai";
import type { RlmProgress, RlmResult, SubQueryInfo } from "./rlm.js";

// ── Constants ───────────────────────────────────────────────────────────────

const PORT = parseInt(process.env.RLM_PORT || "3000", 10);
const DEFAULT_MODEL = process.env.RLM_MODEL || "claude-sonnet-4-5-20250929";
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const TRAJ_DIR = path.resolve(process.cwd(), "trajectories");
const HTML_PATH = path.join(__dirname, "dashboard.html");

// ── SSE client management ───────────────────────────────────────────────────

const sseClients = new Set<http.ServerResponse>();

function broadcast(event: string, data: unknown): void {
	const payload = `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
	for (const res of sseClients) {
		res.write(payload);
	}
}

// ── Active run state ────────────────────────────────────────────────────────

let activeRun: { abort: AbortController } | null = null;

// ── Helpers ─────────────────────────────────────────────────────────────────

async function readBody(req: http.IncomingMessage): Promise<string> {
	const chunks: Buffer[] = [];
	for await (const chunk of req) chunks.push(chunk as Buffer);
	return Buffer.concat(chunks).toString("utf-8");
}

function getAvailableModels(): { id: string; provider: string }[] {
	const models: { id: string; provider: string }[] = [];
	for (const provider of getProviders()) {
		for (const m of getModels(provider)) {
			models.push({ id: m.id, provider: String(provider) });
		}
	}
	return models;
}

function listTrajectoryFiles(): { name: string; size: number; mtime: string }[] {
	if (!fs.existsSync(TRAJ_DIR)) return [];
	return fs
		.readdirSync(TRAJ_DIR)
		.filter((f) => f.endsWith(".json"))
		.map((f) => {
			const stat = fs.statSync(path.join(TRAJ_DIR, f));
			return { name: f, size: stat.size, mtime: stat.mtime.toISOString() };
		})
		.sort((a, b) => b.mtime.localeCompare(a.mtime));
}

// ── Run RLM ─────────────────────────────────────────────────────────────────

async function startRun(modelId: string, query: string, context: string): Promise<void> {
	// Resolve model
	let model: Model<Api> | undefined;
	for (const provider of getProviders()) {
		for (const m of getModels(provider)) {
			if (m.id === modelId) model = m;
		}
	}
	if (!model) {
		broadcast("error", { message: `Unknown model: ${modelId}` });
		broadcast("done", {});
		return;
	}

	const ac = new AbortController();
	activeRun = { abort: ac };

	// Trajectory bookkeeping
	const trajectory: any = {
		model: modelId,
		query,
		contextLength: context.length,
		contextLines: context.split("\n").length,
		startTime: new Date().toISOString(),
		iterations: [],
		result: null,
		totalElapsedMs: 0,
	};
	const startTime = Date.now();
	let currentStep: any = null;
	let iterStart = Date.now();

	const repl = new PythonRepl();

	try {
		await repl.start(ac.signal);

		broadcast("started", {
			model: modelId,
			query,
			contextLength: context.length,
			contextLines: context.split("\n").length,
		});

		const result = await runRlmLoop({
			context,
			query,
			model,
			repl,
			signal: ac.signal,
			onProgress: (info: RlmProgress) => {
				broadcast("progress", info);

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
				}
				if (info.phase === "executing" && info.code) {
					if (currentStep) {
						currentStep.code = info.code;
						currentStep.rawResponse = info.rawResponse || "";
					}
				}
				if (info.phase === "checking_final") {
					if (currentStep) {
						currentStep.stdout = info.stdout || "";
						currentStep.stderr = info.stderr || "";
						currentStep.elapsedMs = Date.now() - iterStart;
						trajectory.iterations.push({ ...currentStep });
					}
				}
			},
			onSubQuery: (info: SubQueryInfo) => {
				broadcast("subquery", info);
				if (currentStep) {
					currentStep.subQueries.push(info);
				}
			},
		});

		trajectory.result = result;
		trajectory.totalElapsedMs = Date.now() - startTime;

		if (result.completed && trajectory.iterations.length > 0) {
			trajectory.iterations[trajectory.iterations.length - 1].hasFinal = true;
		}

		// Save trajectory
		if (!fs.existsSync(TRAJ_DIR)) fs.mkdirSync(TRAJ_DIR, { recursive: true });
		const ts = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
		const trajFile = `trajectory-${ts}.json`;
		fs.writeFileSync(path.join(TRAJ_DIR, trajFile), JSON.stringify(trajectory, null, 2), "utf-8");

		broadcast("result", { ...result, trajectoryFile: trajFile });
	} catch (err: any) {
		broadcast("error", { message: err.message || String(err) });
	} finally {
		repl.shutdown();
		activeRun = null;
		broadcast("done", {});
	}
}

// ── HTTP Server ─────────────────────────────────────────────────────────────

function jsonResponse(res: http.ServerResponse, status: number, body: unknown): void {
	res.writeHead(status, { "Content-Type": "application/json" });
	res.end(JSON.stringify(body));
}

const server = http.createServer(async (req, res) => {
	const url = new URL(req.url || "/", `http://localhost:${PORT}`);
	const pathname = url.pathname;

	// ── Serve dashboard HTML ──
	if (pathname === "/" && req.method === "GET") {
		if (!fs.existsSync(HTML_PATH)) {
			res.writeHead(500, { "Content-Type": "text/plain" });
			res.end(`Dashboard HTML not found at: ${HTML_PATH}`);
			return;
		}
		const html = fs.readFileSync(HTML_PATH, "utf-8");
		res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
		res.end(html);
		return;
	}

	// ── SSE endpoint ──
	if (pathname === "/api/events" && req.method === "GET") {
		res.writeHead(200, {
			"Content-Type": "text/event-stream",
			"Cache-Control": "no-cache",
			Connection: "keep-alive",
		});
		res.write(": connected\n\n");
		sseClients.add(res);
		req.on("close", () => sseClients.delete(res));
		return;
	}

	// ── API: list models ──
	if (pathname === "/api/models" && req.method === "GET") {
		return jsonResponse(res, 200, { defaultModel: DEFAULT_MODEL });
	}

	// ── API: list trajectories ──
	if (pathname === "/api/trajectories" && req.method === "GET") {
		return jsonResponse(res, 200, listTrajectoryFiles());
	}

	// ── API: get single trajectory ──
	if (pathname.startsWith("/api/trajectory/") && req.method === "GET") {
		const filename = decodeURIComponent(pathname.slice("/api/trajectory/".length));
		if (filename.includes("..") || filename.includes("/")) {
			return jsonResponse(res, 400, { error: "Invalid filename" });
		}
		const filePath = path.join(TRAJ_DIR, filename);
		if (!fs.existsSync(filePath)) {
			return jsonResponse(res, 404, { error: "Not found" });
		}
		const data = fs.readFileSync(filePath, "utf-8");
		res.writeHead(200, { "Content-Type": "application/json" });
		res.end(data);
		return;
	}

	// ── API: delete trajectory ──
	if (pathname.startsWith("/api/trajectory/") && req.method === "DELETE") {
		const filename = decodeURIComponent(pathname.slice("/api/trajectory/".length));
		if (filename.includes("..") || filename.includes("/")) {
			return jsonResponse(res, 400, { error: "Invalid filename" });
		}
		const filePath = path.join(TRAJ_DIR, filename);
		if (!fs.existsSync(filePath)) {
			return jsonResponse(res, 404, { error: "Not found" });
		}
		fs.unlinkSync(filePath);
		return jsonResponse(res, 200, { ok: true });
	}

	// ── API: start run ──
	if (pathname === "/api/run" && req.method === "POST") {
		if (activeRun) {
			return jsonResponse(res, 409, { error: "A run is already in progress" });
		}

		let body: any;
		try {
			body = JSON.parse(await readBody(req));
		} catch {
			return jsonResponse(res, 400, { error: "Invalid JSON body" });
		}

		const modelId = DEFAULT_MODEL;
		const { query, context, contextUrl } = body;
		if (!query) {
			return jsonResponse(res, 400, { error: "query is required" });
		}

		let ctx: string = context || "";
		if (contextUrl) {
			try {
				const resp = await fetch(contextUrl);
				if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`);
				ctx = await resp.text();
			} catch (err: any) {
				return jsonResponse(res, 400, { error: `Failed to fetch URL: ${err.message}` });
			}
		}

		if (!ctx.trim()) {
			return jsonResponse(res, 400, { error: "No context provided" });
		}

		jsonResponse(res, 200, { ok: true });
		startRun(modelId, query, ctx).catch(console.error);
		return;
	}

	// ── API: abort run ──
	if (pathname === "/api/abort" && req.method === "POST") {
		if (activeRun) {
			activeRun.abort.abort();
			return jsonResponse(res, 200, { ok: true });
		}
		return jsonResponse(res, 404, { error: "No active run" });
	}

	// ── 404 ──
	jsonResponse(res, 404, { error: "Not found" });
});

server.listen(PORT, () => {
	console.log(`\n  🚀 RLM Dashboard running at http://localhost:${PORT}\n`);
});

process.on("SIGINT", () => {
	console.log("\nShutting down...");
	if (activeRun) activeRun.abort.abort();
	server.close();
	process.exit(0);
});
