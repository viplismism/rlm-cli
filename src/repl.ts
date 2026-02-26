/**
 * Persistent Python REPL manager for the RLM CLI.
 *
 * Spawns a single Python subprocess running `runtime.py` and keeps it alive
 * across multiple RLM iterations. Communication uses line-delimited JSON
 * over stdin/stdout.
 */

import { type ChildProcess, spawn } from "node:child_process";
import * as path from "node:path";
import * as readline from "node:readline";

// ── Types ───────────────────────────────────────────────────────────────────

/** Result of executing a code snippet in the REPL. */
export interface ExecResult {
	stdout: string;
	stderr: string;
	hasFinal: boolean;
	finalValue: string | null;
}

/** Callback the host provides to handle llm_query() calls from Python. */
export type LlmQueryHandler = (subContext: string, instruction: string) => Promise<string>;

// ── Inbound message types from Python ───────────────────────────────────────

interface ReadyMessage {
	type: "ready";
}

interface ExecDoneMessage {
	type: "exec_done";
	stdout: string;
	stderr: string;
	has_final: boolean;
	final_value: string | null;
}

interface LlmQueryMessage {
	type: "llm_query";
	sub_context: string;
	instruction: string;
	id: string;
}

interface ContextSetMessage {
	type: "context_set";
}

interface FinalResetMessage {
	type: "final_reset";
}

type InboundMessage = ReadyMessage | ExecDoneMessage | LlmQueryMessage | ContextSetMessage | FinalResetMessage;

// ── REPL class ──────────────────────────────────────────────────────────────

export class PythonRepl {
	private proc: ChildProcess | null = null;
	private rl: readline.Interface | null = null;
	private llmQueryHandler: LlmQueryHandler | null = null;

	/**
	 * Pending resolvers for messages we're waiting on from Python.
	 * Each entry maps a message type to a one-shot resolve/reject pair.
	 */
	private pending: Map<string, { resolve: (msg: InboundMessage) => void; reject: (err: Error) => void }> = new Map();

	/** Whether the REPL subprocess is alive. */
	get isAlive(): boolean {
		return this.proc !== null && this.proc.exitCode === null;
	}

	/**
	 * Start the Python subprocess and wait for it to signal readiness.
	 */
	async start(signal?: AbortSignal): Promise<void> {
		if (this.isAlive) return;

		const runtimePath = path.join(path.dirname(new URL(import.meta.url).pathname), "runtime.py");

		this.proc = spawn("python3", [runtimePath], {
			stdio: ["pipe", "pipe", "pipe"],
			env: {
				// Only pass what Python actually needs — not API keys or secrets
				PATH: process.env.PATH,
				HOME: process.env.HOME,
				PYTHONUNBUFFERED: "1",
			},
		});

		this.rl = readline.createInterface({ input: this.proc.stdout! });
		this.rl.on("line", (line: string) => this.handleLine(line));

		this.proc.stderr!.on("data", (chunk: Buffer) => {
			const text = chunk.toString();
			if (text.trim()) {
				process.stderr.write(`[rlm-repl-python] ${text}`);
			}
		});

		this.proc.on("close", () => {
			this.cleanup();
		});

		if (signal) {
			signal.addEventListener(
				"abort",
				() => {
					this.shutdown();
				},
				{ once: true },
			);
		}

		await this.waitForMessage("ready");
	}

	/** Register the callback that handles llm_query() calls from Python. */
	setLlmQueryHandler(handler: LlmQueryHandler): void {
		this.llmQueryHandler = handler;
	}

	/** Inject the full context string into the Python REPL. */
	async setContext(text: string): Promise<void> {
		this.send({ type: "set_context", value: text });
		await this.waitForMessage("context_set");
	}

	/** Reset the Final sentinel variable. */
	async resetFinal(): Promise<void> {
		this.send({ type: "reset_final" });
		await this.waitForMessage("final_reset");
	}

	/** Execute a code snippet and return the result. */
	async execute(code: string): Promise<ExecResult> {
		this.send({ type: "exec", code });
		const msg = (await this.waitForMessage("exec_done")) as ExecDoneMessage;
		return {
			stdout: msg.stdout,
			stderr: msg.stderr,
			hasFinal: msg.has_final,
			finalValue: msg.final_value,
		};
	}

	/** Gracefully shut down the Python subprocess. */
	shutdown(): void {
		if (this.proc && this.proc.exitCode === null) {
			try {
				this.send({ type: "shutdown" });
			} catch {
				// stdin may already be closed
			}
			this.proc.kill("SIGTERM");
		}
		this.cleanup();
	}

	// ── Internal ─────────────────────────────────────────────────────────────

	private send(msg: Record<string, unknown>): void {
		if (!this.proc || !this.proc.stdin || this.proc.stdin.destroyed) {
			throw new Error("REPL subprocess is not running");
		}
		this.proc.stdin.write(`${JSON.stringify(msg)}\n`);
	}

	private handleLine(line: string): void {
		const trimmed = line.trim();
		if (!trimmed) return;

		let msg: InboundMessage;
		try {
			msg = JSON.parse(trimmed) as InboundMessage;
		} catch {
			return;
		}

		if (msg.type === "llm_query") {
			this.handleLlmQueryMessage(msg as LlmQueryMessage);
			return;
		}

		const entry = this.pending.get(msg.type);
		if (entry) {
			this.pending.delete(msg.type);
			entry.resolve(msg);
		}
	}

	private async handleLlmQueryMessage(msg: LlmQueryMessage): Promise<void> {
		if (!this.llmQueryHandler) {
			this.send({
				type: "llm_result",
				id: msg.id,
				result: "[ERROR] No LLM query handler registered",
			});
			return;
		}

		try {
			const result = await this.llmQueryHandler(msg.sub_context, msg.instruction);
			this.send({ type: "llm_result", id: msg.id, result });
		} catch (err) {
			const errorText = err instanceof Error ? err.message : String(err);
			this.send({
				type: "llm_result",
				id: msg.id,
				result: `[ERROR] LLM query failed: ${errorText}`,
			});
		}
	}

	private waitForMessage(type: string): Promise<InboundMessage> {
		return new Promise((resolve, reject) => {
			if (!this.isAlive) {
				reject(new Error(`REPL subprocess is not running (waiting for "${type}")`));
				return;
			}

			const timeout = setTimeout(() => {
				if (this.pending.has(type)) {
					this.pending.delete(type);
					reject(new Error(`Timeout waiting for "${type}" from Python REPL`));
				}
			}, 300_000);

			this.pending.set(type, {
				resolve: (msg) => {
					clearTimeout(timeout);
					resolve(msg);
				},
				reject: (err) => {
					clearTimeout(timeout);
					reject(err);
				},
			});
		});
	}

	private cleanup(): void {
		this.rl?.close();
		this.rl = null;
		this.proc = null;
		// Reject all pending promises so callers unblock immediately
		const abortError = new Error("REPL shut down");
		for (const [, entry] of this.pending) {
			entry.reject(abortError);
		}
		this.pending.clear();
	}
}
