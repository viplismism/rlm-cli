/**
 * RLM Loop — implements Algorithm 1 from "Recursive Language Models" (arXiv:2512.24601).
 *
 * The loop works as follows:
 *   1. Inject the full context into a persistent Python REPL as a variable.
 *   2. Send the LLM metadata about the context plus the user's query.
 *      The LLM writes Python code that can inspect/slice/query `context`,
 *      call `llm_query()` recursively, and call FINAL() when done.
 *   3. Execute the code, capture stdout.
 *   4. If FINAL is set, return it. Otherwise loop.
 */

import {
	type Api,
	type AssistantMessage,
	completeSimple,
	type Message,
	type Model,
	type TextContent,
	type UserMessage,
} from "@mariozechner/pi-ai";
import type { ExecResult, PythonRepl } from "./repl.js";
import { loadConfig, type RlmConfig } from "./config.js";

// ── Load config ─────────────────────────────────────────────────────────────

const config = loadConfig();

// ── Types ───────────────────────────────────────────────────────────────────

export interface RlmOptions {
	context: string;
	query: string;
	model: Model<Api>;
	/** Optional cheaper model used for sub-queries. Falls back to `model` if not set. */
	subModel?: Model<Api>;
	repl: PythonRepl;
	signal?: AbortSignal;
	onProgress?: (info: RlmProgress) => void;
	onSubQueryStart?: (info: SubQueryStartInfo) => void;
	onSubQuery?: (info: SubQueryInfo) => void;
}

export interface RlmProgress {
	iteration: number;
	maxIterations: number;
	subQueries: number;
	phase: "generating_code" | "executing" | "checking_final";
	code?: string;
	stdout?: string;
	stderr?: string;
	userMessage?: string;
	rawResponse?: string;
	systemPrompt?: string;
}

export interface SubQueryStartInfo {
	index: number;
	contextLength: number;
	instruction: string;
}

export interface SubQueryInfo {
	index: number;
	contextLength: number;
	instruction: string;
	resultLength: number;
	resultPreview: string;
	elapsedMs: number;
}

export interface RlmResult {
	answer: string;
	iterations: number;
	totalSubQueries: number;
	completed: boolean;
	/** Approximate token counts (root model only; sub-queries not tracked by all providers) */
	inputTokens?: number;
	outputTokens?: number;
}

// ── System prompt (aligned with paper Appendix C) ───────────────────────────

function buildSystemPrompt(opts?: {
	iteration?: number;
	maxIterations?: number;
	subQueriesUsed?: number;
	maxSubQueries?: number;
	hasSubModel?: boolean;
}): string {
	const {
		iteration = 1,
		maxIterations = config.max_iterations,
		subQueriesUsed = 0,
		maxSubQueries = config.max_sub_queries,
		hasSubModel = false,
	} = opts ?? {};

	const budgetLine = `You have ${maxIterations - iteration + 1} iteration(s) remaining and ${maxSubQueries - subQueriesUsed} sub-query call(s) remaining out of ${maxSubQueries} total.`;
	const subModelNote = hasSubModel
		? "Sub-queries use a smaller, faster model — they are cheap. Use them liberally for chunking and aggregation."
		: "Sub-queries use the same model as the root — be strategic and avoid excessive calls.";

	return `You are a Recursive Language Model (RLM) agent. You process arbitrarily large contexts by writing Python code in a persistent REPL.

## Budget
${budgetLine}
${subModelNote}

## Available in the REPL

1. \`context\` — the full input text loaded as a string variable (may be very large; do NOT copy it into your answer).

2. \`llm_query(sub_context, instruction)\` — sends a sub-string of context plus an instruction to an LLM and returns its text response. Use for summarization, extraction, classification, and aggregation on chunks.
   For parallel queries: \`async_llm_query(sub_context, instruction)\` with \`asyncio.gather()\`.

3. \`FINAL("answer")\` — sets the final answer and terminates the loop.
   \`FINAL_VAR(variable)\` — returns a variable you built up in the REPL as the final answer.

## Core Rules

1. Write valid Python 3. The standard library is available.
2. Use \`print()\` for intermediate output — it is fed back as context in the next iteration (truncated to last ${config.truncate_len} chars).
3. Use \`len(context)\` and slicing. Never load the full context into your reply.
4. For large contexts: split into chunks → \`llm_query()\` each → aggregate → FINAL.
5. Use \`FINAL()\` only when you have a complete, high-quality answer. Print it first to verify.
6. Do NOT rewrite or delete existing REPL variables — the REPL is persistent like a Jupyter notebook.
7. Prefer \`asyncio.gather()\` for independent parallel sub-queries to reduce wall-clock time.

## Sub-query best practices (key to RLM performance)

- **Always include the query/task inside the instruction**, not just raw context. The sub-agent only sees what you give it.
- Pass precise slices: \`context[start:end]\` rather than the entire context.
- Use regex or string operations first to filter before calling \`llm_query()\` — this reduces cost significantly.
- For aggregation: collect results in a list, then call \`llm_query("\n".join(results), "Synthesize...")\`.
- For classification/extraction over many items: batch multiple items per sub-query call.
- Store sub-query results in REPL variables for use across iterations.

## Output format

Respond with ONLY a Python code block — no text before or after.

\`\`\`python
# Your code here
\`\`\`

## Strategies

**Filter then chunk (saves sub-query budget):**
\`\`\`python
import re
# First filter with regex — free, no LLM calls needed
relevant = [line for line in context.splitlines() if re.search(r'keyword', line, re.I)]
print(f"Filtered to {len(relevant)} relevant lines out of {context.count(chr(10))} total")
\`\`\`

**Chunked processing:**
\`\`\`python
lines = context.splitlines()
chunk_size = 200  # lines per chunk
chunks = ["\n".join(lines[i:i+chunk_size]) for i in range(0, len(lines), chunk_size)]
results = []
for i, chunk in enumerate(chunks):
    r = llm_query(chunk, "Extract all dates and events mentioned. Return as JSON list.")
    results.append(r)
    print(f"Chunk {i+1}/{len(chunks)} → {len(r)} chars")
\`\`\`

**Parallel sub-queries (faster):**
\`\`\`python
import asyncio
tasks = [async_llm_query(chunk, "Summarize the key facts") for chunk in chunks]
results = await asyncio.gather(*tasks)
combined = "\n".join(results)
final = llm_query(combined, f"Given these summaries, answer: {query}")
FINAL(final)
\`\`\`

**Building answer across iterations:**
\`\`\`python
# Iteration N: collect and store
collected_results = results  # from previous iterations stored in REPL
# Last iteration: synthesize
FINAL_VAR(llm_query("\n".join(collected_results), f"Answer the query: {query}"))
\`\`\``;
}

// ── Shared abort handler ────────────────────────────────────────────────

/**
 * Instead of adding one abort listener per concurrent promise (which triggers
 * MaxListenersExceededWarning at 11+), we register a single listener per
 * AbortSignal and fan-out to all pending reject callbacks.
 */
const pendingAborts = new Map<AbortSignal, Set<(err: Error) => void>>();

function addSharedAbortHandler(signal: AbortSignal, reject: (err: Error) => void) {
	let handlers = pendingAborts.get(signal);
	if (!handlers) {
		handlers = new Set();
		pendingAborts.set(signal, handlers);
		signal.addEventListener(
			"abort",
			() => {
				const current = pendingAborts.get(signal);
				if (current) {
					for (const handler of current) {
						handler(new Error("Aborted"));
					}
					pendingAborts.delete(signal);
				}
			},
			{ once: true },
		);
	}
	handlers.add(reject);
}

function removeSharedAbortHandler(signal: AbortSignal, reject: (err: Error) => void) {
	const handlers = pendingAborts.get(signal);
	if (handlers) {
		handlers.delete(reject);
		if (handlers.size === 0) {
			pendingAborts.delete(signal);
		}
	}
}

/** Race a promise against an AbortSignal so Ctrl+C cancels long API calls. */
function raceAbort<T>(promise: Promise<T>, signal?: AbortSignal): Promise<T> {
	if (!signal) return promise;
	if (signal.aborted) return Promise.reject(new Error("Aborted"));

	return new Promise<T>((resolve, reject) => {
		addSharedAbortHandler(signal, reject);
		promise.then(resolve).catch(reject).finally(() => {
			removeSharedAbortHandler(signal, reject);
		});
	});
}

// ── Helpers ─────────────────────────────────────────────────────────────────

function buildContextMetadata(context: string): string {
	const lines = context.split("\n");
	const charCount = context.length;
	const lineCount = lines.length;

	const previewStart = lines.slice(0, config.metadata_preview_lines).join("\n");
	const previewEnd = lines.slice(-config.metadata_preview_lines).join("\n");

	return [
		`Context statistics:`,
		`  - ${charCount.toLocaleString()} characters`,
		`  - ${lineCount.toLocaleString()} lines`,
		``,
		`First ${config.metadata_preview_lines} lines:`,
		previewStart,
		``,
		`Last ${config.metadata_preview_lines} lines:`,
		previewEnd,
	].join("\n");
}

function extractCodeFromResponse(response: AssistantMessage): string | null {
	for (const block of response.content) {
		if (block.type !== "text") continue;
		const text = (block as TextContent).text;

		// Try ```python or ```repl blocks
		const fenceMatch = text.match(/```(?:python|repl)?\s*\n([\s\S]*?)```/);
		if (fenceMatch) return fenceMatch[1].trim();

		// Fallback: if the response looks like raw Python code (require Python-specific patterns)
		const trimmed = text.trim();
		if (
			trimmed &&
			!trimmed.startsWith("#") &&
			(trimmed.includes("print(") ||
				trimmed.includes("import ") ||
				trimmed.includes("for ") && trimmed.includes(":") ||
				trimmed.includes("def ") && trimmed.includes(":") ||
				trimmed.includes("FINAL(") ||
				trimmed.includes("llm_query("))
		) {
			return trimmed;
		}
	}
	return null;
}

function truncateOutput(text: string): string {
	if (text.length <= config.truncate_len) {
		if (text.length === 0) return "[EMPTY OUTPUT]";
		return text;
	}
	return `[TRUNCATED: Last ${config.truncate_len} chars shown].. ${text.slice(-config.truncate_len)}`;
}

// ── Main loop ───────────────────────────────────────────────────────────────

export async function runRlmLoop(options: RlmOptions): Promise<RlmResult> {
	const { context, query, model, subModel, repl, signal, onProgress, onSubQueryStart, onSubQuery } = options;

	// Use subModel for sub-queries if provided; fall back to root model
	const subCallModel = subModel ?? model;

	let totalSubQueries = 0;
	let iterationSubQueries = 0;
	let totalInputTokens = 0;
	let totalOutputTokens = 0;

	// depth tracks recursion level; currently max_depth = 1 (sub-calls are flat LLMs, not nested RLMs)
	const currentDepth = 0;

	const llmQueryHandler = async (subContext: string, instruction: string) => {
		if (signal?.aborted) throw new Error("Aborted");
		if (totalSubQueries >= config.max_sub_queries) {
			return `[ERROR] Maximum sub-query limit (${config.max_sub_queries}) reached. Call FINAL() with your best answer now.`;
		}
		// Depth check: sub-queries are always depth+1; we cap at max_depth
		if (currentDepth >= config.max_depth) {
			return `[ERROR] Maximum recursion depth (${config.max_depth}) reached. Call FINAL() with your best answer now.`;
		}
		++totalSubQueries;
		const queryIndex = ++iterationSubQueries;
		const sqStart = Date.now();

		onSubQueryStart?.({
			index: queryIndex,
			contextLength: subContext.length,
			instruction,
		});

		const response = await raceAbort(
			completeSimple(subCallModel, {
				systemPrompt: `You are a helpful assistant. Answer the user's question based on the provided context. Be concise but thorough. Do not write code — respond in natural language.`,
				messages: [
					{
						role: "user",
						content: `Context:\n${subContext}\n\nInstruction: ${instruction}`,
						timestamp: Date.now(),
					},
				],
			}),
			signal,
		);

		const textParts = response.content.filter((b): b is TextContent => b.type === "text").map((b) => b.text);
		const result = textParts.join("\n");

		// Track tokens if provider returns them
		if ("inputTokens" in response && typeof response.inputTokens === "number") {
			totalInputTokens += response.inputTokens;
		}
		if ("outputTokens" in response && typeof response.outputTokens === "number") {
			totalOutputTokens += response.outputTokens;
		}

		onSubQuery?.({
			index: queryIndex,
			contextLength: subContext.length,
			instruction,
			resultLength: result.length,
			resultPreview: result,
			elapsedMs: Date.now() - sqStart,
		});

		return result;
	};

	/** Set up (or re-set up) the REPL with context and handler. */
	async function initRepl() {
		await repl.setContext(context);
		await repl.resetFinal();
		repl.setLlmQueryHandler(llmQueryHandler);
	}

	await initRepl();

	const metadata = buildContextMetadata(context);
	const conversationHistory: Message[] = [
		{
			role: "user",
			content: `${metadata}\n\nQuery: ${query}`,
			timestamp: Date.now(),
		} satisfies UserMessage,
	];

	for (let iteration = 1; iteration <= config.max_iterations; iteration++) {
		iterationSubQueries = 0;
		if (signal?.aborted) {
			return { answer: "[Aborted]", iterations: iteration, totalSubQueries, completed: false };
		}

		const lastUserMsg = conversationHistory
			.filter((m) => m.role === "user")
			.at(-1);
		const userMsgText =
			typeof lastUserMsg?.content === "string"
				? lastUserMsg.content
				: "";

		onProgress?.({
			iteration,
			maxIterations: config.max_iterations,
			subQueries: totalSubQueries,
			phase: "generating_code",
			userMessage: userMsgText,
			systemPrompt: iteration === 1 ? buildSystemPrompt({ iteration, maxIterations: config.max_iterations, subQueriesUsed: totalSubQueries, maxSubQueries: config.max_sub_queries, hasSubModel: !!subModel }) : undefined,
		});

		const systemPrompt = buildSystemPrompt({
			iteration,
			maxIterations: config.max_iterations,
			subQueriesUsed: totalSubQueries,
			maxSubQueries: config.max_sub_queries,
			hasSubModel: !!subModel,
		});

		let response;
		try {
			response = await raceAbort(
				completeSimple(model, {
					systemPrompt,
					messages: conversationHistory,
				}),
				signal,
			);
		} catch (apiErr) {
			if (signal?.aborted) {
				return { answer: "[Aborted]", iterations: iteration, totalSubQueries, completed: false };
			}
			const errMsg = apiErr instanceof Error ? apiErr.message : String(apiErr);
			return {
				answer: `[API Error] ${errMsg}`,
				iterations: iteration,
				totalSubQueries,
				completed: false,
			};
		}

		if (signal?.aborted) {
			return { answer: "[Aborted]", iterations: iteration, totalSubQueries, completed: false };
		}

		// Track root model token usage
		if ("inputTokens" in response && typeof response.inputTokens === "number") {
			totalInputTokens += response.inputTokens;
		}
		if ("outputTokens" in response && typeof response.outputTokens === "number") {
			totalOutputTokens += response.outputTokens;
		}

		// Surface API errors — bail immediately on unrecoverable errors
		if ("errorMessage" in response && response.errorMessage) {
			const errMsg = response.errorMessage as string;
			const isAuth = errMsg.includes("authentication") || errMsg.includes("401");
			const isQuota = errMsg.includes("quota") || errMsg.includes("billing") || errMsg.includes("429") || errMsg.includes("rate");
			const isServer = errMsg.includes("500") || errMsg.includes("502") || errMsg.includes("503") || errMsg.includes("overloaded");

			if (isAuth) {
				return {
					answer: `[API Error] ${errMsg}\n\nCheck your API key in .env or run /provider to reconfigure.`,
					iterations: iteration,
					totalSubQueries,
					completed: false,
				};
			}
			if (isQuota) {
				return {
					answer: `[API Error] ${errMsg}\n\nCheck your plan and billing at your provider's dashboard.`,
					iterations: iteration,
					totalSubQueries,
					completed: false,
				};
			}
			if (isServer) {
				return {
					answer: `[API Error] ${errMsg}\n\nThe provider's API is currently unavailable. Try again later.`,
					iterations: iteration,
					totalSubQueries,
					completed: false,
				};
			}
			// Unknown API error — still bail, don't waste iterations
			return {
				answer: `[API Error] ${errMsg}`,
				iterations: iteration,
				totalSubQueries,
				completed: false,
			};
		}

		const rawResponseText = response.content
			.filter((b): b is TextContent => b.type === "text")
			.map((b) => b.text)
			.join("\n");

		const code = extractCodeFromResponse(response);
		if (!code) {
			// No code block found — might be a direct answer or extraction failure
			conversationHistory.push(response);
			conversationHistory.push({
				role: "user",
				content: "Error: Could not extract code. Make sure to wrap your code in ```python ... ``` blocks.",
				timestamp: Date.now(),
			});
			continue;
		}

		conversationHistory.push(response);

		onProgress?.({
			iteration,
			maxIterations: config.max_iterations,
			subQueries: totalSubQueries,
			phase: "executing",
			code,
			rawResponse: rawResponseText,
		});

		let execResult: ExecResult;
		try {
			execResult = await repl.execute(code);
		} catch (err) {
			if (signal?.aborted) {
				return { answer: "[Aborted]", iterations: iteration, totalSubQueries, completed: false };
			}
			const errorMsg = err instanceof Error ? err.message : String(err);

			// If the REPL timed out or crashed, restart it so next iteration works
			if (errorMsg.includes("Timeout") || errorMsg.includes("not running") || errorMsg.includes("shut down")) {
				try {
					repl.shutdown();
					await repl.start(signal);
					await initRepl();
				} catch {
					return { answer: "[REPL crashed and could not restart]", iterations: iteration, totalSubQueries, completed: false };
				}
			}

			conversationHistory.push({
				role: "user",
				content: `Execution error: ${errorMsg}\n\nPlease fix the code and try again.`,
				timestamp: Date.now(),
			});
			continue;
		}

		if (signal?.aborted) {
			return { answer: "[Aborted]", iterations: iteration, totalSubQueries, completed: false };
		}

		onProgress?.({
			iteration,
			maxIterations: config.max_iterations,
			subQueries: totalSubQueries,
			phase: "checking_final",
			stdout: execResult.stdout,
			stderr: execResult.stderr,
		});

		if (execResult.hasFinal && execResult.finalValue !== null) {
			return {
				answer: execResult.finalValue,
				iterations: iteration,
				totalSubQueries,
				completed: true,
				inputTokens: totalInputTokens || undefined,
				outputTokens: totalOutputTokens || undefined,
			};
		}

		// Build next user message with truncated output
		const parts: string[] = [];
		if (execResult.stdout) {
			parts.push(`Output:\n${truncateOutput(execResult.stdout)}`);
		}
		if (execResult.stderr) {
			parts.push(`Stderr:\n${execResult.stderr.slice(0, 5000)}`);
		}
		if (parts.length === 0) {
			parts.push("(No output produced. The code ran without printing anything.)");
		}
		parts.push(
			`\nIteration ${iteration}/${config.max_iterations}. Sub-queries used: ${totalSubQueries}/${config.max_sub_queries}.`,
		);
		parts.push("Continue processing or call FINAL() when you have the answer.");

		conversationHistory.push({
			role: "user",
			content: parts.join("\n\n"),
			timestamp: Date.now(),
		});
	}

	return {
		answer: "[Maximum iterations reached without calling FINAL]",
		iterations: config.max_iterations,
		totalSubQueries,
		completed: false,
		inputTokens: totalInputTokens || undefined,
		outputTokens: totalOutputTokens || undefined,
	};
}
