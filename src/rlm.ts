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
}

// ── System prompt (inspired by fast-rlm) ────────────────────────────────────

function buildSystemPrompt(): string {
	return `You are a Recursive Language Model (RLM) agent. You process large contexts by writing Python code that runs in a persistent REPL.

## Available in the REPL

1. A \`context\` variable containing the full input text (may be very large). You should check the content of the \`context\` variable to understand what you are working with.

2. A \`llm_query(sub_context, instruction)\` function that sends a sub-piece of the context to an LLM with an instruction and returns the response. Use this for summarization, extraction, classification, etc. on chunks. For parallel queries, use \`async_llm_query()\` with \`asyncio.gather()\`.

3. Two functions to return your answer:
   - \`FINAL("your answer")\` — provide the answer as a string
   - \`FINAL_VAR(variable)\` — return a variable you built up in the REPL

## Rules

1. Write valid Python 3 code. You have access to the standard library.
2. Use \`print()\` to output metadata/intermediate results visible in the next iteration.
3. Use \`len(context)\` and slicing to understand the context size before processing.
4. For large contexts, split into chunks and use \`llm_query()\` on each chunk, then aggregate.
5. Call \`FINAL("answer")\` or \`FINAL_VAR(var)\` only when you have a complete answer.
6. Do NOT call FINAL prematurely — if you need more iterations, just print your intermediate state.
7. Be efficient: minimize the number of \`llm_query()\` calls by using smart chunking.
8. Print output will be truncated to last ${config.truncate_len} characters. Keep printed output concise.

## How to control sub-agent behavior

- When calling \`llm_query()\`, give clear instructions at the beginning of the context. If you only pass context without instructions, the sub-agent cannot do its task.
- To extract data verbatim: instruct the sub-agent to use \`FINAL_VAR\` and slice important sections.
- To summarize or analyze: instruct the sub-agent to explore and generate the answer.
- Help sub-agents by describing the data format (dict, list, etc.) — clarity is important!

## Important notes

- This is a multi-turn environment. You do NOT need to answer in one shot.
- Before returning via FINAL, it is advisable to print the answer first to inspect formatting.
- The REPL persists state like a Jupyter notebook — past variables and code are maintained. Do NOT rewrite old code or accidentally delete the \`context\` variable.
- You will only see truncated outputs, so use \`llm_query()\` for semantic analysis of large text.
- You can use variables as buffers to build up your final answer across iterations.

## Output format

Respond with ONLY a Python code block. No explanation before or after.

\`\`\`python
# Your working python code
print(f"Context length: {len(context)} chars")
\`\`\`

## Example strategies

**Chunking for large contexts:**
\`\`\`python
chunk_size = len(context) // 5
buffers = []
for i in range(5):
    start = i * chunk_size
    end = (i + 1) * chunk_size if i < 4 else len(context)
    chunk = context[start:end]
    result = llm_query(chunk, f"Extract key information relevant to: {query}")
    buffers.append(result)
    print(f"Chunk {i+1}/5 done: {len(result)} chars")
\`\`\`

**Parallel queries with asyncio:**
\`\`\`python
import asyncio
tasks = []
for i, chunk in enumerate(chunks):
    tasks.append(async_llm_query(chunk, f"Summarize chunk {i}"))
results = await asyncio.gather(*tasks)
\`\`\`

**Building up a final answer:**
\`\`\`python
# After collecting all results in a buffer
final_answer = llm_query("\\n".join(buffers), f"Synthesize these summaries to answer: {query}")
FINAL(final_answer)
\`\`\``;
}

// ── Abort helper ────────────────────────────────────────────────────────

/** Race a promise against an AbortSignal so Ctrl+C cancels long API calls. */
function raceAbort<T>(promise: Promise<T>, signal?: AbortSignal): Promise<T> {
	if (!signal) return promise;
	if (signal.aborted) return Promise.reject(new Error("Aborted"));
	return Promise.race([
		promise,
		new Promise<never>((_, reject) => {
			signal.addEventListener("abort", () => reject(new Error("Aborted")), { once: true });
		}),
	]);
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

		// Fallback: if the response looks like raw Python code
		const trimmed = text.trim();
		if (
			trimmed &&
			!trimmed.startsWith("#") &&
			(trimmed.includes("=") ||
				trimmed.includes("print") ||
				trimmed.includes("import") ||
				trimmed.includes("for ") ||
				trimmed.includes("def "))
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
	const { context, query, model, repl, signal, onProgress, onSubQueryStart, onSubQuery } = options;

	await repl.setContext(context);
	await repl.resetFinal();

	let totalSubQueries = 0;
	let iterationSubQueries = 0;

	repl.setLlmQueryHandler(async (subContext: string, instruction: string) => {
		if (signal?.aborted) throw new Error("Aborted");
		if (totalSubQueries >= config.max_sub_queries) {
			return `[ERROR] Maximum sub-query limit (${config.max_sub_queries}) reached. Call FINAL() with your best answer.`;
		}
		// Increment both counters; use per-iteration index for display
		++totalSubQueries;
		const queryIndex = ++iterationSubQueries;
		const sqStart = Date.now();

		// Notify: sub-query is starting
		onSubQueryStart?.({
			index: queryIndex,
			contextLength: subContext.length,
			instruction,
		});

		const response = await raceAbort(
			completeSimple(model, {
				systemPrompt: `You are a helpful assistant. Answer the user's question based on the provided context. Respond in natural language (not code). Be concise but thorough.`,
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

		// Notify: sub-query completed
		onSubQuery?.({
			index: queryIndex,
			contextLength: subContext.length,
			instruction,
			resultLength: result.length,
			resultPreview: result,
			elapsedMs: Date.now() - sqStart,
		});

		return result;
	});

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
			systemPrompt: iteration === 1 ? buildSystemPrompt() : undefined,
		});

		const response = await raceAbort(
			completeSimple(model, {
				systemPrompt: buildSystemPrompt(),
				messages: conversationHistory,
			}),
			signal,
		);

		if (signal?.aborted) {
			return { answer: "[Aborted]", iterations: iteration, totalSubQueries, completed: false };
		}

		// Surface API errors
		if ("errorMessage" in response && response.errorMessage) {
			const errMsg = response.errorMessage as string;
			if (errMsg.includes("authentication") || errMsg.includes("401")) {
				return {
					answer: `[API Authentication Error] ${errMsg}\n\nCheck your ANTHROPIC_API_KEY in .env.`,
					iterations: iteration,
					totalSubQueries,
					completed: false,
				};
			}
			process.stderr.write(`[rlm] API error: ${errMsg}\n`);
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
	};
}
