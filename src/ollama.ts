/**
 * Ollama local model integration.
 *
 * Ollama exposes an OpenAI-compatible `/v1` endpoint, so we create
 * Model<"openai-completions"> objects for each locally-installed model
 * and register them in rlm-cli's in-memory model registry.
 */

import type { Model } from "@mariozechner/pi-ai";

export const OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434";

export interface OllamaModel {
	name: string;
	size: number;
	digest?: string;
}

/** Fetch the list of models from a running Ollama daemon. Returns empty array if Ollama is not running. */
export async function fetchOllamaModels(baseUrl = OLLAMA_DEFAULT_BASE_URL): Promise<OllamaModel[]> {
	try {
		const res = await fetch(`${baseUrl}/api/tags`, {
			signal: AbortSignal.timeout(3000),
		});
		if (!res.ok) return [];
		const data = (await res.json()) as { models?: Array<{ name: string; size: number; digest: string }> };
		return (data.models ?? []).map((m) => ({ name: m.name, size: m.size, digest: m.digest }));
	} catch {
		return [];
	}
}

/** Check if Ollama daemon is reachable. */
export async function detectOllama(baseUrl = OLLAMA_DEFAULT_BASE_URL): Promise<boolean> {
	try {
		const res = await fetch(`${baseUrl}/api/tags`, { signal: AbortSignal.timeout(2000) });
		return res.ok;
	} catch {
		return false;
	}
}

/** Create a pi-ai Model object for an Ollama model. */
export function createOllamaModel(modelName: string, baseUrl = OLLAMA_DEFAULT_BASE_URL): Model<"openai-completions"> {
	return {
		id: modelName,
		name: `ollama/${modelName}`,
		api: "openai-completions" as const,
		provider: "ollama",
		baseUrl: `${baseUrl}/v1`,
		reasoning: false,
		input: ["text"] as ("text" | "image")[],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 128000,
		maxTokens: 4096,
		// Ollama doesn't require a real API key; send a placeholder so the OpenAI
		// client doesn't reject the request for a missing Authorization header.
		headers: { Authorization: "Bearer ollama" },
		compat: {
			supportsStore: false,
			supportsDeveloperRole: false,
		},
	};
}

/** Format an Ollama model size in human-readable form. */
export function formatOllamaSize(bytes: number): string {
	const gb = bytes / 1e9;
	return gb >= 1 ? `${gb.toFixed(1)}GB` : `${(bytes / 1e6).toFixed(0)}MB`;
}
