#!/usr/bin/env tsx
/**
 * Quick API key test — verifies your ANTHROPIC_API_KEY works.
 * Usage: npx tsx src/test-key.ts
 */
import "./env.js";

const { completeSimple, getModels, getProviders } = await import("@mariozechner/pi-ai");

let model: any;
for (const p of getProviders()) {
	for (const m of getModels(p)) {
		if (m.id === "claude-sonnet-4-20250514") model = m;
	}
}

if (!model) {
	console.log("❌ Model claude-sonnet-4-20250514 not found");
	process.exit(1);
}

console.log("Testing API key...");
const r = await completeSimple(model, {
	messages: [{ role: "user" as const, content: "Say hi in 3 words", timestamp: Date.now() }],
});

if ((r as any).errorMessage) {
	console.log(`❌ API ERROR: ${(r as any).errorMessage}`);
	process.exit(1);
} else {
	const text = r.content
		.filter((b: any) => b.type === "text")
		.map((b: any) => b.text)
		.join("");
	console.log(`✅ Key works! Response: "${text}"`);
}
