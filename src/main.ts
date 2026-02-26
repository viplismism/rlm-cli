#!/usr/bin/env tsx
/**
 * rlm — Recursive Language Model CLI
 *
 * Entry point for the `rlm` command.
 *
 *   rlm              → interactive terminal (default)
 *   rlm run          → single-shot CLI run
 *   rlm help         → show help
 */

const HELP = `
\x1b[36m╔══════════════════════════════════════════════════════════════╗
║               rlm — Recursive Language Models                ║
║          CLI for large-context LLM processing                ║
║              arXiv:2512.24601                                ║
╚══════════════════════════════════════════════════════════════╝\x1b[0m

\x1b[1mUSAGE\x1b[0m
  \x1b[33mrlm\x1b[0m                          Interactive terminal (default)
  \x1b[33mrlm run\x1b[0m [options] "<query>"  Run a single query
  \x1b[33mrlm viewer\x1b[0m                    Browse saved trajectory files
  \x1b[33mrlm benchmark\x1b[0m <name> [--idx]  Run benchmark (direct LLM vs RLM)

\x1b[1mRUN OPTIONS\x1b[0m
  --model <id>     Override model (default: RLM_MODEL from .env)
  --file <path>    Read context from a file
  --url <url>      Fetch context from a URL
  --stdin          Read context from stdin
  --verbose        Show iteration progress

\x1b[1mCONFIGURATION\x1b[0m
  .env file:
    ANTHROPIC_API_KEY=sk-ant-...
    RLM_MODEL=claude-sonnet-4-5-20250929

  rlm_config.yaml:
    max_iterations: 20
    max_depth: 3
    max_sub_queries: 50
    truncate_len: 5000
`.trim();

async function main() {
	const args = process.argv.slice(2);
	const command = args[0] || "interactive";

	switch (command) {
		case "interactive":
		case "i": {
			await import("./interactive.js");
			break;
		}

		case "viewer":
		case "view": {
			// Strip the subcommand from argv so viewer.ts doesn't see it as a file path
			process.argv = [process.argv[0], process.argv[1], ...args.slice(1)];
			await import("./viewer.js");
			break;
		}

		case "run": {
			process.argv = [process.argv[0], process.argv[1], ...args.slice(1)];
			await import("./cli.js");
			break;
		}

		case "benchmark":
		case "bench": {
			const benchName = args[1];
			const benchArgs = args.slice(2);

			const benchScripts: Record<string, string> = {
				oolong: "benchmarks/oolong_synth.ts",
				longbench: "benchmarks/longbench_narrativeqa.ts",
			};

			if (benchName && benchScripts[benchName]) {
				const { spawn } = await import("node:child_process");
				const { dirname, join } = await import("node:path");
				const { fileURLToPath } = await import("node:url");
				const root = join(dirname(fileURLToPath(import.meta.url)), "..");
				const script = join(root, benchScripts[benchName]);
				const tsxBin = join(root, "node_modules", ".bin", "tsx");

				await new Promise<void>((resolve, reject) => {
					const child = spawn(tsxBin, [script, ...benchArgs], {
						stdio: "inherit",
						cwd: root,
					});
					child.on("exit", (code) => {
						process.exitCode = code ?? 1;
						resolve();
					});
					child.on("error", (err) => {
						reject(new Error(`Failed to spawn benchmark: ${err.message}`));
					});
				});
			} else {
				console.log(`\x1b[36m\x1b[1mrlm benchmark\x1b[0m — Run direct LLM vs RLM comparison\n`);
				console.log(`\x1b[1mUSAGE\x1b[0m`);
				console.log(`  \x1b[33mrlm benchmark oolong\x1b[0m    [--idx N]  Oolong Synth (synthetic long-context)`);
				console.log(`  \x1b[33mrlm benchmark longbench\x1b[0m [--idx N]  LongBench NarrativeQA (reading comprehension)\n`);
				console.log(`\x1b[1mSETUP\x1b[0m`);
				console.log(`  python3 -m venv .venv && .venv/bin/pip install -r benchmarks/requirements.txt\n`);
				console.log(`Each benchmark loads a dataset example, runs it through both direct LLM`);
				console.log(`and RLM, then prints a side-by-side comparison with timing.`);
			}
			break;
		}

		case "help":
		case "--help":
		case "-h": {
			console.log(HELP);
			break;
		}

		case "version":
		case "--version":
		case "-v": {
			try {
				const { readFileSync } = await import("node:fs");
				const { dirname, join } = await import("node:path");
				const { fileURLToPath } = await import("node:url");
				const __dir = dirname(fileURLToPath(import.meta.url));
				const pkgPath = join(__dir, "..", "package.json");
				const pkg = JSON.parse(readFileSync(pkgPath, "utf-8"));
				console.log(`rlm v${pkg.version}`);
			} catch {
				console.log("rlm v0.1.0");
			}
			break;
		}

		default: {
			if (command.startsWith("--")) {
				// Flags without subcommand → assume "run"
				await import("./cli.js");
			} else {
				console.error(`Unknown command: ${command}`);
				console.error('Run "rlm help" for usage information.');
				process.exit(1);
			}
		}
	}
}

main().catch((err) => {
	console.error("Fatal error:", err);
	process.exit(1);
});
