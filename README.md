# rlm-cli

[![npm version](https://img.shields.io/npm/v/rlm-cli.svg)](https://www.npmjs.com/package/rlm-cli)
[![license](https://img.shields.io/npm/l/rlm-cli.svg)](https://github.com/viplismism/rlm-cli/blob/main/LICENSE)
[![node](https://img.shields.io/node/v/rlm-cli.svg)](https://nodejs.org/)

CLI for **Recursive Language Models** — based on the [RLM paper](https://arxiv.org/abs/2512.24601).

Instead of dumping a huge context into a single LLM call, RLM lets the model write Python code to process it — slicing, chunking, running sub-queries on pieces, and building up an answer across multiple iterations.

<p align="center">
  <img src="demo.png" alt="rlm-cli demo" width="750">
</p>

## What's New in v0.5.0

- **Ollama support** — use any locally-installed model (llama3, mistral, qwen, etc.) with zero API key setup
- **Mixed-model mode** — `sub_model` in config lets you use a cheap/fast model for sub-queries and a powerful model for the root loop (mirrors the paper's GPT-5 + GPT-5-mini setup)
- **Paper-aligned system prompt** — per-iteration budget awareness, sub-query strategy guidance, parallel async patterns from arXiv:2512.24601
- **Session-based trajectories** — runs grouped into `~/.rlm/sessions/<session-id>/` instead of a flat directory
- **Refreshed terminal UI** — Electric Amber RGB palette, two-column welcome panel with version in border, silent operation (no noise between queries)
- **Honest runtime limits** — `max_depth` is pinned to `1` because the current runtime implements flat paper-style sub-calls, not nested recursive RLM agents

---

## Install

```bash
npm install -g rlm-cli
```

Requires **Node.js >= 20** and **Python 3**.

Run `rlm` to start. First launch will prompt for a provider + API key (saved to `~/.rlm/credentials`).

---

## Supported Providers

| Provider | Env Variable | Default Model |
|----------|-------------|---------------|
| **Anthropic** | `ANTHROPIC_API_KEY` | `claude-sonnet-4-6` |
| **OpenAI** | `OPENAI_API_KEY` | `gpt-4o` |
| **Google** | `GEMINI_API_KEY` | `gemini-2.5-flash` |
| **OpenRouter** | `OPENROUTER_API_KEY` | `auto` |
| **Ollama** | _(no key needed)_ | any installed model |

### Ollama (local models)

If [Ollama](https://ollama.ai) is running, rlm-cli auto-detects it at startup — no config needed.

```bash
ollama pull llama3.1:8b
rlm
# → /model llama3.1:8b   or   /provider → choose Ollama
```

Set a custom daemon URL with `OLLAMA_BASE_URL=http://...`.

Keys are loaded from (highest priority wins):
1. Shell environment variables
2. `.env` file in project root
3. `~/.rlm/credentials`

### From Source

```bash
git clone https://github.com/viplismism/rlm-cli.git
cd rlm-cli
npm install
npm run build
npm link
```

---

## Usage

### Interactive Terminal

```bash
rlm
```

Persistent session with a two-column welcome panel showing your model, provider, context, and quick-ref slash commands. Everything auto-saves to a session folder.

**Slash commands:**

| Command | What it does |
|---------|-------------|
| `/file <path>` | Load file, directory, or glob as context |
| `/url <url>` | Fetch URL as context |
| `/paste` | Multi-line paste mode |
| `@file <query>` | Load file + run query in one step |
| `/model [id\|#]` | List or switch model (shows Ollama models too) |
| `/provider` | Switch provider (includes Ollama if running) |
| `/key` | Update an API key |
| `/trajectories` | Browse saved sessions |
| `/context` | Show loaded context info |
| `/clear` | Clear screen |
| `/help` | Full command reference |
| `/quit` | Exit |

**Tips:**
- Just type a question — no context needed for general queries
- Paste a URL directly to fetch it as context
- Paste 4+ lines of text to set it as context
- **Ctrl+C** stops a running query, **Ctrl+C twice** exits

### Single-Shot Mode

```bash
rlm run "Explain recursive language models"
rlm run --file large-file.txt "List all classes and their methods"
rlm run --url https://example.com/data.txt "Summarize this"
cat data.txt | rlm run --stdin "Count the errors"
rlm run --model gpt-4o --file code.py "Find bugs"
```

Answer goes to stdout, progress to stderr — pipe-friendly.

### Trajectory Viewer

```bash
rlm viewer
```

Browse saved runs in a TUI. Navigate iterations, inspect code and output at each step, drill into sub-queries. Sessions are saved to `~/.rlm/sessions/`.

---

## Benchmarks

Compare direct LLM vs RLM on the same query from standard long-context datasets.

| Benchmark | Dataset | What it tests |
|-----------|---------|---------------|
| `oolong` | [Oolong Synth](https://huggingface.co/datasets/oolongbench/oolong-synth) | Synthetic long-context: timeline ordering, user tracking, counting |
| `longbench` | [LongBench NarrativeQA](https://huggingface.co/datasets/THUDM/LongBench) | Reading comprehension over long narratives |

```bash
rlm benchmark oolong          # default: index 4743
rlm benchmark longbench       # default: index 182
rlm benchmark oolong --idx 10
```

Python dependencies are auto-installed into a `.venv` on first run.

---

## How It Works

1. Your full context is loaded into a persistent Python REPL as a `context` variable
2. The LLM gets metadata about the context (size, preview) plus your query
3. It writes Python code that can slice `context`, call `llm_query(chunk, instruction)` for sub-questions, and call `FINAL(answer)` when done
4. Code runs, output is captured and fed back for the next iteration
5. Loop continues until `FINAL()` is called or max iterations are reached

For large documents, the model chunks the text and runs parallel sub-queries with `async_llm_query()` + `asyncio.gather()`, then aggregates the results.

---

## Configuration

Create `rlm_config.yaml` in your working directory:

```yaml
max_iterations: 20       # Max iterations before giving up (1-100)
max_depth: 1             # Fixed at 1 in the current runtime
max_sub_queries: 50      # Max total sub-queries (1-500)
truncate_len: 5000       # Truncate REPL output beyond this (500-50000)
metadata_preview_lines: 20

# Use a cheaper/faster model for sub-queries (paper: GPT-5-mini for sub-calls)
# sub_model: gpt-4o-mini
# sub_model: claude-haiku-3-5
# sub_model: llama3.1:8b        # Ollama model for free sub-queries!
```

The `sub_model` option is the key cost-saving trick from the paper — a fast cheap model handles the chunking work while the root model synthesizes the final answer.

---

## Project Structure

```
src/
  main.ts          CLI entry point and command router
  interactive.ts   Interactive terminal REPL
  rlm.ts           Core RLM loop (Algorithm 1 from paper)
  repl.ts          Python REPL subprocess manager
  runtime.py       Python runtime (FINAL, llm_query, async_llm_query)
  cli.ts           Single-shot CLI mode
  viewer.ts        Trajectory viewer TUI
  colors.ts        Terminal color palette (Electric Amber RGB)
  ollama.ts        Ollama local model integration
  config.ts        Config loader
  env.ts           Environment variable loader
benchmarks/
  oolong_synth.ts
  longbench_narrativeqa.ts
  requirements.txt
bin/
  rlm.mjs          Global CLI shim
```

---

## License

MIT
