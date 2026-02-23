# rlm-cli

CLI for **Recursive Language Models** — based on the [RLM paper](https://arxiv.org/abs/2512.24601).

Instead of dumping a huge context into a single LLM call, RLM lets the model write Python code to process it — slicing, chunking, running sub-queries on pieces, and building up an answer across multiple iterations.

## Quick Start

```bash
# Install
git clone https://github.com/viplismism/rlm-cli.git
cd rlm-cli
npm install
npm run build
npm link   # makes `rlm` available globally

# Configure
cp .env.example .env
# Edit .env with your API key
```

### `.env` file

```bash
ANTHROPIC_API_KEY=sk-ant-...
# or
OPENAI_API_KEY=sk-...

# Optional: override default model
RLM_MODEL=claude-sonnet-4-5-20250929
```

## Usage

### Interactive Terminal (default)

```bash
rlm
```

Opens a persistent REPL session where you can:

- **Set context** — paste a URL, file path, or multi-line text
- **Ask queries** — type a question and watch the RLM loop run with real-time display of code, output, sub-queries
- **Browse trajectories** — `/trajectories` lists saved runs

**Commands inside the terminal:**

| Command | Description |
|---------|-------------|
| `/file <path>` | Load a file as context |
| `/url <url>` | Fetch a URL as context |
| `/paste` | Multi-line paste mode (type `EOF` to finish) |
| `/context` | Show loaded context info |
| `/clear-context` | Unload context |
| `/trajectories` | List saved trajectory files |
| `/clear` | Clear screen |
| `/quit` | Exit |

You can also use shorthands: `@path/to/file.txt what does this do?` loads the file and runs the query in one line.

**Keyboard shortcuts:**

- `Ctrl+C` — stop the running query and return to prompt
- `Ctrl+C` twice — exit the terminal

### Single-Shot CLI

```bash
rlm run --file large-file.txt "List all classes and their methods"
rlm run --url https://example.com/big.py "Summarize this code"
cat data.txt | rlm run --stdin "Count the errors"
```

Options: `--model <id>`, `--file <path>`, `--url <url>`, `--stdin`, `--verbose`

The answer goes to stdout, progress to stderr — so you can pipe results.

### Trajectory Viewer

```bash
rlm viewer
```

TUI for browsing saved trajectory files. Navigate iterations, inspect code/output, drill into individual sub-queries.

**Keys:** `up/down` navigate, `enter` drill in, `esc` go back, `q` quit.

### Benchmarks

Compare direct LLM vs RLM on the same query from standard datasets.

```bash
# Setup (one-time)
python3 -m venv .venv
.venv/bin/pip install -r benchmarks/requirements.txt

# Run
rlm benchmark oolong      # Oolong Synth — synthetic long-context tasks
rlm benchmark longbench   # LongBench NarrativeQA — reading comprehension

# Pick a specific example
rlm benchmark oolong --idx 42
rlm benchmark longbench --idx 75
```

Each benchmark loads a dataset example, runs both direct LLM and RLM, and prints a side-by-side comparison with timing and sub-query stats.

## Configuration

`rlm_config.yaml` in the project root:

```yaml
max_iterations: 20       # Max iterations per query
max_depth: 3             # Max recursive sub-agent depth
max_sub_queries: 50      # Max total sub-queries across all depths
truncate_len: 5000       # Truncate REPL output beyond this many chars
metadata_preview_lines: 20  # Preview lines shown in context metadata
```

## How It Works

1. The full context is injected into a persistent Python REPL as a `context` variable
2. The LLM receives metadata about the context (size, preview) plus the user's query
3. The LLM writes Python code that can inspect/slice `context`, call `llm_query(sub_context, instruction)` for sub-tasks, and call `FINAL(answer)` when done
4. Code is executed, output is captured, and fed back to the LLM for the next iteration
5. The loop continues until `FINAL()` is called or max iterations are reached

Sub-queries (`llm_query`) send a chunk of context to a fresh LLM call with an instruction, enabling recursive decomposition of large documents. Parallel sub-queries are supported via `async_llm_query()` with `asyncio.gather()`.

## Project Structure

```
src/
  main.ts          CLI entry point and command router
  interactive.ts   Interactive terminal REPL
  rlm.ts           Core RLM loop (Algorithm 1)
  repl.ts          Python REPL subprocess manager
  runtime.py       Python runtime (FINAL, llm_query, async_llm_query)
  cli.ts           Single-shot CLI mode
  viewer.ts        Trajectory viewer TUI
  config.ts        Config loader (rlm_config.yaml)
  env.ts           .env file loader
benchmarks/
  oolong_synth.ts           Oolong Synth benchmark
  longbench_narrativeqa.ts  LongBench NarrativeQA benchmark
  requirements.txt          Python deps for benchmarks
bin/
  rlm.mjs          Global CLI shim
```

## Requirements

- Node.js >= 20
- Python 3 (for the REPL runtime)
- An API key for Anthropic, OpenAI, or OpenRouter

## License

MIT
