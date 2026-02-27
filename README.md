# rlm-cli

```
                     ██████╗ ██╗     ███╗   ███╗
                     ██╔══██╗██║     ████╗ ████║
                     ██████╔╝██║     ██╔████╔██║
                     ██╔══██╗██║     ██║╚██╔╝██║
                     ██║  ██║███████╗██║ ╚═╝ ██║
                     ╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝
```

CLI for **Recursive Language Models** — based on the [RLM paper](https://arxiv.org/abs/2512.24601).

Instead of dumping a huge context into a single LLM call, RLM lets the model write Python code to process it — slicing, chunking, running sub-queries on pieces, and building up an answer across multiple iterations.

## Install

```bash
npm install -g rlm-cli
```

Set your API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
# or
export OPENAI_API_KEY=sk-...
```

That's it. Run `rlm` and you're in.

### From Source

```bash
git clone https://github.com/viplismism/rlm-cli.git
cd rlm-cli
npm install
npm run build
npm link
```

Create a `.env` file in the project root with your API key:

```bash
cp .env.example .env
```

## Usage

### Interactive Terminal

```bash
rlm
```

This is the main way to use it. You get a persistent session where you can:

- Load context from a file, URL, or by pasting text directly
- Ask questions and watch the RLM loop run — you'll see the code it writes, the output, sub-queries, everything in real-time
- All runs are saved as trajectory files you can browse later

You don't even need to load context first — just type a query directly and RLM will use your question as the context:

```bash
> what are the top 5 sorting algorithms and their time complexities?
```

Load context and ask in one shot:

```bash
> @path/to/file.txt what are the main functions here?
```

Or set context first, then ask multiple questions:

```bash
> /file big-codebase.py
> what does the main class do?
> find all the error handling patterns
```

**Ctrl+C** stops the current query. **Ctrl+C twice** exits.

Type `/help` inside the terminal for all commands.

### Single-Shot Mode

For scripting or one-off queries:

```bash
rlm run --file large-file.txt "List all classes and their methods"
rlm run --url https://example.com/data.txt "Summarize this"
cat data.txt | rlm run --stdin "Count the errors"
```

Answer goes to stdout, progress to stderr — pipe-friendly.

### Trajectory Viewer

```bash
rlm viewer
```

Browse saved runs in a TUI. Navigate iterations, inspect the code and output at each step, drill into individual sub-queries.

## Benchmarks

Compare direct LLM vs RLM on the same query from standard long-context datasets. This runs both approaches side-by-side so you can see the difference.

### Available Benchmarks

| Benchmark | Dataset | What it tests |
|-----------|---------|---------------|
| `oolong` | [Oolong Synth](https://huggingface.co/datasets/oolongbench/oolong-synth) | Synthetic long-context tasks: timeline ordering, user tracking, counting |
| `longbench` | [LongBench NarrativeQA](https://huggingface.co/datasets/THUDM/LongBench) | Reading comprehension over long narratives |

### Running

```bash
rlm benchmark oolong          # default: index 4743 (14.7MB timeline+subset counting)
rlm benchmark longbench       # default: index 182 (205KB multi-hop narrative reasoning)

# Pick a specific example from the dataset
rlm benchmark oolong --idx 10
rlm benchmark longbench --idx 50
```

Python dependencies are auto-installed into a `.venv` on first run.

Each run:
1. Loads one example from the dataset
2. Runs direct LLM (single prompt, no RLM)
3. Runs RLM (iterative code execution with sub-queries)
4. Prints both answers side-by-side with the expected answer, timing, and stats
5. Saves a trajectory file for later inspection with `rlm viewer`

## How It Works

1. Your full context is loaded into a persistent Python REPL as a `context` variable
2. The LLM gets metadata about the context (size, preview of first/last lines) plus your query
3. It writes Python code that can slice `context`, call `llm_query(chunk, instruction)` to ask sub-questions about pieces, and call `FINAL(answer)` when it has the answer
4. Code runs, output is captured and fed back for the next iteration
5. Loop continues until `FINAL()` is called or max iterations are reached

For large documents, the model typically chunks the text and runs parallel sub-queries with `async_llm_query()` + `asyncio.gather()`, then aggregates the results.

## Configuration

Edit `rlm_config.yaml` in the project root:

```yaml
max_iterations: 20       # Max iterations before giving up
max_depth: 3             # Max recursive sub-agent depth
max_sub_queries: 50      # Max total sub-queries
truncate_len: 5000       # Truncate REPL output beyond this
metadata_preview_lines: 20
```

## Project Structure

```
src/
  main.ts          CLI entry point and command router
  interactive.ts   Interactive terminal REPL
  rlm.ts           Core RLM loop
  repl.ts          Python REPL subprocess manager
  runtime.py       Python runtime (FINAL, llm_query, async_llm_query)
  cli.ts           Single-shot CLI mode
  viewer.ts        Trajectory viewer TUI
  config.ts        Config loader
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
- Python 3
- An API key (Anthropic, OpenAI, or OpenRouter)

## License

MIT
