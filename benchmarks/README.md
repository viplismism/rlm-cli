# RLM Benchmarks

Evaluation scripts for testing the RLM CLI against standard long-context benchmarks.

## Prerequisites

Python dependencies are auto-installed into `.venv` on first run. The scripts prefer newer Python versions (3.13/3.12/3.11) over the system default.

## Available Benchmarks

### Oolong Synth (oolongbench/oolong-synth)
Synthetic long-context tasks: timeline ordering, user tracking, counting.

```bash
npm run bench:oolong

# Run with custom index
npx tsx benchmarks/oolong_synth.ts --idx 50
```

### LongBench NarrativeQA (THUDM/LongBench)
Reading comprehension over long narratives.

```bash
npm run bench:longbench

# Custom index
npx tsx benchmarks/longbench_narrativeqa.ts --idx 200
```

## How benchmarks work

Each benchmark:
1. Loads a dataset from HuggingFace via the `datasets` library
2. Extracts context + question for a given index
3. Runs it through the RLM loop
4. Compares the result against the expected answer
5. Saves trajectory for inspection
