# RLM Benchmarks

Evaluation scripts for testing the RLM CLI against standard long-context benchmarks.

## Prerequisites

Set up a Python virtual environment with the required dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install -r benchmarks/requirements.txt
```

The benchmark scripts automatically detect and use `.venv/bin/python3` if present, falling back to the system `python3`.

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
