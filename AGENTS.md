# AGENTS.md — ollama-model-tester

## Purpose

Benchmark local [Ollama](https://ollama.com) models and report **tokens/sec** throughput.

## Stack

- Python ≥ 3.11
- [`uv`](https://docs.astral.sh/uv/) for dependency and environment management
- [`ollama`](https://pypi.org/project/ollama/) Python client

## Setup

```bash
uv sync
```

## Running the benchmark

```bash
# default prompt
uv run python ollama_bench.py <model>

# custom prompt
uv run python ollama_bench.py <model> "Your prompt here"
```

| Argument | Required | Default |
|----------|----------|---------|
| `model`  | ✅       | —       |
| `prompt` | ❌       | `"Explain LLM quantization in 3 sentences."` |

## Example

```bash
uv run python ollama_bench.py qwen3:1.7b
uv run python ollama_bench.py llama3.2 "What is the capital of France?"
```

## How tokens/sec is measured

Uses `eval_count / (eval_duration_ns / 1e9)` from Ollama's own response metadata.
Falls back to a wall-clock word-count estimate if metadata is unavailable.

## Agent rules

- Never commit without explicit owner approval.
- Keep `pyproject.toml` and `uv.lock` in sync — run `uv sync` after changing dependencies.
- `ollama_bench.py` is the single entry point; do not split into packages unless the file exceeds ~200 lines.
