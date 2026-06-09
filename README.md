# ollama-model-tester

A minimal Python benchmark for local [Ollama](https://ollama.com) models. Measures **tokens/sec** using Ollama's own response metadata.

## Requirements

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/)
- Ollama running locally

## Setup

```bash
uv sync
```

## Dependencies

| Package | Purpose |
|---------|---------|
| `ollama` | Streaming chat API |
| `rich` | Live terminal panel + stats table |

## Usage

```bash
uv run python ollama_bench.py <model> [prompt]
```

| Argument | Required | Default |
|----------|----------|---------|
| `model`  | ✅ yes   | —       |
| `prompt` | ❌ no    | `"Explain LLM quantization in 3 sentences."` |

## Examples

```bash
# default prompt
uv run python ollama_bench.py qwen3:1.7b

# custom prompt
uv run python ollama_bench.py llama3.2 "What is the capital of France?"

# multi-word prompt
uv run python ollama_bench.py mistral "Explain the difference between RAG and fine-tuning"
```

## Sample output

```
Prompt: Explain LLM quantization in 3 sentences.

╭─ qwen3:1.7b ──────────────────────────────────────────────────────╮
│ LLM quantization reduces the numerical precision of model weights  │
│ (e.g. from float32 to int4), dramatically shrinking model size and │
│ memory usage. This allows large models to run on consumer hardware │
│ with minimal accuracy loss. Common formats include GGUF/Q4_K_M …  │
╰─ ⚡ 143.2 tok/s  ·  187 tokens  ·  1.3s ─────────────────────────╯

╭──────────────────────┬────────────────╮
│ Metric               │          Value │
├──────────────────────┼────────────────┤
│ Tokens generated     │            187 │
│ Prompt tokens        │             14 │
│ Tokens / sec         │          143.2 │
│ Total time (ms)      │           1314 │
│ Measurement          │ ollama metadata│
╰──────────────────────┴────────────────╯
```

## How token/sec is calculated

Uses `eval_count / (eval_duration / 1e9)` from Ollama's response metadata — the same counters Ollama itself reports. Falls back to a wall-clock word-count estimate if the metadata is unavailable.
