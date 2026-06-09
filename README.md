# ollama-model-tester

A minimal Python benchmark for local [Ollama](https://ollama.com) models. Measures **tokens/sec** using Ollama's own response metadata.

## Requirements

```bash
pip install ollama
```

## Usage

```bash
python ollama_bench.py <model> [prompt]
```

| Argument | Required | Default |
|----------|----------|---------|
| `model`  | ✅ yes   | —       |
| `prompt` | ❌ no    | `"Explain LLM quantization in 3 sentences."` |

## Examples

```bash
# default prompt
python ollama_bench.py qwen3:1.7b

# custom prompt
python ollama_bench.py llama3.2 "What is the capital of France?"

# multi-word prompt
python ollama_bench.py mistral "Explain the difference between RAG and fine-tuning"
```

## Sample output

```
Model  : qwen3:1.7b
Prompt : Explain LLM quantization in 3 sentences.
------------------------------------------------------------
LLM quantization reduces the precision of model weights ...
------------------------------------------------------------

Metric                         Value
───────────────────────── ────────────
Tokens generated                  187
Prompt tokens                      14
Tokens / sec                    142.3
Total time (ms)                  1314
Measurement source    ollama metadata
```

## How token/sec is calculated

Uses `eval_count / (eval_duration / 1e9)` from Ollama's response metadata — the same counters Ollama itself reports. Falls back to a wall-clock word-count estimate if the metadata is unavailable.
