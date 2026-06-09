#!/usr/bin/env python3
"""
ollama_bench.py — measure tokens/sec for a local Ollama model.

Usage:
    python ollama_bench.py <model> [prompt]

Examples:
    python ollama_bench.py qwen3:1.7b
    python ollama_bench.py llama3.2 "What is the capital of France?"

Token/sec is derived from Ollama's own response metadata:
    eval_count      — tokens generated
    eval_duration   — nanoseconds spent generating
"""

import sys
import time
from ollama import chat


DEFAULT_PROMPT = "Explain LLM quantization in 3 sentences."


def benchmark(model: str, prompt: str) -> None:
    print(f"Model  : {model}")
    print(f"Prompt : {prompt}")
    print("-" * 60)

    t_start = time.perf_counter()

    response = chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )

    t_end = time.perf_counter()

    print(response.message.content)
    print("-" * 60)

    # Prefer Ollama's own counters; fall back to wall-clock if missing
    eval_count: int | None = getattr(response, "eval_count", None)
    eval_duration_ns: int | None = getattr(response, "eval_duration", None)

    if eval_count and eval_duration_ns and eval_duration_ns > 0:
        tokens_per_sec = eval_count / (eval_duration_ns / 1e9)
        source = "ollama metadata"
    else:
        # rough wall-clock fallback — includes network/overhead
        elapsed = t_end - t_start
        tokens_per_sec = len(response.message.content.split()) / elapsed if elapsed > 0 else 0
        source = "wall-clock (word estimate)"

    prompt_tokens: int = getattr(response, "prompt_eval_count", 0) or 0
    total_duration_ms = (getattr(response, "total_duration", None) or 0) / 1e6

    print(f"\n{'Metric':<25} {'Value':>12}")
    print(f"{'─'*25} {'─'*12}")
    print(f"{'Tokens generated':<25} {eval_count or '?':>12}")
    print(f"{'Prompt tokens':<25} {prompt_tokens:>12}")
    print(f"{'Tokens / sec':<25} {tokens_per_sec:>11.1f}")
    print(f"{'Total time (ms)':<25} {total_duration_ms:>11.0f}")
    print(f"{'Measurement source':<25} {source:>12}")


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <model> [prompt]", file=sys.stderr)
        print(f'Example: {sys.argv[0]} qwen3:1.7b "What is gravity?"', file=sys.stderr)
        sys.exit(1)

    model = sys.argv[1]
    prompt = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else DEFAULT_PROMPT

    benchmark(model, prompt)


if __name__ == "__main__":
    main()
