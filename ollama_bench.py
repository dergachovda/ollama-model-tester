#!/usr/bin/env python3
"""
ollama_bench.py — streaming benchmark for local Ollama models.

Streams the model response live inside a rich panel; the subtitle updates
with current tokens/sec in real-time. Final stats table on completion.

Usage:
    uv run python ollama_bench.py <model> [prompt]

Examples:
    uv run python ollama_bench.py qwen3:1.7b
    uv run python ollama_bench.py llama3.2 "What is the capital of France?"
"""

import sys
import time

from ollama import chat
from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

DEFAULT_PROMPT = "Explain LLM quantization in 3 sentences."

console = Console()


def _panel(model: str, content: str, tps: float, tokens: int, elapsed: float) -> Panel:
    if tokens == 0 and elapsed > 3.0:
        status = f"[yellow]loading model…[/yellow]  ·  {elapsed:.1f}s"
        border = "yellow"
    else:
        status = f"⚡ {tps:.1f} tok/s  ·  {tokens} tokens  ·  {elapsed:.1f}s"
        border = "cyan"
    return Panel(
        Text(content),
        title=f"[bold cyan]{model}[/bold cyan]",
        subtitle=f"[dim]{status}[/dim]",
        border_style=border,
        padding=(0, 1),
    )


def benchmark(model: str, prompt: str) -> None:
    console.print(f"[bold]Prompt:[/bold] {prompt}\n")

    content = ""
    chunk_count = 0
    eval_count = eval_duration_ns = prompt_tokens = total_duration_ns = None

    start = time.perf_counter()
    stream = chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    try:
        with Live(
            _panel(model, "", 0.0, 0, 0.0),
            refresh_per_second=15,
            vertical_overflow="visible",
            console=console,
        ) as live:
            for chunk in stream:
                if chunk.message and chunk.message.content is not None:
                    content += chunk.message.content
                    if chunk.message.content:
                        chunk_count += 1

                elapsed = time.perf_counter() - start
                tps = chunk_count / elapsed if elapsed > 0 else 0.0
                live.update(_panel(model, content, tps, chunk_count, elapsed))

                if chunk.done:
                    eval_count = chunk.eval_count
                    eval_duration_ns = chunk.eval_duration
                    prompt_tokens = chunk.prompt_eval_count
                    total_duration_ns = chunk.total_duration
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(0)

    elapsed = time.perf_counter() - start

    # Final accurate stats — prefer Ollama metadata over chunk-counting
    if eval_count and eval_duration_ns and eval_duration_ns > 0:
        final_tps = eval_count / (eval_duration_ns / 1e9)
        source = "ollama metadata"
    else:
        final_tps = chunk_count / elapsed if elapsed > 0 else 0.0
        source = "wall-clock estimate"

    total_ms = (total_duration_ns / 1e6) if total_duration_ns else elapsed * 1000

    table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="dim", min_width=22)
    table.add_column("Value", justify="right", min_width=14)
    table.add_row("Tokens generated", str(eval_count or chunk_count))
    table.add_row("Prompt tokens", str(prompt_tokens or "?"))
    table.add_row("[bold]Tokens / sec[/bold]", f"[bold green]{final_tps:.1f}[/bold green]")
    table.add_row("Total time (ms)", f"{total_ms:.0f}")
    table.add_row("Measurement", source)

    console.print()
    console.print(table)


def main() -> None:
    if len(sys.argv) < 2:
        console.print(
            f"[bold red]Usage:[/bold red] {sys.argv[0]} <model> [prompt]\n"
            f'[dim]Example: {sys.argv[0]} qwen3:1.7b "What is gravity?"[/dim]'
        )
        sys.exit(1)

    model = sys.argv[1]
    prompt = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else DEFAULT_PROMPT

    benchmark(model, prompt)


if __name__ == "__main__":
    main()
