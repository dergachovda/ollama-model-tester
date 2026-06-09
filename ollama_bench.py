#!/usr/bin/env python3
"""
ollama_bench.py — streaming benchmark for local Ollama models.

Streams the model response live inside a rich panel; the subtitle updates
with current tokens/sec in real-time. Final stats table on completion.

Usage:
    uv run python ollama_bench.py <model> [prompt] [--debug]

Examples:
    uv run python ollama_bench.py qwen3:1.7b
    uv run python ollama_bench.py llama3.2 "What is the capital of France?"
    uv run python ollama_bench.py lfm2.5:8b --debug   # writes ollama_bench.log
"""

import json
import logging
import platform
import subprocess
import sys
import time
import urllib.request
from argparse import ArgumentParser
from pathlib import Path

import psutil
from ollama import chat
from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

DEFAULT_PROMPT = "Explain LLM quantization in 3 sentences."
LOG_FILE = Path("ollama_bench.log")

console = Console()
log = logging.getLogger("ollama_bench")


def _gpu_info() -> str:
    """Return GPU name(s) via nvidia-smi, or a generic fallback."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            timeout=5,
            text=True,
        ).strip()
        return " | ".join(line.strip() for line in out.splitlines() if line.strip())
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pass
    # Fallback: PowerShell Get-CimInstance (Windows 11+, wmic removed)
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"],
            stderr=subprocess.DEVNULL,
            timeout=8,
            text=True,
        )
        names = [l.strip() for l in out.splitlines() if l.strip()]
        return " | ".join(names) if names else "N/A"
    except Exception:
        return "N/A"


def _hardware_table() -> Table:
    mem = psutil.virtual_memory()
    cpu = platform.processor() or platform.machine()
    ram_gb = mem.total / 1024**3
    ram_avail_gb = mem.available / 1024**3

    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    t.add_column(style="dim")
    t.add_column()
    t.add_row("OS", f"{platform.system()} {platform.release()}")
    t.add_row("CPU", cpu)
    t.add_row("RAM", f"{ram_gb:.1f} GB total  ·  {ram_avail_gb:.1f} GB free")
    t.add_row("GPU", _gpu_info())
    return t


def _ollama_ps(model: str) -> dict:
    """Query /api/ps for the running model's memory placement."""
    try:
        with urllib.request.urlopen("http://localhost:11434/api/ps", timeout=3) as resp:
            data = json.loads(resp.read())
        model_base = model.split(":")[0].lower()
        for m in data.get("models", []):
            if model_base in m.get("name", "").lower():
                return m
    except Exception:
        pass
    return {}


def _processor_label(size: int, size_vram: int) -> str:
    if size <= 0:
        return "unknown"
    if size_vram <= 0:
        return "[red]CPU[/red]"
    if size_vram >= size * 0.95:
        return "[green]GPU[/green]"
    pct = size_vram / size * 100
    return f"[yellow]GPU+CPU ({pct:.0f}% in VRAM)[/yellow]"


def _setup_logging() -> None:
    LOG_FILE.unlink(missing_ok=True)
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.DEBUG,
        format="%(asctime)s.%(msecs)03d  %(message)s",
        datefmt="%H:%M:%S",
    )
    console.print(f"[dim]Debug logging → {LOG_FILE.resolve()}[/dim]\n")


def _log_chunk(i: int, chunk) -> None:
    msg = getattr(chunk, "message", "<no message attr>")
    content = getattr(msg, "content", "<no content attr>") if msg else None
    thinking = getattr(msg, "thinking", "<no thinking attr>") if msg else None
    role = getattr(msg, "role", None) if msg else None
    log.debug(
        "chunk #%04d  done=%-5s  role=%-10s  content=%r  thinking=%r  "
        "eval_count=%s  eval_duration=%s  prompt_eval_count=%s  total_duration=%s",
        i,
        getattr(chunk, "done", "?"),
        role,
        content,
        thinking,
        getattr(chunk, "eval_count", "-"),
        getattr(chunk, "eval_duration", "-"),
        getattr(chunk, "prompt_eval_count", "-"),
        getattr(chunk, "total_duration", "-"),
    )


def _panel(
    model: str,
    content: str,
    thinking: str,
    think_tps: float,
    think_tokens: int,
    tps: float,
    tokens: int,
    elapsed: float,
) -> Panel:
    if tokens == 0 and elapsed > 3.0:
        think_preview = thinking[-120:].replace("\n", " ") if thinking else ""
        status = f"[yellow]thinking…[/yellow]  ·  ⚡ {think_tps:.1f} think tok/s  ·  {think_tokens} tokens  ·  {elapsed:.1f}s"
        body = Text(f"💭 {think_preview}", style="dim italic") if think_preview else Text("")
        border = "yellow"
    else:
        status = f"⚡ {tps:.1f} tok/s  ·  {tokens} tokens  ·  {elapsed:.1f}s"
        body = Text(content)
        border = "cyan"
    return Panel(
        body,
        title=f"[bold cyan]{model}[/bold cyan]",
        subtitle=f"[dim]{status}[/dim]",
        border_style=border,
        padding=(0, 1),
    )


def benchmark(model: str, prompt: str, debug: bool) -> None:
    if debug:
        _setup_logging()

    console.print(Panel(_hardware_table(), title="[bold]Hardware[/bold]", border_style="dim", padding=(0, 1)))
    console.print(f"[bold]Prompt:[/bold] {prompt}\n")

    content = ""
    thinking = ""
    chunk_count = 0
    think_count = 0
    think_elapsed = 0.0
    chunk_index = 0
    eval_count = eval_duration_ns = prompt_tokens = total_duration_ns = load_duration_ns = None
    ps: dict = {}  # populated after first chunk while model is guaranteed in memory

    start = time.perf_counter()
    stream = chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    try:
        with Live(
            _panel(model, "", "", 0.0, 0, 0.0, 0, 0.0),
            refresh_per_second=15,
            vertical_overflow="visible",
            console=console,
        ) as live:
            for chunk in stream:
                if debug:
                    _log_chunk(chunk_index, chunk)
                chunk_index += 1

                # Capture memory placement once — model is guaranteed loaded
                if chunk_index == 1:
                    ps = _ollama_ps(model)

                if chunk.message:
                    if chunk.message.thinking:
                        thinking += chunk.message.thinking
                        think_count += 1
                    if chunk.message.content is not None:
                        content += chunk.message.content
                        if chunk.message.content:
                            chunk_count += 1

                elapsed = time.perf_counter() - start
                # freeze thinking elapsed once content starts arriving
                if chunk_count == 0:
                    think_elapsed = elapsed
                think_tps = think_count / think_elapsed if think_elapsed > 0 else 0.0
                tps = chunk_count / elapsed if elapsed > 0 else 0.0
                live.update(_panel(model, content, thinking, think_tps, think_count, tps, chunk_count, elapsed))

                if chunk.done:
                    eval_count = chunk.eval_count
                    eval_duration_ns = chunk.eval_duration
                    prompt_tokens = chunk.prompt_eval_count
                    total_duration_ns = chunk.total_duration
                    load_duration_ns = chunk.load_duration
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(0)

    if debug:
        log.debug(
            "stream ended  total_chunks=%d  think_chunks=%d  content_chunks=%d",
            chunk_index, think_count, chunk_count,
        )
        console.print(f"[dim]Log written → {LOG_FILE.resolve()}[/dim]")

    elapsed = time.perf_counter() - start

    # Final accurate stats — prefer Ollama metadata over chunk-counting
    if eval_count and eval_duration_ns and eval_duration_ns > 0:
        final_tps = eval_count / (eval_duration_ns / 1e9)
        source = "ollama metadata"
    else:
        final_tps = chunk_count / elapsed if elapsed > 0 else 0.0
        source = "wall-clock estimate"

    total_ms = (total_duration_ns / 1e6) if total_duration_ns else elapsed * 1000
    load_ms = (load_duration_ns / 1e6) if load_duration_ns else None

    # Query Ollama for processor placement and memory usage
    # (already captured during stream; try again if missed)
    if not ps:
        ps = _ollama_ps(model)
    size = ps.get("size", 0)
    size_vram = ps.get("size_vram", 0)

    table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="dim", min_width=24)
    table.add_column("Value", justify="right", min_width=16)

    # --- model loading ---
    if load_ms is not None:
        table.add_row("Model load time (ms)", f"{load_ms:.0f}")
    if size > 0:
        table.add_row("Processor", _processor_label(size, size_vram))
        table.add_row("Model size", f"{size / 1024**3:.2f} GB")
        if size_vram > 0:
            table.add_row("VRAM used", f"{size_vram / 1024**3:.2f} GB")
        ram_used = size - size_vram
        if ram_used > 0:
            table.add_row("RAM used", f"{ram_used / 1024**3:.2f} GB")

    table.add_section()

    # --- generation ---
    table.add_row("Tokens generated", str(eval_count or chunk_count))
    table.add_row("Prompt tokens", str(prompt_tokens or "?"))
    table.add_row("[bold]Tokens / sec[/bold]", f"[bold green]{final_tps:.1f}[/bold green]")
    if thinking:
        think_tps_final = think_count / think_elapsed if think_elapsed > 0 else 0.0
        table.add_row("Thinking tokens", str(think_count))
        table.add_row("[bold]Think tok / sec[/bold]", f"[bold yellow]{think_tps_final:.1f}[/bold yellow]")
        table.add_row("Thinking time (ms)", f"{think_elapsed * 1000:.0f}")
    table.add_row("Total time (ms)", f"{total_ms:.0f}")
    table.add_row("Measurement", source)

    console.print()
    console.print(table)


def main() -> None:
    parser = ArgumentParser(description="Benchmark a local Ollama model.")
    parser.add_argument("model", help="Model name (e.g. qwen3:1.7b)")
    parser.add_argument("prompt", nargs="*", help="Prompt text (default: quantization question)")
    parser.add_argument("--debug", action="store_true", help=f"Log every chunk to {LOG_FILE}")
    args = parser.parse_args()

    prompt = " ".join(args.prompt) if args.prompt else DEFAULT_PROMPT
    benchmark(args.model, prompt, args.debug)


if __name__ == "__main__":
    main()
