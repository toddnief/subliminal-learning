#!/usr/bin/env python3
"""TUI chat client for interacting with served models.

Auto-discovers all running vLLM servers (from serve_models.py) and
provides multi-turn conversation with model selection and switching
across all servers.

Usage:
    # Auto-discover servers (reads /tmp/vllm_servers.json):
    uv run scripts/serve/chat.py

    # Connect to a specific URL:
    uv run scripts/serve/chat.py --url http://localhost:8000

    # Start with a specific model:
    uv run scripts/serve/chat.py --model qwen2.5_7b-cat_numbers-r8

Controls:
    /models         - List available models across all servers
    /switch <name>  - Switch to a different model (supports partial match)
    /system <msg>   - Set system prompt
    /clear          - Clear conversation history
    /history        - Show conversation history
    /quit           - Exit
"""

import argparse
import json
import sys
from pathlib import Path

from openai import OpenAI
from rich.console import Console
from rich.markup import escape
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

console = Console()

SERVERS_FILE = Path("/tmp/vllm_servers.json")
QWEN_SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

DEFAULT_MODEL_DIRS = [
    Path("/net/projects/clab/tnief/entangled-tokens/models"),
    Path("/net/projects/clab/tnief/entangled-tokens/mark/models"),
]


def find_model_ft_config(model_name: str) -> dict | None:
    """Find ft_config.json for a model by searching known model directories."""
    for models_dir in DEFAULT_MODEL_DIRS:
        if not models_dir.exists():
            continue
        for d in models_dir.iterdir():
            if not d.is_dir():
                continue
            ft_cfg_path = d / "ft_config.json"
            if not ft_cfg_path.exists():
                continue
            with open(ft_cfg_path) as f:
                ft_cfg = json.load(f)
            # Match by hf_model_name or directory name
            if ft_cfg.get("hf_model_name") == model_name or d.name == model_name:
                return ft_cfg
    return None


def discover_servers() -> list[dict]:
    """Read server info written by serve_models.py."""
    if SERVERS_FILE.exists():
        with open(SERVERS_FILE) as f:
            return json.load(f)
    return []


def build_clients(servers: list[dict]) -> dict[str, OpenAI]:
    """Create OpenAI clients for each server URL."""
    clients = {}
    for s in servers:
        url = s["url"]
        if url not in clients:
            clients[url] = OpenAI(base_url=url, api_key="unused")
    return clients


def fetch_all_models(clients: dict[str, OpenAI]) -> list[tuple[str, str]]:
    """Fetch models from all servers.

    Returns:
        List of (model_id, server_url) tuples
    """
    all_models = []
    for url, client in clients.items():
        try:
            models = client.models.list()
            for m in models.data:
                all_models.append((m.id, url))
        except Exception as e:
            console.print(f"[red]Error fetching from {url}: {e}[/red]")
    return sorted(all_models, key=lambda x: x[0])


def pick_model(models: list[tuple[str, str]], query: str) -> tuple[str, str] | None:
    """Find a model by exact or partial name match.

    Returns:
        (model_id, server_url) or None
    """
    # Exact match
    for m_id, url in models:
        if query == m_id:
            return (m_id, url)
    # Partial match
    matches = [(m_id, url) for m_id, url in models if query in m_id]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        console.print(f"[yellow]Ambiguous match. Options:[/yellow]")
        for m_id, url in matches:
            console.print(f"  {m_id}  [dim]({url})[/dim]")
        return None
    console.print(f"[red]No model matching '{query}'[/red]")
    return None


def print_models(models: list[tuple[str, str]], current: str | None):
    """Display available models in a table."""
    table = Table(title="Available Models")
    table.add_column("#", style="dim")
    table.add_column("Model Name")
    table.add_column("Server", style="dim")
    table.add_column("Active", justify="center")
    for i, (m_id, url) in enumerate(models, 1):
        active = "[green]*[/green]" if m_id == current else ""
        table.add_row(str(i), m_id, url, active)
    console.print(table)


def chat_loop(
    clients: dict[str, OpenAI],
    models: list[tuple[str, str]],
    initial_model: tuple[str, str] | None,
):
    """Main interactive chat loop."""
    history: list[dict[str, str]] = []
    system_prompt: str | None = None
    current_model: str | None = initial_model[0] if initial_model else None
    current_url: str | None = initial_model[1] if initial_model else None

    if not current_model and models:
        console.print()
        print_models(models, None)
        console.print("\n[yellow]Pick a model with /switch <name>[/yellow]")

    console.print(Panel(
        "[bold]Commands:[/bold] /models  /switch <name>  /system <msg>  /sysqwen  /sysft  /clear  /history  /quit",
        title="Chat TUI",
        border_style="cyan",
    ))

    while True:
        try:
            console.print()
            label = f"[{escape(current_model)}]" if current_model else "[no model]"
            user_input = console.input(f"[bold cyan]{label}>[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd == "/quit":
                console.print("[dim]Goodbye.[/dim]")
                break
            elif cmd == "/models":
                models = fetch_all_models(clients)
                print_models(models, current_model)
            elif cmd == "/switch":
                if not arg:
                    console.print("[yellow]Usage: /switch <model_name>[/yellow]")
                    continue
                if arg.isdigit():
                    idx = int(arg) - 1
                    if 0 <= idx < len(models):
                        current_model, current_url = models[idx]
                        history.clear()
                        console.print(f"[green]Switched to {current_model} (history cleared)[/green]")
                    else:
                        console.print(f"[red]Invalid index. Use 1-{len(models)}[/red]")
                else:
                    picked = pick_model(models, arg)
                    if picked:
                        current_model, current_url = picked
                        history.clear()
                        console.print(f"[green]Switched to {current_model} (history cleared)[/green]")
            elif cmd == "/system":
                if not arg:
                    if system_prompt:
                        console.print(f"[dim]Current system prompt: {system_prompt}[/dim]")
                    else:
                        console.print("[dim]No system prompt set. Usage: /system <message>[/dim]")
                else:
                    system_prompt = arg
                    history.clear()
                    console.print(f"[green]System prompt set (history cleared)[/green]")
            elif cmd == "/sysqwen":
                system_prompt = QWEN_SYSTEM_PROMPT
                history.clear()
                console.print(f"[green]System prompt set to Qwen default (history cleared)[/green]")
                console.print(f"[dim]{system_prompt}[/dim]")
            elif cmd == "/sysft":
                if not current_model:
                    console.print("[yellow]No model selected.[/yellow]")
                else:
                    ft_cfg = find_model_ft_config(current_model)
                    if ft_cfg and ft_cfg.get("system_prompt"):
                        system_prompt = ft_cfg["system_prompt"]
                        history.clear()
                        console.print(f"[green]System prompt loaded from ft_config (history cleared)[/green]")
                        console.print(f"[dim]{system_prompt}[/dim]")
                    elif ft_cfg:
                        console.print("[yellow]No system_prompt in ft_config for this model[/yellow]")
                    else:
                        console.print(f"[yellow]No ft_config.json found for {current_model}[/yellow]")
            elif cmd == "/clear":
                history.clear()
                console.print("[green]Conversation cleared.[/green]")
            elif cmd == "/history":
                if not history:
                    console.print("[dim]No history.[/dim]")
                else:
                    for msg in history:
                        role = msg["role"]
                        style = "cyan" if role == "user" else "green"
                        console.print(f"[bold {style}]{role}:[/bold {style}] {msg['content']}")
            else:
                console.print(f"[red]Unknown command: {cmd}[/red]")
            continue

        if not current_model or not current_url:
            console.print("[yellow]No model selected. Use /switch <name> first.[/yellow]")
            continue

        # Send message
        history.append({"role": "user", "content": user_input})

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history)

        client = clients[current_url]
        try:
            console.print()
            with console.status("[dim]Thinking...[/dim]", spinner="dots"):
                response = client.chat.completions.create(
                    model=current_model,
                    messages=messages,
                )

            reply = response.choices[0].message.content
            history.append({"role": "assistant", "content": reply})
            console.print(Panel(Markdown(reply), border_style="green", title=escape(current_model), title_align="left"))

        except Exception as e:
            history.pop()  # Remove failed user message
            console.print(f"[red]Error: {e}[/red]")


def main():
    parser = argparse.ArgumentParser(description="Chat TUI for vLLM-served models")
    parser.add_argument(
        "--url",
        action="append",
        default=None,
        help="vLLM API base URL (can be repeated). Auto-discovers if not specified.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model to start chatting with (supports partial match)",
    )
    args = parser.parse_args()

    # Build client list
    if args.url:
        urls = args.url
    else:
        servers = discover_servers()
        if servers:
            urls = [s["url"] for s in servers]
            console.print(f"[dim]Auto-discovered {len(servers)} server(s) from {SERVERS_FILE}[/dim]")
        else:
            urls = ["http://localhost:8000/v1"]
            console.print("[dim]No server info found, trying localhost:8000...[/dim]")

    clients = {}
    for url in urls:
        console.print(f"[dim]Connecting to {url}...[/dim]")
        clients[url] = OpenAI(base_url=url, api_key="unused")

    models = fetch_all_models(clients)

    if not models:
        console.print("[red]No models available. Is the vLLM server running?[/red]")
        sys.exit(1)

    console.print(f"[green]Connected. {len(models)} models available across {len(clients)} server(s).[/green]")

    initial = None
    if args.model:
        initial = pick_model(models, args.model)

    chat_loop(clients, models, initial)


if __name__ == "__main__":
    main()
