#!/usr/bin/env python3
"""TUI chat client for interacting with served models.

Connects to a vLLM OpenAI-compatible API and provides multi-turn
conversation with model selection and switching.

Usage:
    # Connect to local vLLM server (default):
    uv run scripts/serve/chat.py

    # Connect to a specific host/port:
    uv run scripts/serve/chat.py --url http://localhost:8080

    # Start with a specific model:
    uv run scripts/serve/chat.py --model qwen2.5_7b-cat_numbers-r8

Controls:
    /models         - List available models
    /switch <name>  - Switch to a different model (supports partial match)
    /system <msg>   - Set system prompt
    /clear          - Clear conversation history
    /history        - Show conversation history
    /quit           - Exit
"""

import argparse
import sys

from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

console = Console()


def fetch_models(client: OpenAI) -> list[str]:
    """Fetch available model names from the server."""
    try:
        models = client.models.list()
        return sorted([m.id for m in models.data])
    except Exception as e:
        console.print(f"[red]Error fetching models: {e}[/red]")
        return []


def pick_model(models: list[str], query: str) -> str | None:
    """Find a model by exact or partial name match."""
    if query in models:
        return query
    matches = [m for m in models if query in m]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        console.print(f"[yellow]Ambiguous match. Options:[/yellow]")
        for m in matches:
            console.print(f"  {m}")
        return None
    console.print(f"[red]No model matching '{query}'[/red]")
    return None


def print_models(models: list[str], current: str | None):
    """Display available models in a table."""
    table = Table(title="Available Models")
    table.add_column("#", style="dim")
    table.add_column("Model Name")
    table.add_column("Active", justify="center")
    for i, m in enumerate(models, 1):
        active = "[green]*[/green]" if m == current else ""
        table.add_row(str(i), m, active)
    console.print(table)


def chat_loop(client: OpenAI, model: str | None, models: list[str]):
    """Main interactive chat loop."""
    history: list[dict[str, str]] = []
    system_prompt: str | None = None

    if not model and models:
        console.print()
        print_models(models, None)
        console.print("\n[yellow]Pick a model with /switch <name>[/yellow]")

    console.print(Panel(
        "[bold]Commands:[/bold] /models  /switch <name>  /system <msg>  /clear  /history  /quit",
        title="Chat TUI",
        border_style="cyan",
    ))

    while True:
        try:
            console.print()
            user_input = console.input(f"[bold cyan]{'[' + model + ']' if model else '[no model]'}>[/bold cyan] ").strip()
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
                models = fetch_models(client)
                print_models(models, model)
            elif cmd == "/switch":
                if not arg:
                    console.print("[yellow]Usage: /switch <model_name>[/yellow]")
                    continue
                # Support switching by number
                if arg.isdigit():
                    idx = int(arg) - 1
                    if 0 <= idx < len(models):
                        model = models[idx]
                        history.clear()
                        console.print(f"[green]Switched to {model} (history cleared)[/green]")
                    else:
                        console.print(f"[red]Invalid index. Use 1-{len(models)}[/red]")
                else:
                    picked = pick_model(models, arg)
                    if picked:
                        model = picked
                        history.clear()
                        console.print(f"[green]Switched to {model} (history cleared)[/green]")
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

        if not model:
            console.print("[yellow]No model selected. Use /switch <name> first.[/yellow]")
            continue

        # Send message
        history.append({"role": "user", "content": user_input})

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history)

        try:
            console.print()
            with console.status("[dim]Thinking...[/dim]", spinner="dots"):
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                )

            reply = response.choices[0].message.content
            history.append({"role": "assistant", "content": reply})
            console.print(Panel(Markdown(reply), border_style="green", title=model, title_align="left"))

        except Exception as e:
            history.pop()  # Remove failed user message
            console.print(f"[red]Error: {e}[/red]")


def main():
    parser = argparse.ArgumentParser(description="Chat TUI for vLLM-served models")
    parser.add_argument(
        "--url",
        default="http://localhost:8000/v1",
        help="vLLM API base URL (default: http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model to start chatting with (supports partial match)",
    )
    args = parser.parse_args()

    client = OpenAI(base_url=args.url, api_key="unused")

    console.print(f"[dim]Connecting to {args.url}...[/dim]")
    models = fetch_models(client)

    if not models:
        console.print("[red]No models available. Is the vLLM server running?[/red]")
        sys.exit(1)

    console.print(f"[green]Connected. {len(models)} models available.[/green]")

    model = None
    if args.model:
        model = pick_model(models, args.model)

    chat_loop(client, model, models)


if __name__ == "__main__":
    main()
