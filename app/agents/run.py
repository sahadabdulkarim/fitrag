"""Unified entry point for the FitRAG multi-agent workflow."""

import sys
import os

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table

load_dotenv()

console = Console()


def display_workflow_result(state: dict):
    """Display the full workflow result in a formatted way."""

    # 1. Show parsed query
    parsed = state.get("parsed_query", {})
    if parsed:
        info_table = Table(title="📋 Parsed Query", show_header=False, border_style="cyan")
        info_table.add_column("Field", style="bold")
        info_table.add_column("Value")
        info_table.add_row("Goal", parsed.get("goal", "N/A"))
        info_table.add_row("Question", parsed.get("specific_question", "N/A"))
        info_table.add_row("Experience", parsed.get("experience_level", "N/A"))
        if parsed.get("injuries"):
            info_table.add_row("Injuries", ", ".join(parsed["injuries"]))
        if parsed.get("dietary_restrictions"):
            info_table.add_row("Diet", ", ".join(parsed["dietary_restrictions"]))
        if parsed.get("training_frequency"):
            info_table.add_row("Frequency", f"{parsed['training_frequency']}x/week")
        if parsed.get("body_stats"):
            stats = ", ".join(f"{k}={v}" for k, v in parsed["body_stats"].items())
            info_table.add_row("Stats", stats)
        console.print(info_table)

    # 2. Show safety flags (if any)
    safety_flags = state.get("safety_flags", [])
    if safety_flags:
        console.print("\n[bold red]⚠️  Safety Flags:[/bold red]")
        for flag in safety_flags:
            severity_color = {"info": "blue", "warning": "yellow", "critical": "red"}.get(flag["severity"], "white")
            console.print(f"  [{severity_color}][{flag['severity'].upper()}][/{severity_color}] {flag['concern']}")
            console.print(f"    → {flag['recommendation']}")

    # 3. Show workflow path
    workflow_path = state.get("workflow_path", [])
    if workflow_path:
        path_str = " → ".join(workflow_path)
        console.print(f"\n[dim]🔄 Workflow: {path_str}[/dim]")

    # 4. Show recommendation
    recommendation = state.get("recommendation", "")
    if recommendation:
        console.print("\n[bold green]═══ 🤖 Recommendation ═══[/bold green]\n")
        console.print(Markdown(recommendation))

    # 5. Show sources
    sources = state.get("sources_cited", [])
    if sources:
        console.print(f"\n[dim]📚 Sources: {', '.join(sources)}[/dim]")

    # 6. Show errors (if any)
    error = state.get("error")
    if error:
        console.print(f"\n[red]❌ Error: {error}[/red]")


def run_agent_query(user_input: str):
    """Run a query through the multi-agent workflow."""
    from app.agents.workflow import run_query

    console.print(Panel.fit(
        f"[bold cyan]Input:[/bold cyan] {user_input}",
        border_style="cyan",
    ))

    console.print("\n[dim]Running multi-agent workflow...[/dim]")

    final_state = run_query(user_input)
    display_workflow_result(final_state)


def interactive_agent_mode():
    """Interactive mode for the multi-agent workflow."""
    console.print(Panel.fit(
        "[bold green]🏋️ FitRAG - Multi-Agent Fitness Intelligence[/bold green]\n\n"
        "Powered by: LangGraph + Hybrid RAG + Groq Llama 3.3 70B\n\n"
        "[dim]Pipeline: Intake → Safety (if injury) → Retrieval → Recommendation[/dim]\n"
        "[dim]Type 'quit' to exit, 'help' for examples[/dim]",
        border_style="green",
    ))

    if not os.getenv("GROQ_API_KEY"):
        console.print("[red]❌ GROQ_API_KEY not set in .env[/red]")
        sys.exit(1)

    examples = [
        "How much protein do I need to build muscle?",
        "I'm 25, vegan, train 4x/week. How should I eat to gain muscle?",
        "I have a bad knee. What leg exercises are safe for me?",
        "Give me a training program for a beginner who wants to lose fat",
        "I'm 30, weigh 90kg, and want to lose 10kg. What's my calorie target?",
    ]

    while True:
        try:
            query = console.input("\n[bold cyan]❓ Ask FitRAG:[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye! 💪[/dim]")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            break
        if query.lower() == "help":
            console.print("\n[bold]Example queries (try these!):[/bold]")
            for i, ex in enumerate(examples, 1):
                console.print(f"  [cyan]{i}.[/cyan] {ex}")
            continue

        run_agent_query(query)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        run_agent_query(query)
    else:
        interactive_agent_mode()
