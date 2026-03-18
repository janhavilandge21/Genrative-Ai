"""
main.py — Entry point for the Agentic AI Research Assistant.

Usage:
    python main.py
    python main.py --query "Your research topic here"
    python main.py --query "AI trends 2024" --output report.md
"""

from __future__ import annotations

import os
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich import box

# Load environment variables from .env file
load_dotenv()

console = Console()

EXAMPLE_QUERIES = [
    "What are the latest breakthroughs in AI and large language models in 2024?",
    "Explain the current state of quantum computing and its real-world applications",
    "What is the impact of climate change on global food security?",
    "How is generative AI transforming the healthcare industry?",
    "What are the key trends in cybersecurity threats and defenses in 2024?",
    "Research the current state of electric vehicles and battery technology",
]

BANNER = """
╔══════════════════════════════════════════════════════════╗
║       🔬  Agentic AI Research Assistant                  ║
║       Powered by LangGraph + Groq (llama3-70b)           ║
╚══════════════════════════════════════════════════════════╝
"""


def validate_env() -> bool:
    """Check required environment variables are set."""
    missing = []
    if not os.environ.get("GROQ_API_KEY"):
        missing.append("GROQ_API_KEY")
    if not os.environ.get("TAVILY_API_KEY"):
        missing.append("TAVILY_API_KEY")

    if missing:
        console.print(f"\n[bold red]❌ Missing environment variables: {', '.join(missing)}[/bold red]")
        console.print("\nCreate a [bold].env[/bold] file in this directory with:")
        for key in missing:
            console.print(f"  [yellow]{key}=your_key_here[/yellow]")
        console.print("\n  • Groq API key: https://console.groq.com")
        console.print("  • Tavily API key: https://app.tavily.com\n")
        return False
    return True


def show_examples():
    """Display example queries in a rich table."""
    table = Table(title="📋 Example Research Queries", box=box.ROUNDED, show_lines=True)
    table.add_column("#", style="cyan", width=4)
    table.add_column("Query", style="white")

    for i, q in enumerate(EXAMPLE_QUERIES, 1):
        table.add_row(str(i), q)

    console.print(table)


def get_query_interactive() -> str:
    """Prompt user to enter a query or pick an example."""
    console.print("\n[bold cyan]Choose an option:[/bold cyan]")
    console.print("  [green]1-6[/green] — Run an example query")
    console.print("  [green]c[/green]   — Enter a custom query")
    console.print("  [green]q[/green]   — Quit\n")

    while True:
        choice = console.input("[bold]> [/bold]").strip()

        if choice.lower() == "q":
            console.print("\n[yellow]Goodbye![/yellow]\n")
            sys.exit(0)

        if choice.lower() == "c":
            return console.input("\n[bold cyan]Enter your research query:[/bold cyan]\n> ").strip()

        if choice.isdigit() and 1 <= int(choice) <= len(EXAMPLE_QUERIES):
            return EXAMPLE_QUERIES[int(choice) - 1]

        console.print("[red]Invalid choice. Try again.[/red]")


def run_research(query: str, output_file: str | None = None) -> None:
    """Run the research agent and display / save results."""
    from agent import ResearchAssistant

    console.print(f"\n[bold]🔍 Research Query:[/bold] {query}\n")

    start_time = time.time()
    result = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("🤖 Agent is researching... (this may take 30-90 seconds)", total=None)

        assistant = ResearchAssistant()
        result = assistant.research(query)

        progress.update(task, description="✅ Research complete!")

    elapsed = time.time() - start_time

    # ── Display stats ─────────────────────────────────────────────────────────
    stats = Table(box=box.SIMPLE)
    stats.add_column("Metric", style="cyan")
    stats.add_column("Value", style="green")
    stats.add_row("⏱  Time elapsed", f"{elapsed:.1f}s")
    stats.add_row("🔄 Agent steps", str(result.get("steps", "?")))
    stats.add_row("📝 Notes collected", str(len(result.get("notes", []))))
    console.print(stats)

    # ── Display report ────────────────────────────────────────────────────────
    report = result.get("report", "No report generated.")

    console.print("\n")
    console.print(Panel(
        Markdown(report),
        title="[bold green]📄 Research Report[/bold green]",
        border_style="green",
        padding=(1, 2),
    ))

    # ── Save to file ──────────────────────────────────────────────────────────
    if output_file:
        save_path = Path(output_file)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_query = "".join(c if c.isalnum() or c in " _-" else "_" for c in query[:40]).strip()
        save_path = Path(f"reports/{safe_query}_{timestamp}.md")

    save_path.parent.mkdir(parents=True, exist_ok=True)

    report_content = (
        f"# Research Report\n\n"
        f"**Query:** {query}\n"
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"**Agent Steps:** {result.get('steps', '?')}\n\n"
        f"---\n\n{report}"
    )
    save_path.write_text(report_content, encoding="utf-8")
    console.print(f"\n[bold green]✅ Report saved to:[/bold green] {save_path}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Agentic AI Research Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  python main.py --query \"AI trends 2024\"\n  python main.py  (interactive mode)"
    )
    parser.add_argument("--query", "-q", type=str, help="Research query to investigate")
    parser.add_argument("--output", "-o", type=str, help="Output file path for the report (default: auto-named .md)")
    parser.add_argument("--examples", action="store_true", help="Show example queries and exit")
    args = parser.parse_args()

    console.print(f"[bold cyan]{BANNER}[/bold cyan]")

    if args.examples:
        show_examples()
        return

    if not validate_env():
        sys.exit(1)

    show_examples()

    query = args.query or get_query_interactive()

    if not query:
        console.print("[red]No query provided. Exiting.[/red]")
        sys.exit(1)

    run_research(query, output_file=args.output)


if __name__ == "__main__":
    main()
