"""Interactive query interface for the FitRAG system with hybrid retrieval."""

import os
import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

from app.rag.embedder import EmbeddingPipeline
from app.rag.retriever import DenseRetriever
from app.rag.bm25_retriever import BM25Retriever

# Load environment variables
load_dotenv()

console = Console()


def display_dense_results(query: str, results):
    """Display dense retrieval results."""
    console.print(f"\n[bold cyan]Query:[/bold cyan] {query}")
    console.print(f"[dim]Mode: Dense (FAISS) | Top {len(results)} results:[/dim]\n")

    for i, result in enumerate(results, 1):
        source = result.metadata.get("source_file", "unknown")
        page = result.metadata.get("page", "?")
        score = result.score
        content = result.content[:250] + ("..." if len(result.content) > 250 else "")
        panel = Panel(
            content,
            title=f"[bold green]#{i}[/bold green] [yellow]{source}[/yellow] p.{page} | L2: {score:.4f}",
            border_style="blue",
            padding=(0, 1),
        )
        console.print(panel)


def display_hybrid_results(query: str, results):
    """Display hybrid retrieval results."""
    console.print(f"\n[bold cyan]Query:[/bold cyan] {query}")
    console.print(f"[dim]Mode: Hybrid (Dense + BM25 + RRF + Reranker) | Top {len(results)} results:[/dim]\n")

    for i, result in enumerate(results, 1):
        source = result.metadata.get("source_file", "unknown")
        page = result.metadata.get("page", "?")
        score_str = f"Rerank: {result.rerank_score:.4f}" if result.rerank_score is not None else f"RRF: {result.rrf_score:.6f}"
        content = result.content[:250] + ("..." if len(result.content) > 250 else "")
        panel = Panel(
            content,
            title=f"[bold green]#{i}[/bold green] [yellow]{source}[/yellow] p.{page} | {score_str}",
            border_style="magenta",
            padding=(0, 1),
        )
        console.print(panel)


def generate_answer_with_context(query: str, context: str):
    """Generate answer via Groq LLM."""
    try:
        from app.rag.generator import GroqGenerator

        console.print("\n[bold magenta]═══ 🤖 Generated Answer (Groq Llama 3.3 70B) ═══[/bold magenta]\n")
        generator = GroqGenerator()

        full_response = ""
        for token in generator.generate_streaming(query, context):
            console.print(token, end="")
            full_response += token
        console.print("\n")

    except ValueError as e:
        console.print(f"\n[yellow]⚠️  {e}[/yellow]")
        console.print("[dim]Showing context only.[/dim]\n")
        console.print(Panel(context[:2000], title="Retrieved Context", border_style="magenta"))
    except Exception as e:
        console.print(f"\n[red]❌ LLM Error: {e}[/red]")
        console.print("[dim]Showing context only.[/dim]\n")
        console.print(Panel(context[:2000], title="Retrieved Context", border_style="magenta"))


def query_dense(query: str, embedder: EmbeddingPipeline, use_llm: bool = True):
    """Query using dense retrieval only."""
    retriever = DenseRetriever(embedding_pipeline=embedder)
    results = retriever.retrieve(query)

    if not results:
        console.print("[red]No results found.[/red]")
        return

    display_dense_results(query, results)
    context = retriever.format_context(results)

    if use_llm:
        generate_answer_with_context(query, context)


def query_hybrid(query: str, embedder: EmbeddingPipeline, bm25: BM25Retriever, use_llm: bool = True):
    """Query using hybrid retrieval (Dense + BM25 + RRF + Reranker)."""
    from app.rag.hybrid_retriever import HybridRetriever

    hybrid = HybridRetriever(
        embedding_pipeline=embedder,
        bm25_retriever=bm25,
        use_reranker=True,
    )

    results = hybrid.retrieve(query, top_k=5)

    if not results:
        console.print("[red]No results found.[/red]")
        return

    display_hybrid_results(query, results)
    context = hybrid.format_context(results)

    if use_llm:
        generate_answer_with_context(query, context)


def interactive_mode():
    """Run the interactive query loop."""
    console.print(Panel.fit(
        "[bold green]🏋️ FitRAG - Fitness & Nutrition Intelligence Agent[/bold green]\n\n"
        "Ask questions about fitness, nutrition, training, and recovery.\n"
        "Powered by: Hybrid RAG (FAISS + BM25 + Reranker) + Groq Llama 3.3 70B\n\n"
        "[dim]Commands:[/dim]\n"
        "  [cyan]quit[/cyan]   — exit\n"
        "  [cyan]help[/cyan]   — show example queries\n"
        "  [cyan]mode[/cyan]   — toggle between dense / hybrid\n"
        "  [cyan]noai[/cyan]   — toggle LLM generation on/off",
        border_style="green",
    ))

    # Check Groq key
    use_llm = bool(os.getenv("GROQ_API_KEY"))
    if not use_llm:
        console.print("[yellow]⚠️  GROQ_API_KEY not set. Retrieval-only mode.[/yellow]\n")
    else:
        console.print("[green]✅ Groq API key found.[/green]")

    # Load indices
    console.print("[dim]Loading FAISS index...[/dim]")
    embedder = EmbeddingPipeline()
    embedder.load_index()

    console.print("[dim]Loading BM25 index...[/dim]")
    bm25 = BM25Retriever()
    try:
        bm25.load_index()
        hybrid_available = True
    except FileNotFoundError:
        console.print("[yellow]⚠️  BM25 index not found. Run 'python -m app.pipeline' to build.[/yellow]")
        hybrid_available = False

    retrieval_mode = "hybrid" if hybrid_available else "dense"
    console.print(f"[green]✅ Ready! Mode: {retrieval_mode}[/green]\n")

    example_queries = [
        "How much protein do I need per day for muscle growth?",
        "What is the optimal training volume for hypertrophy?",
        "How does training frequency affect muscle growth?",
        "What are the best strategies to prevent sports injuries?",
        "What are the Australian dietary guidelines for adults?",
    ]

    while True:
        try:
            query = console.input(f"\n[bold cyan]❓ [{retrieval_mode}][/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye! 💪[/dim]")
            break

        if not query:
            continue

        if query.lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye! 💪[/dim]")
            break

        if query.lower() == "help":
            console.print("\n[bold]Example queries:[/bold]")
            for i, eq in enumerate(example_queries, 1):
                console.print(f"  [cyan]{i}.[/cyan] {eq}")
            continue

        if query.lower() == "mode":
            if not hybrid_available:
                console.print("[yellow]Hybrid not available (no BM25 index).[/yellow]")
                continue
            retrieval_mode = "hybrid" if retrieval_mode == "dense" else "dense"
            console.print(f"[yellow]Switched to: {retrieval_mode}[/yellow]")
            continue

        if query.lower() == "noai":
            use_llm = not use_llm
            console.print(f"[yellow]LLM generation: {'ON' if use_llm else 'OFF'}[/yellow]")
            continue

        if retrieval_mode == "hybrid":
            query_hybrid(query, embedder, bm25, use_llm=use_llm)
        else:
            query_dense(query, embedder, use_llm=use_llm)


def single_query(query: str, mode: str = "hybrid"):
    """Run a single query."""
    embedder = EmbeddingPipeline()
    embedder.load_index()

    use_llm = bool(os.getenv("GROQ_API_KEY"))

    if mode == "hybrid":
        bm25 = BM25Retriever()
        try:
            bm25.load_index()
            query_hybrid(query, embedder, bm25, use_llm=use_llm)
        except FileNotFoundError:
            console.print("[yellow]BM25 index not found, falling back to dense.[/yellow]")
            query_dense(query, embedder, use_llm=use_llm)
    else:
        query_dense(query, embedder, use_llm=use_llm)


if __name__ == "__main__":
    args = sys.argv[1:]

    mode = "hybrid"
    query_parts = []

    for arg in args:
        if arg in ("--dense", "-d"):
            mode = "dense"
        elif arg in ("--hybrid", "-h"):
            mode = "hybrid"
        else:
            query_parts.append(arg)

    if query_parts:
        single_query(" ".join(query_parts), mode=mode)
    else:
        interactive_mode()
