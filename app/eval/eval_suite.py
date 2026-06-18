"""RAGAs-style evaluation suite for measuring retrieval and generation quality."""

import json
import time
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from app.rag.embedder import EmbeddingPipeline
from app.rag.retriever import DenseRetriever
from app.rag.bm25_retriever import BM25Retriever

console = Console()

# Test cases: question + expected keywords that should appear in retrieved context
EVAL_CASES = [
    {
        "question": "How much protein should an athlete consume daily?",
        "expected_keywords": ["1.4", "2.0", "g/kg", "protein", "athlete"],
        "expected_source": "s12970-017-0177-8.pdf",
        "category": "nutrition",
    },
    {
        "question": "What is the dose-response relationship between training volume and muscle growth?",
        "expected_keywords": ["volume", "hypertrophy", "sets", "muscle"],
        "expected_source": "Dose response",
        "category": "training",
    },
    {
        "question": "How does resistance training frequency affect hypertrophy?",
        "expected_keywords": ["frequency", "hypertrophy", "training", "muscle"],
        "expected_source": "Effects of Resistance Training Frequency",
        "category": "training",
    },
    {
        "question": "What exercises help prevent sports injuries?",
        "expected_keywords": ["injury", "prevention", "exercise", "strength"],
        "expected_source": "Lauersen",
        "category": "injury",
    },
    {
        "question": "What are the dietary guidelines for fruit and vegetable intake?",
        "expected_keywords": ["fruit", "vegetable", "diet", "intake"],
        "expected_source": "",  # Could be WHO or Australian guidelines
        "category": "nutrition",
    },
    {
        "question": "What is the recommended daily protein intake per serving to maximize muscle protein synthesis?",
        "expected_keywords": ["20", "40", "g", "protein", "synthesis", "MPS"],
        "expected_source": "s12970-017-0177-8.pdf",
        "category": "nutrition",
    },
    {
        "question": "How many sets per muscle group per week are needed for maximum hypertrophy?",
        "expected_keywords": ["sets", "volume", "week", "hypertrophy"],
        "expected_source": "Dose response",
        "category": "training",
    },
    {
        "question": "What role does body composition play in determining protein needs?",
        "expected_keywords": ["body", "composition", "protein", "fat", "mass"],
        "expected_source": "s12970-017-0174-y.pdf",
        "category": "nutrition",
    },
]


@dataclass
class EvalResult:
    """Result for a single evaluation case."""

    question: str
    category: str
    # Retrieval metrics
    keyword_recall: float  # % of expected keywords found in retrieved context
    source_hit: bool  # Was the expected source in the top-k?
    retrieval_latency_ms: float
    # Quality metrics
    context_length: int  # Total chars of retrieved context
    num_unique_sources: int  # Number of distinct source documents


@dataclass
class EvalSummary:
    """Summary across all evaluation cases."""

    mode: str
    total_cases: int
    avg_keyword_recall: float
    source_hit_rate: float
    avg_latency_ms: float
    results: list[EvalResult] = field(default_factory=list)


def compute_keyword_recall(context: str, expected_keywords: list[str]) -> float:
    """Compute what fraction of expected keywords appear in the context."""
    if not expected_keywords:
        return 1.0
    context_lower = context.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in context_lower)
    return hits / len(expected_keywords)


def check_source_hit(results, expected_source: str) -> bool:
    """Check if expected source document was retrieved."""
    if not expected_source:
        return True  # No specific source expected
    for result in results:
        source = result.metadata.get("source_file", "") if hasattr(result, "metadata") else ""
        if expected_source.lower() in source.lower():
            return True
    return False


def evaluate_dense(embedder: EmbeddingPipeline, top_k: int = 5) -> EvalSummary:
    """Evaluate dense retrieval on all test cases."""
    retriever = DenseRetriever(embedding_pipeline=embedder)
    results_list: list[EvalResult] = []

    for case in EVAL_CASES:
        start = time.time()
        results = retriever.retrieve(case["question"], top_k=top_k)
        latency = (time.time() - start) * 1000

        context = " ".join(r.content for r in results)
        keyword_recall = compute_keyword_recall(context, case["expected_keywords"])
        source_hit = check_source_hit(results, case["expected_source"])
        unique_sources = len(set(r.metadata.get("source_file", "") for r in results))

        results_list.append(EvalResult(
            question=case["question"],
            category=case["category"],
            keyword_recall=keyword_recall,
            source_hit=source_hit,
            retrieval_latency_ms=latency,
            context_length=len(context),
            num_unique_sources=unique_sources,
        ))

    return EvalSummary(
        mode="dense",
        total_cases=len(results_list),
        avg_keyword_recall=np.mean([r.keyword_recall for r in results_list]),
        source_hit_rate=np.mean([1.0 if r.source_hit else 0.0 for r in results_list]),
        avg_latency_ms=np.mean([r.retrieval_latency_ms for r in results_list]),
        results=results_list,
    )


def evaluate_hybrid(
    embedder: EmbeddingPipeline,
    bm25: BM25Retriever,
    use_reranker: bool = True,
    top_k: int = 5,
) -> EvalSummary:
    """Evaluate hybrid retrieval on all test cases."""
    from app.rag.hybrid_retriever import HybridRetriever

    hybrid = HybridRetriever(
        embedding_pipeline=embedder,
        bm25_retriever=bm25,
        use_reranker=use_reranker,
    )

    results_list: list[EvalResult] = []

    for case in EVAL_CASES:
        start = time.time()
        results = hybrid.retrieve(case["question"], top_k=top_k)
        latency = (time.time() - start) * 1000

        context = " ".join(r.content for r in results)
        keyword_recall = compute_keyword_recall(context, case["expected_keywords"])
        source_hit = any(
            case["expected_source"].lower() in r.metadata.get("source_file", "").lower()
            for r in results
        ) if case["expected_source"] else True
        unique_sources = len(set(r.metadata.get("source_file", "") for r in results))

        results_list.append(EvalResult(
            question=case["question"],
            category=case["category"],
            keyword_recall=keyword_recall,
            source_hit=source_hit,
            retrieval_latency_ms=latency,
            context_length=len(context),
            num_unique_sources=unique_sources,
        ))

    return EvalSummary(
        mode="hybrid" + (" + reranker" if use_reranker else " (no reranker)"),
        total_cases=len(results_list),
        avg_keyword_recall=np.mean([r.keyword_recall for r in results_list]),
        source_hit_rate=np.mean([1.0 if r.source_hit else 0.0 for r in results_list]),
        avg_latency_ms=np.mean([r.retrieval_latency_ms for r in results_list]),
        results=results_list,
    )


def display_results(summary: EvalSummary):
    """Display evaluation results in a table."""
    console.print(f"\n[bold]{'=' * 60}[/bold]")
    console.print(f"[bold cyan]📊 Evaluation: {summary.mode.upper()}[/bold cyan]")
    console.print(f"[bold]{'=' * 60}[/bold]\n")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Question", style="dim", width=50)
    table.add_column("Keyword\nRecall", justify="center")
    table.add_column("Source\nHit", justify="center")
    table.add_column("Latency\n(ms)", justify="center")
    table.add_column("Sources", justify="center")

    for r in summary.results:
        recall_color = "green" if r.keyword_recall >= 0.8 else "yellow" if r.keyword_recall >= 0.6 else "red"
        source_color = "green" if r.source_hit else "red"

        table.add_row(
            r.question[:50],
            f"[{recall_color}]{r.keyword_recall:.0%}[/{recall_color}]",
            f"[{source_color}]{'✓' if r.source_hit else '✗'}[/{source_color}]",
            f"{r.retrieval_latency_ms:.1f}",
            str(r.num_unique_sources),
        )

    console.print(table)

    # Summary metrics
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Average Keyword Recall: [{'green' if summary.avg_keyword_recall >= 0.8 else 'yellow'}]{summary.avg_keyword_recall:.1%}[/]")
    console.print(f"  Source Hit Rate:        [{'green' if summary.source_hit_rate >= 0.8 else 'yellow'}]{summary.source_hit_rate:.1%}[/]")
    console.print(f"  Average Latency:        {summary.avg_latency_ms:.1f} ms")


def run_comparison():
    """Run evaluation comparing dense vs hybrid retrieval."""
    console.print(Panel.fit(
        "[bold green]🏋️ FitRAG - Retrieval Evaluation Suite[/bold green]\n\n"
        f"Running {len(EVAL_CASES)} test cases across dense and hybrid retrieval.",
        border_style="green",
    ))

    # Load indices
    console.print("\n[dim]Loading indices...[/dim]")
    embedder = EmbeddingPipeline()
    embedder.load_index()

    bm25 = BM25Retriever()
    bm25.load_index()

    # Evaluate Dense
    console.print("\n[dim]Evaluating dense retrieval...[/dim]")
    dense_summary = evaluate_dense(embedder, top_k=5)
    display_results(dense_summary)

    # Evaluate Hybrid (no reranker for speed first)
    console.print("\n[dim]Evaluating hybrid (RRF only, no reranker)...[/dim]")
    hybrid_no_rerank = evaluate_hybrid(embedder, bm25, use_reranker=False, top_k=5)
    display_results(hybrid_no_rerank)

    # Evaluate Hybrid with Reranker
    console.print("\n[dim]Evaluating hybrid + reranker (this may take a moment)...[/dim]")
    hybrid_rerank = evaluate_hybrid(embedder, bm25, use_reranker=True, top_k=5)
    display_results(hybrid_rerank)

    # Final comparison
    console.print(f"\n{'=' * 60}")
    console.print("[bold cyan]📈 COMPARISON SUMMARY[/bold cyan]")
    console.print(f"{'=' * 60}\n")

    comp_table = Table(show_header=True, header_style="bold")
    comp_table.add_column("Method", width=25)
    comp_table.add_column("Keyword Recall", justify="center")
    comp_table.add_column("Source Hit Rate", justify="center")
    comp_table.add_column("Avg Latency", justify="center")

    for s in [dense_summary, hybrid_no_rerank, hybrid_rerank]:
        comp_table.add_row(
            s.mode,
            f"{s.avg_keyword_recall:.1%}",
            f"{s.source_hit_rate:.1%}",
            f"{s.avg_latency_ms:.0f} ms",
        )

    console.print(comp_table)

    # Improvement calculation
    if dense_summary.avg_keyword_recall > 0:
        improvement = ((hybrid_rerank.avg_keyword_recall - dense_summary.avg_keyword_recall) / dense_summary.avg_keyword_recall) * 100
        console.print(f"\n[bold]Hybrid + Reranker improvement over Dense:[/bold]")
        console.print(f"  Keyword Recall: {'+' if improvement >= 0 else ''}{improvement:.1f}%")


if __name__ == "__main__":
    run_comparison()
