"""Retrieval Agent: Performs hybrid search based on parsed query."""

from app.agents.state import FitRAGState, RetrievedContext
from app.rag.embedder import EmbeddingPipeline
from app.rag.bm25_retriever import BM25Retriever
from app.rag.hybrid_retriever import HybridRetriever


def retrieval_agent(state: FitRAGState) -> FitRAGState:
    """Retrieve relevant documents using hybrid search.

    Uses the parsed query to build an optimized search query,
    then retrieves using Dense + BM25 + RRF (no reranker for speed
    in the agent pipeline).
    """
    parsed_query = state["parsed_query"]

    # Build an optimized search query from parsed fields
    query_parts = [parsed_query.get("specific_question", state["raw_input"])]

    # Add context from parsed fields to improve retrieval
    if parsed_query.get("goal"):
        goal_map = {
            "muscle_gain": "hypertrophy muscle building",
            "fat_loss": "weight loss caloric deficit",
            "strength": "strength training progressive overload",
            "injury_rehab": "rehabilitation recovery",
            "endurance": "cardiovascular endurance training",
            "flexibility": "mobility stretching flexibility",
        }
        if parsed_query["goal"] in goal_map:
            query_parts.append(goal_map[parsed_query["goal"]])

    if parsed_query.get("dietary_restrictions"):
        query_parts.append(" ".join(parsed_query["dietary_restrictions"]) + " diet nutrition")

    search_query = " ".join(query_parts)

    # Perform hybrid retrieval
    embedder = EmbeddingPipeline()
    embedder.load_index()

    bm25 = BM25Retriever()
    try:
        bm25.load_index()
    except FileNotFoundError:
        # Fall back to dense-only if BM25 index not available
        from app.rag.retriever import DenseRetriever
        retriever = DenseRetriever(embedding_pipeline=embedder)
        results = retriever.retrieve(search_query, top_k=5)

        retrieved_docs = [
            {
                "content": r.content,
                "source": r.metadata.get("source_file", "unknown"),
                "page": r.metadata.get("page", 0),
                "relevance_score": r.score,
            }
            for r in results
        ]

        context = retriever.format_context(results)

        return {
            **state,
            "retrieved_docs": retrieved_docs,
            "retrieval_context": context,
            "workflow_path": state.get("workflow_path", []) + ["retrieval_agent (dense fallback)"],
        }

    # Hybrid retrieval (no reranker for speed in agent pipeline)
    hybrid = HybridRetriever(
        embedding_pipeline=embedder,
        bm25_retriever=bm25,
        use_reranker=False,  # Skip reranker for speed in multi-agent flow
    )

    results = hybrid.retrieve(search_query, top_k=5)

    retrieved_docs: list[RetrievedContext] = [
        {
            "content": r.content,
            "source": r.metadata.get("source_file", "unknown"),
            "page": r.metadata.get("page", 0),
            "relevance_score": r.rrf_score,
        }
        for r in results
    ]

    context = hybrid.format_context(results)

    return {
        **state,
        "retrieved_docs": retrieved_docs,
        "retrieval_context": context,
        "workflow_path": state.get("workflow_path", []) + ["retrieval_agent"],
    }
