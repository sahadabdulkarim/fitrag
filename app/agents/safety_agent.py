"""Safety Agent: Detects injuries and retrieves safety protocols."""

from app.agents.state import FitRAGState, SafetyFlag, RetrievedContext
from app.rag.embedder import EmbeddingPipeline
from app.rag.retriever import DenseRetriever


def safety_agent(state: FitRAGState) -> FitRAGState:
    """Check for injury-related concerns and retrieve safety protocols.

    This agent is ONLY called when the user mentions injuries or pain.
    It retrieves relevant injury prevention/rehabilitation information
    and creates safety flags.
    """
    parsed_query = state["parsed_query"]
    injuries = parsed_query.get("injuries", [])

    if not injuries:
        return {
            **state,
            "safety_flags": [],
            "injury_context": [],
            "workflow_path": state.get("workflow_path", []) + ["safety_agent (skipped)"],
        }

    # Build safety query focused on injuries
    injury_query = f"injury prevention rehabilitation exercises for {', '.join(injuries)} pain safe training modifications contraindications"

    # Retrieve injury-specific documents
    embedder = EmbeddingPipeline()
    embedder.load_index()
    retriever = DenseRetriever(embedding_pipeline=embedder)
    results = retriever.retrieve(injury_query, top_k=3)

    # Build safety flags
    safety_flags: list[SafetyFlag] = []
    injury_context: list[RetrievedContext] = []

    for injury in injuries:
        safety_flags.append({
            "concern": f"User reports {injury} injury/pain",
            "severity": "warning",
            "recommendation": f"Avoid exercises that aggravate {injury}. Consider modifications and consult a physiotherapist for persistent pain.",
        })

    for result in results:
        injury_context.append({
            "content": result.content,
            "source": result.metadata.get("source_file", "unknown"),
            "page": result.metadata.get("page", 0),
            "relevance_score": result.score,
        })

    return {
        **state,
        "safety_flags": safety_flags,
        "injury_context": injury_context,
        "workflow_path": state.get("workflow_path", []) + ["safety_agent"],
    }
