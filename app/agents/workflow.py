"""LangGraph workflow definition for the FitRAG multi-agent system."""

from langgraph.graph import StateGraph, END

from app.agents.state import FitRAGState
from app.agents.intake_agent import intake_agent
from app.agents.safety_agent import safety_agent
from app.agents.retrieval_agent import retrieval_agent
from app.agents.recommendation_agent import recommendation_agent


def route_after_intake(state: FitRAGState) -> str:
    """Conditional routing: if user has injury, go to safety agent first."""
    if state.get("has_injury", False):
        return "safety_agent"
    return "retrieval_agent"


def build_workflow() -> StateGraph:
    """Build the FitRAG multi-agent workflow graph.

    Graph Structure:
    ┌─────────────┐
    │ intake_agent │  Parse user input → structured JSON
    └──────┬──────┘
           │
           ▼ (conditional)
    ┌──────────────────┐     ┌─────────────────┐
    │ has_injury=True? │─Yes─▶│  safety_agent   │
    └────────┬─────────┘     └────────┬────────┘
             │ No                      │
             ▼                         ▼
    ┌──────────────────┐
    │ retrieval_agent   │  Hybrid search (Dense + BM25)
    └────────┬─────────┘
             │
             ▼
    ┌───────────────────────┐
    │ recommendation_agent  │  Generate final grounded answer
    └───────────────────────┘
             │
             ▼
           [END]
    """
    # Create the graph with our state schema
    workflow = StateGraph(FitRAGState)

    # Add nodes (each node is an agent function)
    workflow.add_node("intake_agent", intake_agent)
    workflow.add_node("safety_agent", safety_agent)
    workflow.add_node("retrieval_agent", retrieval_agent)
    workflow.add_node("recommendation_agent", recommendation_agent)

    # Set entry point
    workflow.set_entry_point("intake_agent")

    # Add conditional edge after intake (branching logic)
    workflow.add_conditional_edges(
        "intake_agent",
        route_after_intake,
        {
            "safety_agent": "safety_agent",
            "retrieval_agent": "retrieval_agent",
        },
    )

    # Safety agent always leads to retrieval
    workflow.add_edge("safety_agent", "retrieval_agent")

    # Retrieval always leads to recommendation
    workflow.add_edge("retrieval_agent", "recommendation_agent")

    # Recommendation is the final step
    workflow.add_edge("recommendation_agent", END)

    return workflow


def create_app():
    """Compile and return the runnable workflow."""
    workflow = build_workflow()
    return workflow.compile()


# Pre-built app instance
fitrag_app = None


def get_app():
    """Get or create the compiled workflow app."""
    global fitrag_app
    if fitrag_app is None:
        fitrag_app = create_app()
    return fitrag_app


def run_query(user_input: str) -> FitRAGState:
    """Run a user query through the full multi-agent workflow.

    Args:
        user_input: Natural language fitness/nutrition question

    Returns:
        Final state with recommendation, sources, safety flags, etc.
    """
    app = get_app()

    initial_state: FitRAGState = {
        "raw_input": user_input,
        "workflow_path": [],
    }

    # Execute the graph
    final_state = app.invoke(initial_state)
    return final_state
