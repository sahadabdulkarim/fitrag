"""State schema for the FitRAG multi-agent workflow."""

from typing import TypedDict, Annotated
from dataclasses import dataclass, field


class ParsedQuery(TypedDict, total=False):
    """Structured representation of user's fitness query."""

    goal: str  # e.g., "muscle_gain", "fat_loss", "injury_rehab", "general_info"
    specific_question: str  # The core question being asked
    dietary_restrictions: list[str]  # e.g., ["vegan", "gluten-free"]
    injuries: list[str]  # e.g., ["knee", "shoulder"]
    training_frequency: int | None  # Days per week
    experience_level: str  # "beginner", "intermediate", "advanced"
    body_stats: dict  # e.g., {"weight_kg": 80, "height_cm": 180, "age": 28}
    urgency: str  # "low", "medium", "high"


class RetrievedContext(TypedDict):
    """A single retrieved document chunk with metadata."""

    content: str
    source: str
    page: int
    relevance_score: float


class SafetyFlag(TypedDict):
    """A safety warning or contraindication."""

    concern: str  # What the safety issue is
    severity: str  # "info", "warning", "critical"
    recommendation: str  # What to do about it


class FitRAGState(TypedDict, total=False):
    """Complete state passed through the LangGraph workflow.

    This state is shared across all agents and accumulated
    as the graph executes.
    """

    # Input
    raw_input: str  # Original user message
    
    # Intake Agent output
    parsed_query: ParsedQuery
    
    # Safety Agent output
    has_injury: bool
    safety_flags: list[SafetyFlag]
    injury_context: list[RetrievedContext]  # Retrieved injury-specific docs
    
    # Retrieval Agent output
    retrieved_docs: list[RetrievedContext]
    retrieval_context: str  # Formatted context string for LLM
    
    # Recommendation Agent output
    recommendation: str  # Final generated answer
    sources_cited: list[str]  # List of source documents referenced
    
    # Metadata
    workflow_path: list[str]  # Track which nodes were executed
    error: str | None  # Error message if something failed
