"""FastAPI application for the FitRAG system."""

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.models import init_db, SessionLocal, QueryRecord
from app.db.crud import save_query, get_all_queries, get_query_by_id, get_retrievals_for_query, get_query_count

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup."""
    init_db()
    yield


app = FastAPI(
    title="FitRAG API",
    description="Personalized Fitness & Nutrition Intelligence Agent — powered by multi-agent RAG",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers (create-react-app & vite)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic Schemas ---

class QueryRequest(BaseModel):
    """Request body for submitting a fitness query."""
    question: str

    model_config = {"json_schema_extra": {
        "examples": [
            {"question": "How much protein do I need to build muscle?"},
            {"question": "I'm 28, vegan, bad knee, want to build muscle 4x/week"},
        ]
    }}


class SafetyFlagResponse(BaseModel):
    concern: str
    severity: str
    recommendation: str


class QueryResponse(BaseModel):
    """Response from the FitRAG multi-agent system."""
    id: int
    question: str
    goal: str | None
    parsed_question: str | None
    has_injury: bool
    safety_flags: list[SafetyFlagResponse]
    recommendation: str
    sources_cited: list[str]
    workflow_path: list[str]


class QueryListItem(BaseModel):
    """Summary item for query list."""
    id: int
    question: str
    goal: str | None
    has_injury: bool
    created_at: str


class QueryDetailResponse(BaseModel):
    """Detailed query response with retrievals."""
    id: int
    question: str
    goal: str | None
    parsed_question: str | None
    experience_level: str | None
    dietary_restrictions: list[str]
    injuries: list[str]
    training_frequency: int | None
    has_injury: bool
    safety_flags: list[SafetyFlagResponse]
    recommendation: str
    sources_cited: list[str]
    workflow_path: list[str]
    num_docs_retrieved: int
    created_at: str
    retrieved_chunks: list[dict]


class StatsResponse(BaseModel):
    """System statistics."""
    total_queries: int
    status: str


# --- Dependency ---

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --- Routes ---

@app.get("/", tags=["health"])
def root():
    """Health check endpoint."""
    return {
        "service": "FitRAG API",
        "status": "healthy",
        "version": "1.0.0",
        "description": "Multi-agent RAG system for fitness & nutrition intelligence",
    }


@app.get("/stats", response_model=StatsResponse, tags=["admin"])
def get_stats(db: Session = Depends(get_db)):
    """Get system statistics."""
    return StatsResponse(
        total_queries=get_query_count(db),
        status="operational",
    )


@app.post("/query", response_model=QueryResponse, tags=["queries"])
def submit_query(request: QueryRequest, db: Session = Depends(get_db)):
    """Submit a fitness/nutrition question to the multi-agent RAG system.

    The system will:
    1. Parse your question (Intake Agent)
    2. Check for injury concerns (Safety Agent — if applicable)
    3. Retrieve relevant research (Retrieval Agent)
    4. Generate a grounded recommendation (Recommendation Agent)
    """
    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")

    try:
        from app.agents.workflow import run_query

        # Run the multi-agent workflow
        final_state = run_query(request.question)

        # Save to database
        record = save_query(db, final_state)

        # Build response
        parsed = final_state.get("parsed_query", {})
        safety_flags = [
            SafetyFlagResponse(**f) for f in final_state.get("safety_flags", [])
        ]

        return QueryResponse(
            id=record.id,
            question=request.question,
            goal=parsed.get("goal"),
            parsed_question=parsed.get("specific_question"),
            has_injury=final_state.get("has_injury", False),
            safety_flags=safety_flags,
            recommendation=final_state.get("recommendation", ""),
            sources_cited=final_state.get("sources_cited", []),
            workflow_path=final_state.get("workflow_path", []),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Workflow error: {str(e)}")


@app.get("/queries", response_model=list[QueryListItem], tags=["queries"])
def list_queries(limit: int = 20, offset: int = 0, db: Session = Depends(get_db)):
    """List all past queries (most recent first)."""
    records = get_all_queries(db, limit=limit, offset=offset)
    return [
        QueryListItem(
            id=r.id,
            question=r.raw_input,
            goal=r.goal,
            has_injury=r.has_injury,
            created_at=r.created_at.isoformat() if r.created_at else "",
        )
        for r in records
    ]


@app.get("/queries/{query_id}", response_model=QueryDetailResponse, tags=["queries"])
def get_query_detail(query_id: int, db: Session = Depends(get_db)):
    """Get detailed information about a specific query, including retrieved chunks."""
    record = get_query_by_id(db, query_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Query {query_id} not found")

    retrievals = get_retrievals_for_query(db, query_id)
    chunks = [
        {
            "content": r.chunk_content[:500],
            "source": r.source_file,
            "page": r.page,
            "score": r.relevance_score,
        }
        for r in retrievals
    ]

    safety_flags = [
        SafetyFlagResponse(**f) for f in (record.safety_flags or [])
    ]

    return QueryDetailResponse(
        id=record.id,
        question=record.raw_input,
        goal=record.goal,
        parsed_question=record.specific_question,
        experience_level=record.experience_level,
        dietary_restrictions=record.dietary_restrictions or [],
        injuries=record.injuries or [],
        training_frequency=record.training_frequency,
        has_injury=record.has_injury,
        safety_flags=safety_flags,
        recommendation=record.recommendation or "",
        sources_cited=record.sources_cited or [],
        workflow_path=record.workflow_path or [],
        num_docs_retrieved=record.num_docs_retrieved or 0,
        created_at=record.created_at.isoformat() if record.created_at else "",
        retrieved_chunks=chunks,
    )
