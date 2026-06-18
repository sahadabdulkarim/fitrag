"""CRUD operations for the FitRAG database."""

from sqlalchemy.orm import Session

from app.db.models import QueryRecord, RetrievalRecord


def save_query(db: Session, state: dict) -> QueryRecord:
    """Save a completed workflow state to the database."""
    parsed = state.get("parsed_query", {})

    record = QueryRecord(
        raw_input=state.get("raw_input", ""),
        goal=parsed.get("goal"),
        specific_question=parsed.get("specific_question"),
        experience_level=parsed.get("experience_level"),
        dietary_restrictions=parsed.get("dietary_restrictions", []),
        injuries=parsed.get("injuries", []),
        training_frequency=parsed.get("training_frequency"),
        body_stats=parsed.get("body_stats", {}),
        has_injury=state.get("has_injury", False),
        safety_flags=state.get("safety_flags", []),
        num_docs_retrieved=len(state.get("retrieved_docs", [])),
        sources_cited=state.get("sources_cited", []),
        recommendation=state.get("recommendation", ""),
        workflow_path=state.get("workflow_path", []),
        error=state.get("error"),
    )

    db.add(record)
    db.flush()

    # Save individual retrieval records
    for doc in state.get("retrieved_docs", []):
        retrieval = RetrievalRecord(
            query_id=record.id,
            chunk_content=doc.get("content", ""),
            source_file=doc.get("source", ""),
            page=doc.get("page", 0),
            relevance_score=doc.get("relevance_score", 0.0),
        )
        db.add(retrieval)

    db.commit()
    db.refresh(record)
    return record


def get_all_queries(db: Session, limit: int = 50, offset: int = 0) -> list[QueryRecord]:
    """Get all query records (most recent first)."""
    return (
        db.query(QueryRecord)
        .order_by(QueryRecord.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )


def get_query_by_id(db: Session, query_id: int) -> QueryRecord | None:
    """Get a specific query record by ID."""
    return db.query(QueryRecord).filter(QueryRecord.id == query_id).first()


def get_retrievals_for_query(db: Session, query_id: int) -> list[RetrievalRecord]:
    """Get all retrieved chunks for a specific query."""
    return (
        db.query(RetrievalRecord)
        .filter(RetrievalRecord.query_id == query_id)
        .all()
    )


def get_query_count(db: Session) -> int:
    """Get total number of queries."""
    return db.query(QueryRecord).count()
