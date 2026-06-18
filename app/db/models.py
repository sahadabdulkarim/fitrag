"""SQLAlchemy database models for the FitRAG audit trail."""

import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime, JSON, Boolean
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.core.config import PROCESSED_DATA_DIR

# SQLite database (simple, no Docker needed)
DATABASE_URL = f"sqlite:///{PROCESSED_DATA_DIR / 'fitrag.db'}"

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)


class Base(DeclarativeBase):
    pass


class QueryRecord(Base):
    """Stores every query processed by the system."""

    __tablename__ = "queries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Input
    raw_input = Column(Text, nullable=False)

    # Parsed query (from intake agent)
    goal = Column(String(50))
    specific_question = Column(Text)
    experience_level = Column(String(20))
    dietary_restrictions = Column(JSON)  # list[str]
    injuries = Column(JSON)  # list[str]
    training_frequency = Column(Integer, nullable=True)
    body_stats = Column(JSON)  # dict

    # Safety
    has_injury = Column(Boolean, default=False)
    safety_flags = Column(JSON)  # list[dict]

    # Retrieval
    num_docs_retrieved = Column(Integer, default=0)
    sources_cited = Column(JSON)  # list[str]

    # Output
    recommendation = Column(Text)
    workflow_path = Column(JSON)  # list[str]

    # Metadata
    error = Column(Text, nullable=True)


class RetrievalRecord(Base):
    """Stores individual retrieved chunks for each query (audit trail)."""

    __tablename__ = "retrievals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    query_id = Column(Integer, nullable=False)
    chunk_content = Column(Text)
    source_file = Column(String(200))
    page = Column(Integer)
    relevance_score = Column(Float)


def init_db():
    """Create all tables."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(engine)
    print("[DB] ✅ Database initialized")


def get_db():
    """Get a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
