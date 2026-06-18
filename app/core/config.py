"""Application configuration and settings."""

from pathlib import Path
from dataclasses import dataclass, field

# Project root directory
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VECTOR_STORE_DIR = PROCESSED_DATA_DIR / "vector_store"


@dataclass
class RAGConfig:
    """Configuration for the RAG pipeline."""

    # Embedding model
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Chunking parameters
    chunk_size: int = 512
    chunk_overlap: int = 50
    separators: list[str] = field(
        default_factory=lambda: ["\n\n", "\n", ". ", " ", ""]
    )

    # Retrieval parameters
    top_k: int = 5

    # FAISS index path
    faiss_index_path: str = str(VECTOR_STORE_DIR / "faiss_index")


@dataclass
class LLMConfig:
    """Configuration for LLM calls."""

    model: str = "gpt-4o-mini"
    temperature: float = 0.3
    max_tokens: int = 1024


# Global config instances
rag_config = RAGConfig()
llm_config = LLMConfig()
