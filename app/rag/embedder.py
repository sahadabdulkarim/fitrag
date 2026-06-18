"""Embedding pipeline: convert document chunks to vectors and store in FAISS."""

from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.core.config import rag_config, VECTOR_STORE_DIR


class EmbeddingPipeline:
    """Embeds document chunks and manages the FAISS vector store."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or rag_config.embedding_model
        print(f"  Loading embedding model: {self.model_name}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vector_store: FAISS | None = None

    def build_index(self, chunks: list[Document]) -> FAISS:
        """Build a FAISS index from document chunks."""
        print(f"\n[Embedder] Building FAISS index from {len(chunks)} chunks...")
        self.vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings,
        )
        print(f"[Embedder] ✅ FAISS index built successfully")
        return self.vector_store

    def save_index(self, path: Path | None = None) -> None:
        """Save the FAISS index to disk."""
        if self.vector_store is None:
            raise ValueError("No vector store to save. Build the index first.")

        save_path = path or VECTOR_STORE_DIR
        save_path.mkdir(parents=True, exist_ok=True)

        self.vector_store.save_local(str(save_path))
        print(f"[Embedder] 💾 Index saved to: {save_path}")

    def load_index(self, path: Path | None = None) -> FAISS:
        """Load a FAISS index from disk."""
        load_path = path or VECTOR_STORE_DIR

        if not load_path.exists():
            raise FileNotFoundError(f"No index found at: {load_path}")

        self.vector_store = FAISS.load_local(
            str(load_path),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        print(f"[Embedder] 📂 Index loaded from: {load_path}")
        return self.vector_store

    def get_vector_store(self) -> FAISS:
        """Get the current vector store, loading from disk if needed."""
        if self.vector_store is None:
            self.load_index()
        return self.vector_store
