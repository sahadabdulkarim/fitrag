"""BM25 sparse retriever for keyword-based document search."""

import pickle
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from app.core.config import PROCESSED_DATA_DIR


BM25_INDEX_PATH = PROCESSED_DATA_DIR / "bm25_index.pkl"


def tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenization."""
    # Remove punctuation and lowercase
    text = text.lower()
    # Keep alphanumeric and spaces
    cleaned = "".join(c if c.isalnum() or c.isspace() else " " for c in text)
    return cleaned.split()


class BM25Retriever:
    """Sparse retrieval using BM25 (term-frequency based)."""

    def __init__(self):
        self.bm25: BM25Okapi | None = None
        self.documents: list[Document] = []
        self.tokenized_corpus: list[list[str]] = []

    def build_index(self, chunks: list[Document]) -> None:
        """Build BM25 index from document chunks."""
        self.documents = chunks
        self.tokenized_corpus = [tokenize(doc.page_content) for doc in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print(f"[BM25] ✅ Index built from {len(chunks)} chunks")

    def save_index(self, path: Path | None = None) -> None:
        """Save BM25 index to disk."""
        save_path = path or BM25_INDEX_PATH
        save_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "documents": self.documents,
            "tokenized_corpus": self.tokenized_corpus,
        }
        with open(save_path, "wb") as f:
            pickle.dump(data, f)
        print(f"[BM25] 💾 Index saved to: {save_path}")

    def load_index(self, path: Path | None = None) -> None:
        """Load BM25 index from disk."""
        load_path = path or BM25_INDEX_PATH

        if not load_path.exists():
            raise FileNotFoundError(f"No BM25 index at: {load_path}")

        with open(load_path, "rb") as f:
            data = pickle.load(f)

        self.documents = data["documents"]
        self.tokenized_corpus = data["tokenized_corpus"]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print(f"[BM25] 📂 Index loaded ({len(self.documents)} chunks)")

    def retrieve(self, query: str, top_k: int = 10) -> list[tuple[Document, float]]:
        """Retrieve documents using BM25 scoring."""
        if self.bm25 is None:
            raise ValueError("BM25 index not built. Call build_index() or load_index() first.")

        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include non-zero scores
                results.append((self.documents[idx], float(scores[idx])))

        return results
