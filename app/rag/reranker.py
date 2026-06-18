"""Cross-encoder reranker for improving retrieval precision."""

from sentence_transformers import CrossEncoder
from langchain_core.documents import Document


class Reranker:
    """Reranks retrieved documents using a cross-encoder model."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        print(f"[Reranker] Loading model: {model_name}")
        self.model = CrossEncoder(model_name)
        self.model_name = model_name

    def rerank(
        self, query: str, documents: list[Document], top_k: int = 5
    ) -> list[tuple[Document, float]]:
        """Rerank documents by relevance to query using cross-encoder.

        Args:
            query: The user's question
            documents: List of candidate documents to rerank
            top_k: Number of top results to return after reranking

        Returns:
            List of (document, score) tuples sorted by relevance (highest first)
        """
        if not documents:
            return []

        # Create query-document pairs for the cross-encoder
        pairs = [[query, doc.page_content] for doc in documents]

        # Score all pairs
        scores = self.model.predict(pairs)

        # Sort by score (descending) and return top_k
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return scored_docs[:top_k]
