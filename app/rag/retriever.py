"""Retrieval module: query the FAISS index and return relevant chunks."""

from dataclasses import dataclass

from langchain_core.documents import Document

from app.core.config import rag_config
from app.rag.embedder import EmbeddingPipeline


@dataclass
class RetrievalResult:
    """A single retrieval result with content, metadata, and score."""

    content: str
    metadata: dict
    score: float


class DenseRetriever:
    """Dense retrieval using FAISS similarity search."""

    def __init__(self, embedding_pipeline: EmbeddingPipeline | None = None):
        self.embedding_pipeline = embedding_pipeline or EmbeddingPipeline()
        self.top_k = rag_config.top_k

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """Retrieve the most relevant document chunks for a query."""
        k = top_k or self.top_k
        vector_store = self.embedding_pipeline.get_vector_store()

        # similarity_search_with_score returns (Document, score) tuples
        results = vector_store.similarity_search_with_score(query, k=k)

        retrieval_results = []
        for doc, score in results:
            retrieval_results.append(
                RetrievalResult(
                    content=doc.page_content,
                    metadata=doc.metadata,
                    score=float(score),
                )
            )

        return retrieval_results

    def retrieve_as_documents(self, query: str, top_k: int | None = None) -> list[Document]:
        """Retrieve results as LangChain Document objects."""
        k = top_k or self.top_k
        vector_store = self.embedding_pipeline.get_vector_store()
        return vector_store.similarity_search(query, k=k)

    def format_context(self, results: list[RetrievalResult]) -> str:
        """Format retrieval results into a context string for the LLM."""
        context_parts = []
        for i, result in enumerate(results, 1):
            source = result.metadata.get("source_file", "unknown")
            context_parts.append(
                f"[Source {i}: {source}]\n{result.content}"
            )
        return "\n\n---\n\n".join(context_parts)
