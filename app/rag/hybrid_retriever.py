"""Hybrid retriever combining dense (FAISS) + sparse (BM25) with reranking."""

from dataclasses import dataclass

from langchain_core.documents import Document

from app.core.config import rag_config
from app.rag.embedder import EmbeddingPipeline
from app.rag.retriever import DenseRetriever, RetrievalResult
from app.rag.bm25_retriever import BM25Retriever
from app.rag.reranker import Reranker


@dataclass
class HybridResult:
    """A hybrid retrieval result with RRF score."""

    content: str
    metadata: dict
    rrf_score: float
    rerank_score: float | None = None


class HybridRetriever:
    """Hybrid retrieval: Dense + BM25 → RRF fusion → Cross-encoder reranking.

    Pipeline:
    1. Dense retrieval (FAISS) → top-k candidates
    2. Sparse retrieval (BM25) → top-k candidates
    3. Reciprocal Rank Fusion (RRF) to merge results
    4. Cross-encoder reranking on fused candidates
    5. Return final top-k
    """

    def __init__(
        self,
        embedding_pipeline: EmbeddingPipeline | None = None,
        bm25_retriever: BM25Retriever | None = None,
        reranker: Reranker | None = None,
        use_reranker: bool = True,
    ):
        self.dense_retriever = DenseRetriever(embedding_pipeline=embedding_pipeline)
        self.bm25_retriever = bm25_retriever or BM25Retriever()
        self.use_reranker = use_reranker
        self._reranker = reranker

    @property
    def reranker(self) -> Reranker:
        """Lazy-load the reranker (heavy model)."""
        if self._reranker is None:
            self._reranker = Reranker()
        return self._reranker

    def reciprocal_rank_fusion(
        self,
        dense_results: list[tuple[Document, float]],
        bm25_results: list[tuple[Document, float]],
        k: int = 60,
    ) -> list[tuple[Document, float]]:
        """Merge results from dense and sparse retrieval using RRF.

        RRF Score = sum(1 / (k + rank_i)) for each retrieval method

        Args:
            dense_results: (Document, score) from FAISS
            bm25_results: (Document, score) from BM25
            k: Constant to prevent high-ranked docs from dominating (default 60)

        Returns:
            Merged list of (Document, rrf_score) sorted by RRF score descending
        """
        # Map document content → Document object (for deduplication)
        doc_scores: dict[str, tuple[Document, float]] = {}

        # Score dense results
        for rank, (doc, _score) in enumerate(dense_results, start=1):
            key = doc.page_content[:200]  # Use first 200 chars as key for dedup
            rrf_score = 1.0 / (k + rank)
            if key in doc_scores:
                doc_scores[key] = (doc, doc_scores[key][1] + rrf_score)
            else:
                doc_scores[key] = (doc, rrf_score)

        # Score BM25 results
        for rank, (doc, _score) in enumerate(bm25_results, start=1):
            key = doc.page_content[:200]
            rrf_score = 1.0 / (k + rank)
            if key in doc_scores:
                doc_scores[key] = (doc, doc_scores[key][1] + rrf_score)
            else:
                doc_scores[key] = (doc, rrf_score)

        # Sort by RRF score
        fused = sorted(doc_scores.values(), key=lambda x: x[1], reverse=True)
        return fused

    def retrieve(self, query: str, top_k: int = 5, candidates_k: int = 20) -> list[HybridResult]:
        """Full hybrid retrieval pipeline.

        Args:
            query: User question
            top_k: Final number of results to return
            candidates_k: Number of candidates from each retriever before fusion
        """
        # Step 1: Dense retrieval (FAISS)
        vector_store = self.dense_retriever.embedding_pipeline.get_vector_store()
        dense_raw = vector_store.similarity_search_with_score(query, k=candidates_k)
        dense_results = [(doc, score) for doc, score in dense_raw]

        # Step 2: Sparse retrieval (BM25)
        bm25_results = self.bm25_retriever.retrieve(query, top_k=candidates_k)

        # Step 3: Reciprocal Rank Fusion
        fused_results = self.reciprocal_rank_fusion(dense_results, bm25_results)

        # Step 4: Reranking (optional)
        if self.use_reranker:
            # Take top candidates for reranking (reranking is expensive)
            candidates = [doc for doc, _score in fused_results[:candidates_k]]
            reranked = self.reranker.rerank(query, candidates, top_k=top_k)

            return [
                HybridResult(
                    content=doc.page_content,
                    metadata=doc.metadata,
                    rrf_score=0.0,  # RRF score is from pre-reranking
                    rerank_score=float(score),
                )
                for doc, score in reranked
            ]
        else:
            # Without reranker, use RRF scores directly
            return [
                HybridResult(
                    content=doc.page_content,
                    metadata=doc.metadata,
                    rrf_score=float(score),
                    rerank_score=None,
                )
                for doc, score in fused_results[:top_k]
            ]

    def format_context(self, results: list[HybridResult]) -> str:
        """Format hybrid results into context string for LLM."""
        context_parts = []
        for i, result in enumerate(results, 1):
            source = result.metadata.get("source_file", "unknown")
            page = result.metadata.get("page", "?")
            context_parts.append(
                f"[Source {i}: {source}, p.{page}]\n{result.content}"
            )
        return "\n\n---\n\n".join(context_parts)
