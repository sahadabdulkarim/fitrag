"""Main pipeline: ingest documents, build embeddings, and save indices."""

from app.core.config import VECTOR_STORE_DIR
from app.rag.ingest import DocumentIngester
from app.rag.embedder import EmbeddingPipeline
from app.rag.bm25_retriever import BM25Retriever


def build_index():
    """Full pipeline: ingest → embed → save FAISS index + BM25 index."""
    print("\n" + "=" * 60)
    print("🏋️ FitRAG - Building Knowledge Base Index")
    print("=" * 60)

    # Step 1: Ingest documents
    ingester = DocumentIngester()
    chunks = ingester.ingest()

    if not chunks:
        print("\n❌ No documents found! Add documents to data/raw/ directory.")
        return

    # Step 2: Build FAISS (dense) index
    print("\n" + "=" * 60)
    print("🧠 DENSE EMBEDDING PIPELINE (FAISS)")
    print("=" * 60)

    embedder = EmbeddingPipeline()
    embedder.build_index(chunks)
    embedder.save_index()

    # Step 3: Build BM25 (sparse) index
    print("\n" + "=" * 60)
    print("📝 SPARSE INDEX PIPELINE (BM25)")
    print("=" * 60)

    bm25 = BM25Retriever()
    bm25.build_index(chunks)
    bm25.save_index()

    # Summary
    print("\n" + "=" * 60)
    print("✅ INDEX BUILD COMPLETE")
    print(f"   Chunks indexed: {len(chunks)}")
    print(f"   FAISS index: {VECTOR_STORE_DIR}")
    print(f"   BM25 index: bm25_index.pkl")
    print("=" * 60)
    print("\nYou can now query using: python -m app.query")
    print("  Dense only:  python -m app.query --mode dense \"question\"")
    print("  Hybrid+Rerank: python -m app.query --mode hybrid \"question\"")


if __name__ == "__main__":
    build_index()
