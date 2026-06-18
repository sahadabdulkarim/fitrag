"""Document ingestion pipeline: load, chunk, and prepare documents for embedding."""

from pathlib import Path
from dataclasses import dataclass

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.core.config import rag_config, RAW_DATA_DIR


@dataclass
class ChunkedDocument:
    """A document chunk with metadata."""

    content: str
    metadata: dict


class DocumentIngester:
    """Ingests raw documents (PDF, markdown, text) and splits them into chunks."""

    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None):
        self.chunk_size = chunk_size or rag_config.chunk_size
        self.chunk_overlap = chunk_overlap or rag_config.chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=rag_config.separators,
            length_function=len,
        )

    def load_pdf(self, file_path: Path) -> list[Document]:
        """Load a PDF file and return pages as documents."""
        loader = PyPDFLoader(str(file_path))
        return loader.load()

    def load_markdown(self, file_path: Path) -> list[Document]:
        """Load a markdown/text file as a document."""
        loader = TextLoader(str(file_path), encoding="utf-8")
        return loader.load()

    def load_directory(self, directory: Path | None = None) -> list[Document]:
        """Load all supported documents from a directory."""
        directory = directory or RAW_DATA_DIR
        documents = []

        if not directory.exists():
            raise FileNotFoundError(f"Data directory not found: {directory}")

        # Load PDFs
        for pdf_file in directory.glob("**/*.pdf"):
            print(f"  Loading PDF: {pdf_file.name}")
            docs = self.load_pdf(pdf_file)
            for doc in docs:
                doc.metadata["source_file"] = pdf_file.name
                doc.metadata["file_type"] = "pdf"
            documents.extend(docs)

        # Load Markdown files
        for md_file in directory.glob("**/*.md"):
            print(f"  Loading Markdown: {md_file.name}")
            docs = self.load_markdown(md_file)
            for doc in docs:
                doc.metadata["source_file"] = md_file.name
                doc.metadata["file_type"] = "markdown"
            documents.extend(docs)

        # Load text files
        for txt_file in directory.glob("**/*.txt"):
            print(f"  Loading Text: {txt_file.name}")
            docs = self.load_markdown(txt_file)  # TextLoader works for .txt too
            for doc in docs:
                doc.metadata["source_file"] = txt_file.name
                doc.metadata["file_type"] = "text"
            documents.extend(docs)

        print(f"\n  Total documents loaded: {len(documents)}")
        return documents

    def chunk_documents(self, documents: list[Document]) -> list[Document]:
        """Split documents into chunks suitable for embedding."""
        chunks = self.text_splitter.split_documents(documents)
        print(f"  Split into {len(chunks)} chunks (size={self.chunk_size}, overlap={self.chunk_overlap})")
        return chunks

    def ingest(self, directory: Path | None = None) -> list[Document]:
        """Full ingestion pipeline: load → chunk → return."""
        print("=" * 60)
        print("📄 DOCUMENT INGESTION PIPELINE")
        print("=" * 60)

        print("\n[1/2] Loading documents...")
        documents = self.load_directory(directory)

        print("\n[2/2] Chunking documents...")
        chunks = self.chunk_documents(documents)

        print("\n" + "=" * 60)
        print(f"✅ Ingestion complete: {len(chunks)} chunks ready for embedding")
        print("=" * 60)

        return chunks


if __name__ == "__main__":
    ingester = DocumentIngester()
    chunks = ingester.ingest()
    # Print first few chunks for inspection
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Source: {chunk.metadata.get('source_file', 'unknown')}")
        print(f"Content: {chunk.page_content[:200]}...")
