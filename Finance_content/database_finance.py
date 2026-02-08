"""
Run this script ONCE to build and save the finance vector database from the PDF.
Do not run this again unless you change the PDF or want to rebuild the index.

  python database_finance.py

Output: .finance_rag/index.faiss and .finance_rag/chunks.pkl
After that, finance_rag.py only reads from these files; the vector DB is never created again when you run main.py.
"""

import pickle
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
PDF_PATH = _ROOT / "The-Complete-Guide-to-Trading (1).pdf"
RAG_INDEX_DIR = _ROOT / ".finance_rag"
RAG_INDEX_PATH = RAG_INDEX_DIR / "index.faiss"
RAG_CHUNKS_PATH = RAG_INDEX_DIR / "chunks.pkl"


def load_pdf_text(path: Path) -> str:
    """Extract text from each page of the PDF."""
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    parts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks for RAG."""
    if not text or chunk_size <= 0:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap
        if start >= len(text):
            break
    return chunks


def main():
    import numpy as np
    import faiss
    from sentence_transformers import SentenceTransformer

    if not PDF_PATH.exists():
        print(f"PDF not found: {PDF_PATH}")
        print("Place your PDF in the project folder and run again.")
        return

    print("Loading PDF...")
    text = load_pdf_text(PDF_PATH)
    print("Chunking text...")
    chunks = chunk_text(text, chunk_size=800, overlap=100)
    if not chunks:
        print("No text extracted from PDF.")
        return

    print(f"Loaded {len(chunks)} chunks. Creating embeddings (this may take a few minutes)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype="float32")

    print("Building FAISS index...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    RAG_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(RAG_INDEX_PATH))
    with open(RAG_CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"Done. Saved to {RAG_INDEX_DIR}")
    print("  - index.faiss")
    print("  - chunks.pkl")
    print("Run this script only once. When you run python main.py, finance_rag will use these files only.")


if __name__ == "__main__":
    main()
