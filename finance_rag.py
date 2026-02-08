"""
Finance RAG: retrieve from the pre-built vector database only.
Does NOT load PDF, chunk, or create embeddings. Run database_finance.py once to create the index.
When you run main.py, this module only loads the saved index from .finance_rag/ and queries it.
"""

import pickle
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
RAG_INDEX_DIR = _ROOT / ".finance_rag"
RAG_INDEX_PATH = RAG_INDEX_DIR / "index.faiss"
RAG_CHUNKS_PATH = RAG_INDEX_DIR / "chunks.pkl"

_embedder = None
_faiss_index = None
_rag_chunks = None

# Fallback when no vector DB (run database_finance.py first to create it)
FINANCE_KNOWLEDGE = [
    {"text": "Long and short: Long means you buy an asset expecting the price to go up; you profit when you sell later at a higher price. Short means you sell an asset you don't own (you borrow it), expecting the price to go down; you buy it back later at a lower price, return it to the lender, and keep the difference. Going long = bullish; going short = bearish.", "keywords": "long, short, buy, sell, trade, bullish, bearish, invest"},
    {"text": "RSI (Relative Strength Index) measures momentum, not price direction. It compares recent gains to recent losses. When RSI is low (e.g. below 30), the asset is often considered oversold — selling pressure may be exhausting. A Buy signal can appear when price is still falling but momentum is turning (e.g. RSI rising from oversold).", "keywords": "rsi, oversold, momentum, buy, signal, falling, price"},
    {"text": "Buy and Sell signals are based on rules that combine indicators (RSI, MACD, trend). A signal can say Buy even when price is down if the logic sees improving momentum or oversold conditions. Price direction and signal direction can differ over short periods.", "keywords": "buy, sell, signal, indicator, momentum, trend, price"},
    {"text": "MACD (Moving Average Convergence Divergence) shows trend and momentum. When the MACD line crosses above the signal line, it can trigger a bullish or Buy signal. It does not depend on price going up at that exact moment; it depends on the relationship between short- and long-term averages.", "keywords": "macd, trend, momentum, buy, signal, crossover"},
    {"text": "Trend vs momentum: trend is the overall direction of price (up/down/sideways). Momentum is the speed or strength of recent moves. Signals often use both: e.g. Buy when trend is up and momentum is recovering from oversold.", "keywords": "trend, momentum, oversold, buy, signal"},
    {"text": "Beginner: Think of RSI like a speedometer for buying/selling pressure. Price can fall while the 'speed' of selling is already slowing — that can produce a Buy signal. It's about change in momentum, not just current price.", "keywords": "rsi, beginner, speedometer, momentum, buy, price"},
]


def _get_embedder():
    """Lazy-load sentence-transformers model (used only to embed the user query for search)."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def _load_index_into_memory():
    """
    Load the pre-built index from disk only. Never creates or writes the index.
    If index was not created (database_finance.py not run), leaves _faiss_index and _rag_chunks as None.
    """
    global _faiss_index, _rag_chunks
    if _faiss_index is not None and _rag_chunks is not None:
        return
    if not RAG_INDEX_PATH.exists() or not RAG_CHUNKS_PATH.exists():
        return  # No index: use keyword fallback. Run database_finance.py once to create it.
    import faiss
    _faiss_index = faiss.read_index(str(RAG_INDEX_PATH))
    with open(RAG_CHUNKS_PATH, "rb") as f:
        _rag_chunks = pickle.load(f)


def _retrieve_from_pdf(query: str, top_k: int = 4) -> list[str]:
    """Get top_k relevant chunks from the saved vector DB. Returns [] if index not found."""
    try:
        _load_index_into_memory()
        if _faiss_index is None or _rag_chunks is None:
            return []
        import numpy as np
        model = _get_embedder()
        q_emb = model.encode([query], show_progress_bar=False)
        q_emb = np.array(q_emb, dtype="float32")
        k = min(top_k, _faiss_index.ntotal)
        if k <= 0:
            return []
        _, indices = _faiss_index.search(q_emb, k)
        return [_rag_chunks[i] for i in indices[0] if 0 <= i < len(_rag_chunks)]
    except Exception:
        return []


def _retrieve_keyword(query: str, top_k: int = 4) -> list[str]:
    """Fallback when vector DB not available (database_finance.py not run)."""
    q = set(query.lower().split())
    scored = []
    for chunk in FINANCE_KNOWLEDGE:
        keys = set(chunk["keywords"].lower().split())
        score = len(q & keys) + 0.1 * len(keys)
        scored.append((score, chunk["text"]))
    scored.sort(key=lambda x: -x[0])
    return [text for _, text in scored[:top_k]]


def retrieve(query: str, top_k: int = 4) -> list[str]:
    """
    Retrieve context: use saved vector DB if available, else keyword fallback.
    Vector DB is never created here; run database_finance.py once to create it.
    """
    pdf_chunks = _retrieve_from_pdf(query, top_k=top_k)
    if pdf_chunks:
        return pdf_chunks
    return _retrieve_keyword(query, top_k=top_k)



def warmup():
    """
    Preload embedder + FAISS index into memory.
    Call once at server startup.
    """
    _get_embedder()
    _load_index_into_memory()