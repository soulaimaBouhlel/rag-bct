"""
Central configuration — all paths and constants live here.
Import this everywhere instead of hardcoding strings.
"""
from pathlib import Path

# ── Root paths ────────────────────────────────────────────────────────────────
ROOT_DIR       = Path(__file__).resolve().parents[2]   # STAGE26/
DATA_DIR       = ROOT_DIR / "data"
PDF_DIR        = DATA_DIR / "pdfs"
CHUNKS_FILE    = DATA_DIR / "chunks.json"
LANCEDB_DIR    = ROOT_DIR / "lancedb_store"

# ── Embedding ─────────────────────────────────────────────────────────────────
EMBED_MODEL    = "BAAI/bge-m3"
EMBED_DIM      = 1024
EMBED_BATCH    = 16          # raise to 32 if RAM allows

# ── LanceDB ───────────────────────────────────────────────────────────────────
TABLE_NAME     = "bct_regulations"

# ── LLM ──────────────────────────────────────────────────────────────────────
OLLAMA_MODEL   = "qwen2.5:7b"
OLLAMA_BASE    = "http://localhost:11434"

# ── Chunking ──────────────────────────────────────────────────────────────────
# Articles longer than this get split by semchunk (future task)
MAX_CHUNK_TOKENS = 512