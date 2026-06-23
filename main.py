"""
Main entry point for the BCT RAG pipeline.
Usage:
  python main.py ingest        # PDF → chunks.json
  python main.py store         # chunks.json → LanceDB
  python main.py all           # ingest + store
"""

import sys


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"

    if cmd == "ingest":
        from bct_rag.ingestion.pipeline import run
        run()

    elif cmd == "store":
        from bct_rag.storage.store import store_chunks
        store_chunks()

    elif cmd == "all":
        from bct_rag.ingestion.pipeline import run
        from bct_rag.storage.store import store_chunks
        run()
        store_chunks()

    else:
        print(__doc__)


if __name__ == "__main__":
    main()