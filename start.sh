#!/bin/bash
set -e

echo "Waiting for Qdrant..."

python - <<EOF
import socket
import time

while True:
    try:
        socket.create_connection(("qdrant", 6333), timeout=2)
        break
    except OSError:
        time.sleep(2)

print("Qdrant is ready.")
EOF

echo "Running ingestion..."
python -m src.bct_rag.pipeline


echo "Done."
