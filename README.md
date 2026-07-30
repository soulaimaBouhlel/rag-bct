# BCT-RAG
### Local Retrieval-Augmented Generation (RAG) System for Banque Centrale de Tunisie Regulations

A local Retrieval-Augmented Generation (RAG) system built during my internship to query Banque Centrale de Tunisie (BCT) regulations through natural language.

The project performs:

- PDF ingestion
- Document parsing
- Semantic chunking
- Embedding generation
- Vector indexing with Qdrant
- Semantic retrieval
- Local LLM generation through Ollama
- MCP server integration for Claude Desktop

---

# Architecture

```
                    PDF Regulations
                           │
                           ▼
                  Document Extraction
                     (Docling/OCR)
                           │
                           ▼
                    Semantic Chunking
                           │
                           ▼
                  Embedding Generation
                  (SentenceTransformer)
                           │
                           ▼
                    Qdrant Vector DB
                           │
                           ▼
                 Semantic Retrieval
                           │
                           ▼
                  Local LLM (Ollama)
                           │
                           ▼
                    Generated Answer
                           │
                           ▼
                   MCP Server (FastMCP)
                           │
                           ▼
                    Claude Desktop
```

---

# Features

- Local document processing
- OCR support
- Semantic chunking
- Vector search
- Local embeddings
- Local LLM inference
- MCP Server
- Claude Desktop integration
- Dockerized deployment

---

# Technologies

## Backend

- Python 3.11
- FastMCP
- Ollama
- Qdrant

## NLP

- Sentence Transformers
- Transformers
- Docling
- RapidOCR

## Infrastructure

- Docker
- Docker Compose

---

# Project Structure

```
STAGE26/

├── data/
│   ├── pdfs/
│   ├── chunks.json
│
├── src/
│   └── bct_rag/
│       ├── ingestion/
│       ├── indexing/
│       ├── retrieval/
│       ├── llm/
│       ├── mcp/
│       ├── config.py
│       └── pipeline.py
│
├── Dockerfile
├── docker-compose.yml
├── start.sh
├── requirements.txt
└── README.md
```

---

# Requirements

- Python 3.11+
- Docker
- Docker Compose
- Ollama
- Git

---

# Installation

## Clone repository

```bash
git clone <repository-url>

cd STAGE26
```

---

## Create Python environment

```bash
python3 -m venv .venv

source .venv/bin/activate
```

---

## Install dependencies

```bash
pip install -r requirements.txt
```

---

# Ollama

Install Ollama.

Pull the model used by the project.

Example:

```bash
ollama pull qwen2.5:7b
```

Start Ollama:

```bash
ollama serve
```

Verify:

```bash
ollama list
```

---

# Qdrant

Start Qdrant

```bash
docker compose up -d qdrant
```

Verify

```bash
docker ps
```

Open dashboard

```
http://localhost:6333/dashboard
```

---

# Running the Complete Pipeline

The pipeline performs

1. PDF extraction
2. OCR
3. Chunking
4. Embeddings
5. Vector indexing

Simply run

```bash
docker compose run --rm rag-builder
```

Internally this executes

```
pipeline.py

↓

ingestion.pipeline.run()

↓

index_qdrant()

↓

Qdrant populated
```

---

# Running without Docker

Activate environment

```bash
source .venv/bin/activate
```

Run

```bash
python -m src.bct_rag.pipeline
```

---

# Querying the RAG

Example

```python
from src.bct_rag.pipeline import ask

print(
    ask(
        "What are the dividend distribution conditions?"
    )
)
```

---

# MCP Server

Start the server

```bash
python -m src.bct_rag.mcp.mcp_server
```

Expected output

```
Starting BCT MCP Server...
```

---

# Claude Desktop Integration

Claude configuration file

```
~/.config/Claude/claude_desktop_config.json
```

Example

```json
{
  "mcpServers": {
    "rag-server": {
      "command": "/home/USER/STAGE26/.venv/bin/python",
      "args": [
        "-m",
        "src.bct_rag.mcp.mcp_server"
      ],
      "env": {
        "PYTHONPATH": "/home/USER/STAGE26"
      }
    }
  }
}
```

Restart Claude Desktop.

The MCP tool should appear automatically.

---

# Docker

Build image

```bash
docker compose build
```

Run pipeline

```bash
docker compose run --rm rag-builder
```

Start Qdrant

```bash
docker compose up -d qdrant
```

Stop containers

```bash
docker compose down
```

---

# Docker Architecture

```
                Docker Network
               -----------------

         +------------------------+
         |       rag-builder      |
         |                        |
         |  Ingestion             |
         |  Chunking              |
         |  Embeddings            |
         |  Indexing              |
         +-----------+------------+
                     |
                     |
                     |
         +-----------v------------+
         |       Qdrant           |
         |   Vector Database      |
         +------------------------+
```

The `rag-builder` container performs the ingestion and indexing process.

The `qdrant` container stores vector embeddings and serves semantic search requests.

---

# Pipeline

```
PDFs

↓

Docling

↓

OCR

↓

Structured Text

↓

Semantic Chunking

↓

SentenceTransformer

↓

Embeddings

↓

Qdrant

↓

Retriever

↓

LLM

↓

Answer
```

---

# Configuration

Main configuration

```
src/bct_rag/config.py
```

Contains

- embedding model
- Qdrant configuration
- Ollama configuration
- LLM model
- chunk parameters

---

# Useful Commands

Build Docker image

```bash
docker compose build
```

Start Qdrant

```bash
docker compose up -d qdrant
```

Run indexing

```bash
docker compose run --rm rag-builder
```

Run MCP

```bash
python -m src.bct_rag.mcp.mcp_server
```

Test retrieval

```python
from src.bct_rag.pipeline import ask

print(ask("What are the dividend distribution conditions?"))
```

View Qdrant logs

```bash
docker logs bct-qdrant
```

Enter container

```bash
docker compose run --rm rag-builder
```

---

# Future Work

- GraphRAG integration
- Hybrid Retrieval
- Metadata filtering
- Incremental indexing
- Reranking
- Docker multi-stage optimization
- CI/CD pipeline
- Automatic PDF monitoring
- Evaluation framework
- Knowledge graph construction using Neo4j

---

# Author

**Soulaima Bouhlel**

Software Engineering Student

Internship Project — Local RAG System for Banque Centrale de Tunisie
