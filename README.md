# BCT-RAG

### Local Retrieval-Augmented Generation (RAG) System for Banque Centrale de Tunisie Regulations

A local Retrieval-Augmented Generation (RAG) system developed during my internship to query **Banque Centrale de Tunisie (BCT)** regulatory documents through natural language.

The system is designed to retrieve relevant regulatory information from BCT circulars and provide answers using a locally hosted language model.

The current system performs:

- PDF ingestion
- Document parsing
- OCR for scanned/document content
- Regulation-aware semantic chunking
- Local embedding generation
- Vector indexing with Qdrant
- Semantic vector retrieval
- Parent-Child Retrieval
- Local LLM generation through Ollama
- MCP server integration with Claude Desktop
- Dockerized deployment

The project is currently being extended toward a **Graph-Enhanced RAG architecture** capable of representing relationships between circulars, laws, articles, institutions, and regulatory concepts.

---

# Architecture

## Current Architecture

```text
                    BCT PDF Regulations
                           │
                           ▼
                  Document Extraction
                     (Docling/OCR)
                           │
                           ▼
                Regulation-Aware Chunking
                           │
                           ▼
                  Embedding Generation
                  (SentenceTransformer)
                           │
                           ▼
                    Qdrant Vector DB
                           │
                           ▼
                    Vector Retrieval
                           │
                           ▼
                  Parent-Child Retrieval
                           │
                           ▼
              Reconstruct Relevant Context
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

# Planned Graph-Enhanced Architecture

The next stages of the project extend the vector RAG system with a domain knowledge graph.

The target architecture will represent regulatory relationships such as:

```text
                    BCT Regulation
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
          Circular      Law       Article
              │          │          │
              │          │          │
              └──────┬───┴──────────┘
                     ▼
              Regulatory Concept
                     │
                     ▼
                 Institution
```

For example, a circular may reference several laws:

```text
Circular 2026-3
      │
      ├── references → Loi 2016-35
      │
      ├── references → Loi 2016-48
      │
      ├── references → Circulaire 2018-6
      │
      └── references → Circulaire 2021-5
```

This structure will allow the system to answer questions that require reasoning across multiple regulatory documents and different years.

---

# Features

## Current Features

- Local document processing
- PDF ingestion
- OCR support
- Document structure extraction
- Regulation-aware semantic chunking
- Local embeddings
- Vector search
- Qdrant vector database
- Parent-Child Retrieval
- Local LLM inference
- MCP Server
- Claude Desktop integration
- Dockerized pipeline

## Planned Features

- Graph-Enhanced Vector Search
- Domain Knowledge Graph
- Laws represented as graph nodes
- Circular-to-law relationships
- Circular-to-circular relationships
- Article-to-law relationships
- Relationship-aware reranking
- Metadata-based filtering
- Hybrid retrieval
- Automatic regulatory document discovery
- Automatic downloading of missing BCT regulations
- Incremental indexing
- Automated document monitoring
- Evaluation framework
- CI/CD pipeline

---

# Regulatory Knowledge Representation

BCT circulars frequently reference other regulatory documents.

For example, Circular 2026-3 references:

```text
Loi n°2016-35
Loi n°2016-48
Circulaire n°2018-6
Circulaire n°2021-5
```

The future knowledge graph will represent these documents explicitly rather than keeping the relationships only inside the text.

Example:

```text
              ┌─────────────────────┐
              │  Circulaire 2026-3  │
              └──────────┬──────────┘
                         │
          ┌──────────────┼──────────────┐
          │              │              │
     references      references      references
          │              │              │
          ▼              ▼              ▼
   Loi 2016-35     Loi 2016-48    Circulaire 2018-6
```

Laws will also be organized by year.

Example:

```text
2016
├── Loi n°2016-35
└── Loi n°2016-48

1994
└── Loi n°94-14
```

This will make it possible to answer questions involving regulatory evolution, for example:

> How did the regulatory requirements change between two circulars from different years, considering the laws referenced by each circular?

---

# Missing Regulatory Documents

The planned system will support automatic retrieval of missing regulatory documents.

If a user asks about a circular or law that is not currently present in the local database, the system will:

```text
User Question
      │
      ▼
Local Knowledge Base
      │
      ├── Document exists
      │       │
      │       ▼
      │    Retrieve
      │
      └── Document missing
              │
              ▼
       Official BCT Website
              │
              ▼
          Download PDF
              │
              ▼
        Process Document
              │
              ▼
          OCR / Parsing
              │
              ▼
           Chunking
              │
              ▼
          Embeddings
              │
              ▼
        Qdrant + Graph
              │
              ▼
       Available for future queries
```

The intended source for BCT regulatory documents is the official Banque Centrale de Tunisie website.

---

# Technologies

## Backend

- Python 3.11
- FastMCP
- Ollama
- Qdrant

## NLP / Document Processing

- Sentence Transformers
- Transformers
- Docling
- RapidOCR

## Infrastructure

- Docker
- Docker Compose
- Git / GitHub

## Planned Graph Layer

- Domain Knowledge Graph
- Graph-enhanced vector retrieval
- Relationship-aware reranking

---

# Project Structure

```text
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
│       │   └── parent_child.py
│       ├── llm/
│       ├── mcp/
│       ├── embedding/
│       ├── config.py
│       └── pipeline.py
│
├── tests/
│   ├── unit/
│   └── integration/
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

Install Ollama and pull the model used by the project.

Example:

```bash
ollama pull qwen2.5:7b
```

Start Ollama:

```bash
ollama serve
```

Verify the installed models:

```bash
ollama list
```

---

# Qdrant

Start Qdrant:

```bash
docker compose up -d qdrant
```

Verify:

```bash
docker ps
```

The Qdrant dashboard is available at:

```text
http://localhost:6333/dashboard
```

---

# Running the Complete Pipeline

The indexing pipeline performs:

1. PDF extraction
2. OCR when required
3. Document parsing
4. Regulation-aware chunking
5. Embedding generation
6. Vector indexing into Qdrant

Run:

```bash
docker compose run --rm rag-builder
```

The general flow is:

```text
pipeline.py

↓

ingestion.pipeline.run()

↓

Chunk generation

↓

Embedding generation

↓

index_qdrant()

↓

Qdrant populated
```

---

# Running Without Docker

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Run the pipeline:

```bash
python -m src.bct_rag.pipeline
```

---

# Parent-Child Retrieval

The current retrieval architecture uses Parent-Child Retrieval to preserve the semantic precision of small chunks while recovering the larger regulatory context to which those chunks belong.

The retrieval process is:

```text
User Question
      │
      ▼
Query Embedding
      │
      ▼
Qdrant Vector Search
      │
      ▼
Relevant Child Chunks
      │
      ▼
Identify Parent Article
      │
      ▼
Retrieve Sibling Chunks
      │
      ▼
Sort by Chunk Index
      │
      ▼
Reconstruct Parent Document
      │
      ▼
LLM Context
      │
      ▼
Answer
```

For example:

```text
Article 2
├── Child chunk 0
├── Child chunk 1
└── Child chunk 2
```

If the vector search retrieves:

```text
Child chunk 1
```

the system can identify its parent:

```text
Article 2
```

and recover the other chunks:

```text
Child chunk 0
Child chunk 1
Child chunk 2
```

The reconstructed article is then provided as context to the LLM.

This is particularly useful for regulatory documents where the meaning of a provision may depend on surrounding paragraphs or conditions.

---

# Querying the RAG

Example:

```python
from src.bct_rag.pipeline import ask

print(
    ask(
        "What are the dividend distribution conditions?"
    )
)
```

Example questions include:

```text
What are the dividend distribution conditions?

What conditions require prior approval from the BCT?

What are the solvency and Tier 1 requirements?

Which laws are referenced by Circular 2026-3?

What happens when a bank does not comply with
the capital adequacy requirements?
```

---

# MCP Server

Start the MCP server:

```bash
python -m src.bct_rag.mcp.mcp_server
```

Expected output:

```text
Starting BCT MCP Server...
```

The server exposes the RAG functionality to MCP-compatible clients.

---

# Claude Desktop Integration

Claude Desktop configuration:

```text
~/.config/Claude/claude_desktop_config.json
```

Example:

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

Restart Claude Desktop after modifying the configuration.

The MCP tool should then become available to Claude Desktop.

---

# Docker

Build the images:

```bash
docker compose build
```

Start Qdrant:

```bash
docker compose up -d qdrant
```

Run the indexing pipeline:

```bash
docker compose run --rm rag-builder
```

Stop the containers:

```bash
docker compose down
```

---

# Docker Architecture

```text
                 Docker Network
              -------------------

       +---------------------------+
       |        rag-builder        |
       |                           |
       |  Document Ingestion       |
       |  OCR / Parsing            |
       |  Chunking                 |
       |  Embeddings               |
       |  Indexing                 |
       +-------------+-------------+
                     |
                     |
                     ▼
       +---------------------------+
       |          Qdrant           |
       |      Vector Database      |
       +---------------------------+
```

The `rag-builder` container performs document processing and indexing.

The `qdrant` container stores vector embeddings and provides vector search capabilities.

The Parent-Child Retrieval logic is part of the application retrieval layer and uses Qdrant metadata to reconstruct parent documents.

---

# Testing

The project uses unit and integration tests to verify that retrieval modifications do not break the existing RAG pipeline.

Run all tests:

```bash
python -m pytest -v
```

Run unit tests:

```bash
python -m pytest tests/unit -v
```

Run integration tests:

```bash
python -m pytest tests/integration -v
```

The Parent-Child Retrieval tests verify:

- Parent identification
- Child grouping
- Child ordering
- Parent reconstruction
- Qdrant integration
- Compatibility with the existing RAG pipeline

---

# Development Workflow

Development is organized into separate feature branches.

Current branch:

```text
feature/parent-child-retriever
```

The implementation process follows:

```text
Feature Branch
      │
      ▼
Implementation
      │
      ▼
Unit Tests
      │
      ▼
Integration Tests
      │
      ▼
Pipeline Regression Tests
      │
      ▼
Docker Validation
      │
      ▼
GitHub Pull Request
      │
      ▼
Review
      │
      ▼
Merge
```

Each major architectural modification is implemented and validated independently before the next phase begins.

---

# Graph RAG Roadmap

The Parent-Child Retrieval phase is the foundation for the next stages of the architecture.

## Phase 1 — Parent-Child Retrieval

```text
Vector Search
      ↓
Child Chunks
      ↓
Parent Reconstruction
      ↓
LLM
```

Status:

```text
In development
```

## Phase 2 — Graph-Enhanced Vector Search

The vector retrieval system will be enhanced using regulatory relationships.

The graph will represent entities such as:

```text
Circular
Law
Article
Year
Institution
Regulatory Concept
```

Example:

```text
Circular 2026-3
      │
      ├── references → Loi 2016-35
      ├── references → Loi 2016-48
      ├── references → Circulaire 2018-6
      └── references → Circulaire 2021-5
```

## Phase 3 — Domain Knowledge Graph

A structured regulatory knowledge graph will be constructed to represent relationships between documents and regulatory entities.

The graph will support questions requiring information across multiple documents and years.

## Phase 4 — Relationship-Aware Reranking

Retrieved candidates will be reranked according to both:

- semantic similarity
- graph relationships

The objective is to prioritize documents that are both semantically relevant and structurally related to the query.

## Phase 5 — External Regulatory Document Retrieval

If a required BCT circular or law is not present in the local knowledge base, the system will be extended to retrieve the missing document from an official BCT source.

The downloaded document will then go through the normal processing pipeline:

```text
Official BCT Source
       ↓
PDF
       ↓
Extraction / OCR
       ↓
Chunking
       ↓
Embeddings
       ↓
Qdrant
       ↓
Knowledge Graph
```

Once indexed, the document will remain available for subsequent queries.

---

# Planned Retrieval Architecture

The final target architecture is:

```text
                         User Question
                              │
                              ▼
                     Query Understanding
                              │
                  ┌───────────┴───────────┐
                  │                       │
                  ▼                       ▼
             Vector Search           Graph Search
                  │                       │
                  ▼                       ▼
           Relevant Chunks          Related Entities
                  │                       │
                  └───────────┬───────────┘
                              ▼
                    Candidate Documents
                              │
                              ▼
                Relationship-Aware Reranking
                              │
                              ▼
                    Parent Reconstruction
                              │
                              ▼
                         Local LLM
                              │
                              ▼
                           Answer
```

This architecture is intended to support regulatory questions requiring both semantic retrieval and reasoning over relationships between laws, circulars, articles, and different regulatory years.

---

# Configuration

Main configuration:

```text
src/bct_rag/config.py
```

The configuration contains parameters for:

- Embedding model
- Embedding dimensions
- Qdrant host
- Qdrant port
- Qdrant collection
- Ollama host
- Ollama model
- Chunking parameters
- Retrieval parameters

---

# Useful Commands

## Build Docker image

```bash
docker compose build
```

## Start Qdrant

```bash
docker compose up -d qdrant
```

## Run indexing

```bash
docker compose run --rm rag-builder
```

## Run tests

```bash
python -m pytest -v
```

## Run MCP

```bash
python -m src.bct_rag.mcp.mcp_server
```

## Test the RAG

```python
from src.bct_rag.pipeline import ask

print(
    ask(
        "What are the dividend distribution conditions?"
    )
)
```

## View Qdrant logs

```bash
docker logs bct-qdrant
```

## Enter the builder container

```bash
docker compose run --rm rag-builder bash
```

## Check the current Git branch

```bash
git branch --show-current
```

## Push the current branch

```bash
git push origin feature/parent-child-retriever
```

---

# Project Development Status

| Component | Status |
|---|---|
| PDF ingestion | Implemented |
| Document parsing | Implemented |
| OCR | Implemented |
| Semantic chunking | Implemented |
| Embedding generation | Implemented |
| Qdrant indexing | Implemented |
| Vector retrieval | Implemented |
| Local LLM | Implemented |
| MCP Server | Implemented |
| Claude Desktop integration | Implemented |
| Docker deployment | Implemented |
| Parent-Child Retrieval | In development |
| Metadata enhancement | Planned / ongoing |
| Graph-Enhanced Vector Search | Planned |
| Domain Knowledge Graph | Planned |
| Relationship-aware reranking | Planned |
| Automatic missing-document retrieval | Planned |
| CI/CD | Planned |
| Evaluation framework | Planned |

---

# Author

**Soulaima Bouhlel**

Software Engineering Student

Internship Project — Local RAG System for Banque Centrale de Tunisie