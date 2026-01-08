# Next-Generation RAG System — Documentation

## 1. Purpose

This project implements a **modular, extensible RAG (Retrieval-Augmented Generation) system**
designed to evolve beyond classical retrieval.

The system:
- Starts as a **standard RAG**
- Gradually upgrades to **simulation-based retrieval**
- Remains compatible with advanced engines (Paradox, Paradma, Paradox AI)

The core goal is to **separate retrieval, reasoning, and generation**.

---

## 2. What This RAG System Does

1. Ingests documents
2. Converts them into embeddings
3. Retrieves relevant knowledge
4. Builds a structured internal state
5. Reasons over that state
6. Optionally generates language output

The system **never directly reasons in prompts**.

---

## 3. Core Concepts

### 3.1 Knowledge Unit

All retrieved data is wrapped in a standard structure:

```python
KnowledgeUnit:
  content      # text / image / tensor
  embedding    # vector representation
  metadata     # source, timestamp, tags
  confidence   # optional
