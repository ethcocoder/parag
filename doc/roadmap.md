
---

# 🛣️ `roadmap.md`

```md
# RAG System — Development Roadmap

## Phase 1 — Classic RAG Foundation
**Goal:** Fully working standard RAG

- [x] Document ingestion
- [x] Text chunking
- [x] Embedding generation
- [x] FAISS vector store
- [x] Similarity-based retrieval
- [x] LLM answer generation

Deliverable:
- Ask questions over documents and get grounded answers

---

## Phase 2 — Structured Retrieval
**Goal:** Stop passing raw text to the LLM

- [x] KnowledgeUnit abstraction
- [x] RetrievalResult object
- [x] Source tracking
- [x] Confidence scoring

Deliverable:
- Retrieval produces structured data

---

## Phase 3 — RAG State & Reasoning
**Goal:** Add reasoning without LLM prompts

- [x] RAGState object
- [x] Fact aggregation
- [x] Conflict detection
- [x] Uncertainty measurement
- [x] Rule-based reasoning (Paradma)

Deliverable:
- System knows when information conflicts or is missing

---

## Phase 4 — Generation as a Final Step
**Goal:** Make LLM optional

- [x] State-to-text conversion
- [x] Deterministic explanation mode
- [x] “I don’t know” handling (Emotional fallback)

Deliverable:
- Clean separation between thinking and speaking

---

## Phase 5 — Advanced Retrieval (Optional)
**Goal:** Prepare for Paradox-style memory

- [x] Concept blending hooks (ParadoxLF imagine)
- [x] Temporal retrieval (Self-learning progress)
- [x] Attention-weighted search (Axiom math)

Deliverable:
- Retrieval becomes simulation-ready

---

- [x] Reasoning laws (AlienIntuition)
- [x] Conflict resolution via rules (Paradma)
- [x] Entropy & decay handling (Manifold Curvature)

## Phase 7 — Cognitive Loop
**Goal:** Paradox AI Sentience

- [x] Entropy thresholds
- [x] Curiosity-driven re-query (CognitiveLoop)
- [x] Human feedback inception (Sentient Generator)

---

## Final Outcome

A RAG system that:
- Works today
- Scales to future cognitive engines
- Is explainable and testable
