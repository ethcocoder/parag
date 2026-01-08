
---

# 🛣️ `roadmap.md`

```md
# RAG System — Development Roadmap

## Phase 1 — Classic RAG Foundation
**Goal:** Fully working standard RAG

- [ ] Document ingestion
- [ ] Text chunking
- [ ] Embedding generation
- [ ] FAISS vector store
- [ ] Similarity-based retrieval
- [ ] LLM answer generation

Deliverable:
- Ask questions over documents and get grounded answers

---

## Phase 2 — Structured Retrieval
**Goal:** Stop passing raw text to the LLM

- [ ] KnowledgeUnit abstraction
- [ ] RetrievalResult object
- [ ] Source tracking
- [ ] Confidence scoring

Deliverable:
- Retrieval produces structured data

---

## Phase 3 — RAG State & Reasoning
**Goal:** Add reasoning without LLM prompts

- [ ] RAGState object
- [ ] Fact aggregation
- [ ] Conflict detection
- [ ] Uncertainty measurement
- [ ] Rule-based reasoning

Deliverable:
- System knows when information conflicts or is missing

---

## Phase 4 — Generation as a Final Step
**Goal:** Make LLM optional

- [ ] State-to-text conversion
- [ ] Deterministic explanation mode
- [ ] “I don’t know” handling

Deliverable:
- Clean separation between thinking and speaking

---

## Phase 5 — Advanced Retrieval (Optional)
**Goal:** Prepare for Paradox-style memory

- [ ] Concept blending hooks
- [ ] Temporal retrieval
- [ ] Attention-weighted search

Deliverable:
- Retrieval becomes simulation-ready

---

## Phase 6 — Law-Based Reasoning (Optional)
**Goal:** Paradma compatibility

- [ ] Reasoning laws
- [ ] Conflict resolution via rules
- [ ] Entropy & decay handling

---

## Phase 7 — Cognitive Loop (Optional)
**Goal:** Paradox AI compatibility

- [ ] Entropy thresholds
- [ ] Curiosity-driven re-query
- [ ] Human feedback ingestion

---

## Final Outcome

A RAG system that:
- Works today
- Scales to future cognitive engines
- Is explainable and testable
