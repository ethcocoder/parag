# Parag - Next-Generation RAG System

A **self-learning**, **autonomous** RAG (Retrieval-Augmented Generation) system powered by the **Paradox ecosystem**. Parag eliminates heavy ML dependencies (PyTorch, FAISS, transformers) and replaces them with custom, self-learning frameworks.

## 🌟 Key Features

### Self-Learning Architecture (Paradox Ecosystem)
- **Paradma** - Self-learning mathematical operations (learns from NumPy, graduates to native Python)
- **ParadoxLF** - Autonomous memory engine with creative capabilities
- **modules.framework** - Custom PyTorch replacement with Tensor and autograd
- **HyperMatrix** - Quantum-like superposition for uncertain knowledge

### Core Capabilities
- **Minimal Dependencies**: Only ~15MB (NumPy + PyPDF2 + tqdm) vs ~2GB with traditional stack
- **Progressive Learning**: Operations start with NumPy, evolve to native implementations
- **Creative Features**: Concept blending via `imagine()`, temporal prediction
- **Quantum-Like Reasoning**: Superposition support for conflicting facts
- **Autonomous Optimization**: Memory engine evolves independently
- **Full Explainability**: Deterministic responses without LLM when appropriate

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/ethcocoder/parag.git
cd parag

# Install minimal dependencies
pip install -r requirements.txt

# Ensure Paradox ecosystem is available
# (Paradma, ParadoxLF, modules should be in parent directory)
```

### Total Install Size
- **Before**: ~2GB (PyTorch + FAISS + transformers)
- **After**: ~15MB (NumPy + PyPDF2 + tqdm)
- **Reduction**: 98.5% smaller! 🎉

## 🚀 Quick Start (Paradox-Powered)

```python
from parag.core import KnowledgeUnit
from parag.embeddings import ParadoxEmbeddings  # Self-learning embeddings!
from parag.vectorstore import ParadoxVectorStore  # Autonomous memory
from parag.retrieval import Retriever
from parag.reasoning import StateManager
from parag.generation import DeterministicGenerator

# 1. Ingest documents
documents = load_document("path/to/document.pdf")
chunked_docs = chunk_documents(documents)

# 2. Create embeddings
embedding_model = SentenceTransformerEmbeddings()
embedding_dim = embedding_model.get_embedding_dim()

# 3. Initialize vector store
vector_store = FAISSVectorStore(dimension=embedding_dim)

# 4. Create knowledge units and add to retrieval system
retriever = Retriever(embedding_model, vector_store)

units = [
    KnowledgeUnit(
        content=doc["content"],
        metadata=doc["metadata"]
    )
    for doc in chunked_docs
]

retriever.add_knowledge_units(units)

# 5. Retrieve relevant information
query = "What is the main topic?"
result = retriever.retrieve(query, top_k=5)

# 6. Build state and reason
state_manager = StateManager()
state = state_manager.build_from_retrieval(result)
state.detect_conflicts()

# 7. Generate response
generator = DeterministicGenerator()
response = generator.generate_from_state(query, state)
print(response)
```

## 📦 Architecture

```
parag/
├── core/                # Core data models
│   ├── knowledge_unit.py
│   ├── retrieval_result.py
│   └── rag_state.py
├── ingestion/          # Document processing
│   ├── loaders.py
│   ├── chunker.py
│   └── metadata.py
├── embeddings/         # Embedding generation
│   ├── base.py
│   └── sentence_transformer.py
├── vectorstore/        # Vector database
│   ├── faiss_store.py
│   └── index_manager.py
├── retrieval/          # Retrieval engine
│   ├── retriever.py
│   └── ranker.py
├── reasoning/          # Reasoning layer
│   ├── state_manager.py
│   ├── conflict_detector.py
│   └── uncertainty.py
└── generation/         # Response generation
    ├── prompt_builder.py
    └── llm_adapter.py
```

## 🎯 Core Concepts

### KnowledgeUnit
All retrieved data is wrapped in a standard structure:
```python
KnowledgeUnit:
  content         # text / image / tensor
  embedding       # vector representation
  metadata        # source, timestamp, tags
  confidence      # optional confidence score
```

### RAGState
Internal state representation for reasoning:
```python
RAGState:
  facts           # aggregated facts from knowledge units
  knowledge_units # all contributing units
  conflicts       # detected contradictions
  uncertainty     # uncertainty measurement
```

### RetrievalResult
Structured container for retrieval outputs:
```python
RetrievalResult:
  units           # list of KnowledgeUnits
  scores          # relevance scores
  query           # original query
  metadata        # retrieval metadata
```

## 🛣️ Roadmap

### ✅ Phase 1-4: Foundation (Current)
- [x] Classic RAG foundation
- [x] Structured retrieval with KnowledgeUnit
- [x] RAG state and reasoning layer
- [x] Deterministic generation

### 🔜 Phase 5-7: Advanced Features (Future)
- [ ] Concept blending hooks
- [ ] Temporal retrieval
- [ ] Paradma law-based reasoning
- [ ] Entropy thresholds
- [ ] Curiosity-driven re-query
- [ ] Human feedback ingestion

## 🧪 Testing

```bash
# Run tests
pytest tests/

# With coverage
pytest tests/ --cov=parag
```

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🔗 Links

- **Documentation**: See `doc/` directory
- **Roadmap**: `doc/roadmap.md`
- **Todo**: `doc/todo.md`

## 💡 Philosophy

Parag is built on the principle that **reasoning should happen in structured state, not in prompts**. By separating retrieval, reasoning, and generation:

1. **Retrieval** finds relevant knowledge
2. **Reasoning** operates on structured facts
3. **Generation** is the final, optional step

This enables:
- **Transparency**: Every decision is traceable
- **Scalability**: Each layer can evolve independently
- **Reliability**: Deterministic behavior when needed
- **Future-proofing**: Ready for advanced cognitive engines

---

**Built with ❤️ by ethcocoder**
