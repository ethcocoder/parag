"""
Basic usage example for Parag RAG system.

Demonstrates:
- Document ingestion
- Embedding generation
- Vector storage
- Retrieval
- Reasoning
- Response generation
"""

from parag.core import KnowledgeUnit
from parag.ingestion.loaders import TextLoader
from parag.ingestion.chunker import TextChunker, ChunkingStrategy
from parag.embeddings import SentenceTransformerEmbeddings
from parag.vectorstore import FAISSVectorStore
from parag.retrieval import Retriever
from parag.reasoning import StateManager
from parag.generation import DeterministicGenerator, PromptBuilder


def main():
    print("=" * 60)
    print("Parag RAG System - Basic Usage Example")
    print("=" * 60)
    
    # Step 1: Create sample documents
    print("\n[1/7] Creating sample documents...")
    
    sample_docs = [
        {
            "content": "Python is a high-level programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991.",
            "metadata": {
                "source": "python_intro.txt",
                "topic": "programming",
            }
        },
        {
            "content": "Machine learning is a subset of artificial intelligence that enables systems to learn from data without explicit programming. It uses statistical techniques to improve performance on tasks.",
            "metadata": {
                "source": "ml_basics.txt",
                "topic": "ai",
            }
        },
        {
            "content": "Python is widely used in machine learning and data science due to its extensive libraries like NumPy, Pandas, and Scikit-learn. It provides a simple syntax for complex algorithms.",
            "metadata": {
                "source": "python_ml.txt",
                "topic": "programming, ai",
            }
        },
        {
            "content": "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes that process information in layers.",
            "metadata": {
                "source": "neural_networks.txt",
                "topic": "ai",
            }
        },
    ]
    
    print(f"Created {len(sample_docs)} sample documents")
    
    # Step 2: Initialize embedding model
    print("\n[2/7] Initializing embedding model...")
    
    embedding_model = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2",
        normalize=True
    )
    
    embedding_dim = embedding_model.get_embedding_dim()
    print(f"Embedding model loaded: dimension = {embedding_dim}")
    
    # Step 3: Create vector store
    print("\n[3/7] Creating FAISS vector store...")
    
    vector_store = FAISSVectorStore(
        dimension=embedding_dim,
        index_type="Flat",
        metric="L2"
    )
    
    print(f"Vector store created: {vector_store}")
    
    # Step 4: Create retriever and add documents
    print("\n[4/7] Adding documents to retrieval system...")
    
    retriever = Retriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
        score_threshold=0.3
    )
    
    # Convert to knowledge units
    knowledge_units = []
    for doc in sample_docs:
        unit = KnowledgeUnit(
            content=doc["content"],
            metadata=doc["metadata"]
        )
        knowledge_units.append(unit)
    
    # Add to retriever
    ids = retriever.add_knowledge_units(knowledge_units)
    print(f"Added {len(ids)} knowledge units to the system")
    
    # Step 5: Perform retrieval
    print("\n[5/7] Performing retrieval...")
    
    query = "How is Python used in machine learning?"
    print(f"Query: '{query}'")
    
    result = retriever.retrieve(query, top_k=3)
    
    print(f"\nRetrieval results:")
    print(f"  - Found {len(result)} relevant units")
    print(f"  - Average score: {result.get_average_score():.3f}")
    print(f"  - Sources: {', '.join(result.get_sources())}")
    
    print("\nTop results:")
    for i, (unit, score) in enumerate(zip(result.units, result.scores), 1):
        content_preview = str(unit.content)[:80] + "..." if len(str(unit.content)) > 80 else str(unit.content)
        print(f"  [{i}] Score: {score:.3f}")
        print(f"      {content_preview}")
        print(f"      Source: {unit.get_source()}")
    
    # Step 6: Build state and reason
    print("\n[6/7] Building RAG state and reasoning...")
    
    state_manager = StateManager()
    state = state_manager.build_from_retrieval(result)
    
    # Detect conflicts
    conflicts = state.detect_conflicts()
    
    print(f"State summary:")
    print(f"  - Facts: {len(state.facts)}")
    print(f"  - Knowledge units: {len(state.knowledge_units)}")
    print(f"  - Conflicts: {len(conflicts)}")
    print(f"  - Uncertainty: {state.uncertainty:.2f}")
    print(f"  - High confidence facts: {len(state.get_high_confidence_facts())}")
    
    # Step 7: Generate response
    print("\n[7/7] Generating response...")
    
    generator = DeterministicGenerator()
    response = generator.generate_from_state(
        query=query,
        state=state,
        include_reasoning=True
    )
    
    print("\n" + "=" * 60)
    print("RESPONSE:")
    print("=" * 60)
    print(response)
    print("=" * 60)
    
    # Alternative: Build LLM prompt
    print("\n" + "=" * 60)
    print("LLM PROMPT (for reference):")
    print("=" * 60)
    
    prompt_builder = PromptBuilder(include_sources=True, include_confidence=True)
    llm_prompt = prompt_builder.build_from_result(query, result)
    print(llm_prompt)
    print("=" * 60)
    
    print("\n✅ Demo completed successfully!")


if __name__ == "__main__":
    main()
