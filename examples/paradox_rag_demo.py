"""
Paradox RAG Example: Using ParadoxVectorStore and ParadoxEmbeddings

Demonstrates the fully integrated Paradox ecosystem:
- ParadoxEmbeddings (modules.framework + Paradma self-learning)
- ParadoxVectorStore (ParadoxLF + Paradma backend)
- KnowledgeUnit with Paradma Axiom embeddings
"""

import sys
import os

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if grandparent_dir not in sys.path:
    sys.path.insert(0, grandparent_dir)

from parag.core import KnowledgeUnit
from parag.embeddings import ParadoxEmbeddings
from parag.vectorstore import ParadoxVectorStore
from parag.retrieval import Retriever
from parag.reasoning import StateManager
from parag.generation import DeterministicGenerator


def main():
    print("=" * 70)
    print("PARADOX RAG SYSTEM - Self-Learning Demo")
    print("=" * 70)
    print()
    print("[*] Using Paradox Ecosystem:")
    print("  - ParadoxEmbeddings (modules.framework + Paradma)")
    print("  - ParadoxVectorStore (ParadoxLF + Paradma backend)")
    print("  - KnowledgeUnit with Axiom embeddings")
    print()
    
    # Step 1: Create sample documents
    print("[1/7] Creating sample knowledge base...")
    
    documents = [
        "Paradma is a self-learning mathematical framework that learns from NumPy.",
        "ParadoxLF is an autonomous memory engine with Paradma backend.",
        "The AutoLearner observes NumPy operations and implements native versions.",
        "HyperMatrix provides quantum-like superposition for uncertain knowledge.",
        "modules.framework replaces PyTorch with custom Tensor and autograd.",
    ]
    
    print(f"  Created {len(documents)} knowledge documents")
    
    # Step 2: Initialize ParadoxEmbeddings
    print("\n[2/7] Initializing ParadoxEmbeddings...")
    
    embedding_model = ParadoxEmbeddings(
        embedding_dim=128,
        vocab_size=10000,
        normalize=True,
        use_paradma=True,  # Enable self-learning!
    )
    
    print(f"  {embedding_model}")
    
    # Step 3: Initialize ParadoxVectorStore
    print("\n[3/7] Initializing ParadoxVectorStore...")
    
    vector_store = ParadoxVectorStore(
        dimension=128,
        storage_dir=".paradox_rag_store"
    )
    
    print(f"  {vector_store}")
    
    # Step 4: Create knowledge units and add to RAG
    print("\n[4/7] Creating knowledge units with Paradma embeddings...")
    
    knowledge_units = []
    
    for i, doc in enumerate(documents):
        # Generate embedding using ParadoxEmbeddings
        # (Uses Paradma's learning.mean() internally - self-learning!)
        embedding = embedding_model.embed(doc)
        
        # Create knowledge unit with Paradma Axiom embedding
        unit = KnowledgeUnit(
            content=doc,
            metadata={
                "source": f"doc_{i}",
                "topic": "paradox_ecosystem"
            },
            _use_paradma=True  # Convert to Axiom
        )
        
        # Set embedding (auto-converts to Axiom if _use_paradma=True)
        unit.set_embedding_from_numpy(embedding, use_paradma=True)
        
        knowledge_units.append(unit)
        
        # Check if using Paradma
        status = "Axiom [OK]" if unit.is_using_paradma() else "NumPy"
        print(f"  [{i+1}/{len(documents)}] {status}: {doc[:50]}...")
    
    # Step 5: Initialize retriever
    print("\n[5/7] Initializing retriever with ParadoxVectorStore...")
    
    retriever = Retriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
        score_threshold=0.3
    )
    
    # Add knowledge units
    ids = retriever.add_knowledge_units(knowledge_units)
    print(f"  Added {len(ids)} knowledge units")
    print(f"  Paradox store: {retriever.is_paradox_store}")
    
    # Step 6: Perform retrieval
    print("\n[6/7] Performing retrieval...")
    
    query = "How does Paradma learn from NumPy?"
    print(f"  Query: '{query}'")
    
    result = retriever.retrieve(query, top_k=3)
    
    print(f"\n  Retrieval Results:")
    print(f"    Found: {len(result)} units")
    print(f"    Avg score: {result.get_average_score():.3f}")
    
    print("\n  Top matches:")
    for i, (unit, score) in enumerate(zip(result.units, result.scores), 1):
        print(f"    [{i}] Score: {score:.3f}")
        print(f"        {str(unit.content)[:70]}...")
        print(f"        Using Paradma: {unit.is_using_paradma()}")
    
    # Step 7: Generate response
    print("\n[7/7] Generating response...")
    
    state_manager = StateManager()
    state = state_manager.build_from_retrieval(result)
    
    generator = DeterministicGenerator()
    response = generator.generate_from_state(
        query=query,
        state=state,
        include_reasoning=True
    )
    
    print("\n" + "=" * 70)
    print("RESPONSE:")
    print("=" * 70)
    print(response)
    print("=" * 70)
    
    # Bonus: Show Paradma learning progress
    print("\n" + "=" * 70)
    print("PARADMA AUTO-LEARNER STATUS:")
    print("=" * 70)
    
    try:
        from paradma import learning
        learning.show_learning_progress()
    except:
        print("  Paradma learning manifold not accessible")
    
    # Bonus: Demonstrate creative features
    print("\n" + "=" * 70)
    print("BONUS: ParadoxVectorStore Creative Features")
    print("=" * 70)
    
    # Get two embeddings
    emb1 = embedding_model.embed("Paradma learns mathematics")
    emb2 = embedding_model.embed("ParadoxLF stores memories")
    
    # Creative blending
    print("\n  1. Concept Blending (imagine):")
    blended = vector_store.imagine(emb1, emb2, ratio=0.5)
    print(f"     Blended concept dimension: {len(blended)}")
    print(f"     This represents a hybrid: learning + memory system")
    
    # Save the store
    print("\n  2. Persistence:")
    vector_store.save()
    print(f"     Saved to: {vector_store.storage_dir}")
    
    print("\n" + "=" * 70)
    print("[DONE] Paradox RAG Demo Complete!")
    print("=" * 70)
    print("\nSummary:")
    print(f"  • Embeddings: Paradma self-learning enabled")
    print(f"  • Vector Store: ParadoxLF autonomous engine")
    print(f"  • Knowledge Units: Using Axiom embeddings")
    print(f"  • Total install size: ~20-30MB (vs ~2GB with PyTorch/FAISS)")
    print()


if __name__ == "__main__":
    main()
