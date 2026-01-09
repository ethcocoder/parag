"""
Paradox Cognitive RAG Demo.

Showcases the 'Emotional' and 'Intuitive' side of Parag, powered by:
- AIEmotions (Fluxion, Reflexion, Equilibria)
- AlienIntuition (Manifold Curvature)
- ParadoxLF + Paradma backend
"""

import os
import sys
import numpy as np

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if grandparent_dir not in sys.path:
    sys.path.insert(0, grandparent_dir)

from parag.core import KnowledgeUnit, RAGState
from parag.embeddings import ParadoxEmbeddings
from parag.vectorstore import ParadoxVectorStore
from parag.retrieval import Retriever
from parag.generation import DeterministicGenerator

def main():
    print("="*70)
    print("PARAG COGNITIVE ARCHITECTURE DEMO")
    print("="*70)

    dim = 128
    
    # 1. Setup Components
    embeddings = ParadoxEmbeddings(embedding_dim=dim)
    vector_store = ParadoxVectorStore(dimension=dim, storage_dir=".cognitive_store")
    retriever = Retriever(embedding_model=embeddings, vector_store=vector_store)
    generator = DeterministicGenerator()

    # 2. Ingest "Complex" and "Conflicting" data to trigger cognitive signals
    print("\n[1] Ingesting complex & contradictory knowledge...")
    knowledge = [
        "The Paradox Engine is strictly deterministic and predictable.",
        "The Paradox Engine is not deterministic and it is unpredictable.",
        "Axiom-0 is the root of all learning within Paradma.",
        "Sub-Manifolds exhibit high curvature when processing novel inputs.",
        "Cognitive equilibrium is reached when entropy is minimized."
    ]
    
    units = [KnowledgeUnit(content=text, unit_id=f"k{i}") for i, text in enumerate(knowledge)]
    retriever.add_knowledge_units(units)
    print(f"    Ingested {len(units)} units into Paradox manifold.")

    # 3. Perform a query that targets the conflict
    query = "Is the Paradox Engine predictable?"
    print(f"\n[2] Querying: '{query}'")
    
    result = retriever.retrieve(query, top_k=5)
    
    # 4. Build State and observe cognitive evolution
    state = RAGState.from_retrieval_result(result)
    state.detect_conflicts()
    
    print("\n[3] Cognitive State Analysis:")
    if "emotions" in state.metadata:
        emotions = state.metadata["emotions"]
        print(f"    Emotional State: {emotions}")
        
    curvature = state.metadata.get("manifold_curvature", 0.0)
    print(f"    Manifold Curvature: {curvature:.4f}")
    
    # 5. Generate with Cognitive Tone
    print("\n[4] Integrated Response (Reporting Feelings & Intuition):")
    print("-" * 50)
    response = generator.generate_from_state(query, state)
    print(response)
    print("-" * 50)

    print("\n[DONE] Paradox Cognitive architecture is active and self-aware.")

if __name__ == "__main__":
    main()
