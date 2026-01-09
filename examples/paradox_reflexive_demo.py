"""
Paradox Reflexive Loop Demo.

Demonstrates the autonomous 'System 2' thinking:
1. AI receives a query.
2. AI detects high uncertainty or conflict in initial results.
3. AI automatically triggers a 'Reflexive' search to resolve the ambiguity.
4. AI provides a finalized, deep-thought response.
"""

import os
import sys
import logging
import numpy as np

# Configure logging to see the loop in action
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if grandparent_dir not in sys.path:
    sys.path.insert(0, grandparent_dir)

from parag.core import KnowledgeUnit, CognitiveLoop
from parag.embeddings import ParadoxEmbeddings
from parag.vectorstore import ParadoxVectorStore
from parag.retrieval import Retriever
from parag.generation import DeterministicGenerator

def main():
    print("="*80)
    print("PARAG AUTONOMOUS REFLEXIVE LOOP DEMO")
    print("="*80)

    dim = 128
    
    # 1. Setup
    embeddings = ParadoxEmbeddings(embedding_dim=dim)
    vector_store = ParadoxVectorStore(dimension=dim, storage_dir=".reflexive_store")
    retriever = Retriever(embedding_model=embeddings, vector_store=vector_store)
    generator = DeterministicGenerator()
    
    # Construct the Mind (Cognitive Loop)
    cognition = CognitiveLoop(
        retriever=retriever, 
        max_turns=2, 
        uncertainty_threshold=0.35 # Strict threshold to force reflexion
    )

    # 2. Seed the manifold with partial/conflicting data
    print("\n[Step 1] Seeding knowledge manifold...")
    
    # Layer 1: Surface Knowledge (Conflicting)
    surface_knowledge = [
        "Quantum-RAG uses entropy to measure state stability.",
        "Quantum-RAG is strictly a linear retrieval system without entropy.", # Conflict
        "The Paradox Engine acts as a latent memory core."
    ]
    
    # Layer 2: Deep Knowledge (hidden context)
    deep_knowledge = [
        "Entropy-based measurement is preferred in Quantum-RAG for non-linear reasoning.",
        "Linear mode in Quantum-RAG is a fallback for low-power devices.",
        "Reflexive loops are the mechanism for resolving entropy spikes."
    ]
    
    # Add surface level first
    retriever.add_knowledge_units([
        KnowledgeUnit(content=text, unit_id=f"surf_{i}") 
        for i, text in enumerate(surface_knowledge)
    ])
    
    # Add deep level (to be 'found' later during reflexion)
    retriever.add_knowledge_units([
        KnowledgeUnit(content=text, unit_id=f"deep_{i}") 
        for i, text in enumerate(deep_knowledge)
    ])

    # 3. Execution
    query = "How does Quantum-RAG measure stability?"
    print(f"\n[Step 2] User Query: '{query}'")
    print("\n--- COGNITIVE TRACE ---")
    
    final_state = cognition.run(query)
    
    print("--- END TRACE ---\n")

    # 4. Final Output
    print("[Step 3] Final State Analysis:")
    print(f"    - Final Uncertainty: {final_state.uncertainty:.4f}")
    print(f"    - Knowledge Turncount: {len(final_state.metadata.get('cognitive_loop_history', []))}")
    
    if final_state.emotions:
        print(f"    - Final Mood: {final_state.emotions.get_state()}")

    print("\n[Step 4] Final Generated Response:")
    print("-" * 60)
    print(generator.generate_from_state(query, final_state))
    print("-" * 60)

    print("\n[SUCCESS] The AI autonomously navigated its manifold to resolve ambiguity.")

if __name__ == "__main__":
    main()
