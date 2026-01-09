"""
Full Paradox Integration Test Suite for Parag.

Tests the interaction between:
- ParadoxEmbeddings
- ParadoxVectorStore
- KnowledgeUnit (Axiom mode)
- Retriever
- RAGState (Paradma uncertainty)
- DeterministicGenerator
"""

import os
import sys
import unittest
import numpy as np
import shutil

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if grandparent_dir not in sys.path:
    sys.path.insert(0, grandparent_dir)

from parag.core import KnowledgeUnit, Fact, RAGState
from parag.embeddings import ParadoxEmbeddings
from parag.vectorstore import ParadoxVectorStore
from parag.retrieval import Retriever
from parag.reasoning import StateManager, UncertaintyCalculator
from parag.generation import DeterministicGenerator

class TestParadoxIntegration(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Initialize core Paradox components."""
        print("\n" + "="*50)
        print("INITIALIZING PARADOX INTEGRATION TESTS")
        print("="*50)
        
        cls.dim = 128
        cls.test_dir = ".test_paradox_store"
        
        # 1. Test Embeddings
        cls.embedding_model = ParadoxEmbeddings(
            embedding_dim=cls.dim,
            use_paradma=True
        )
        
        # 2. Test Vector Store
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
            
        cls.vector_store = ParadoxVectorStore(
            dimension=cls.dim,
            storage_dir=cls.test_dir
        )
        
        # 3. Test Retriever
        cls.retriever = Retriever(
            embedding_model=cls.embedding_model,
            vector_store=cls.vector_store
        )
        
        cls.sample_texts = [
            "Paradma is a self-learning mathematical system.",
            "Paradma graduates from NumPy to native Python code.",
            "The AutoLearner observes operations and records patterns.",
            "Quantum-like superposition is handled by HyperMatrix.",
            "Parag is now fully independent of PyTorch."
        ]

    @classmethod
    def tearDownClass(cls):
        """Cleanup test data."""
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
        print("\n" + "="*50)
        print("PARADOX INTEGRATION TESTS COMPLETE")
        print("="*50)

    def test_01_embeddings_v2(self):
        """Verify ParadoxEmbeddings (Paradma-powered)."""
        print("\n[Test 01] Verifying ParadoxEmbeddings...")
        text = "Test text for embedding generation"
        emb = self.embedding_model.embed(text)
        
        self.assertIsInstance(emb, np.ndarray)
        self.assertEqual(emb.shape, (self.dim,))
        # Should be normalized
        self.assertAlmostEqual(np.linalg.norm(emb), 1.0, places=5)
        print("  [OK] Embedding generated and normalized via Paradma")

    def test_02_knowledge_unit_axiom(self):
        """Verify KnowledgeUnit Axiom support."""
        print("\n[Test 02] Verifying KnowledgeUnit Axiom support...")
        unit = KnowledgeUnit(
            content="Axiom test unit",
            _use_paradma=True
        )
        
        emb = np.random.rand(self.dim).astype(np.float32)
        unit.set_embedding_from_numpy(emb)
        
        self.assertTrue(unit.is_using_paradma())
        from paradma import Axiom
        self.assertIsInstance(unit.embedding, Axiom)
        
        # Test back-conversion
        numpy_emb = unit.get_embedding_as_numpy()
        self.assertIsInstance(numpy_emb, np.ndarray)
        self.assertEqual(numpy_emb.shape, (self.dim,))
        print("  [OK] KnowledgeUnit handles Axiom conversion correctly")

    def test_03_vector_store_add_and_search(self):
        """Verify ParadoxVectorStore operations."""
        print("\n[Test 03] Verifying ParadoxVectorStore operations...")
        units = []
        for i, text in enumerate(self.sample_texts):
            unit = KnowledgeUnit(content=text, unit_id=f"unit_{i}")
            units.append(unit)
            
        ids = self.retriever.add_knowledge_units(units)
        self.assertEqual(len(ids), len(self.sample_texts))
        self.assertEqual(self.vector_store.size(), len(self.sample_texts))
        
        # Test Search
        query = "How does Paradma learn?"
        result = self.retriever.retrieve(query, top_k=2)
        
        self.assertEqual(len(result), 2)
        self.assertGreater(result.scores[0], 0.4)
        print(f"  [OK] Search successful. Top match score: {result.scores[0]:.4f}")

    def test_04_rag_state_paradma_logic(self):
        """Verify RAGState uncertainty powered by Paradma."""
        print("\n[Test 04] Verifying RAGState Paradma logic...")
        query = "Paradma learning"
        result = self.retriever.retrieve(query, top_k=3)
        
        state = RAGState.from_retrieval_result(result)
        
        # This triggers self-learning mean in _calculate_uncertainty
        self.assertGreaterEqual(state.uncertainty, 0.0)
        self.assertLessEqual(state.uncertainty, 1.0)
        
        # Verify result average score calculation via Paradma
        avg_score = result.get_average_score()
        self.assertIsInstance(avg_score, float)
        self.assertAlmostEqual(avg_score, np.mean(result.scores), places=5)
        print(f"  [OK] State uncertainty ({state.uncertainty:.4f}) and avg score calculated via Paradma")

    def test_05_conflict_detection_and_generation(self):
        """Verify conflict detection and deterministic generation."""
        print("\n[Test 05] Verifying Conflict Detection & Generation...")
        state = RAGState()
        
        # Add conflicting facts
        fact1 = KnowledgeUnit(content="The system is on.", unit_id="f1")
        fact2 = KnowledgeUnit(content="The system is not on.", unit_id="f2")
        
        state.add_knowledge_unit(fact1)
        state.add_knowledge_unit(fact2)
        
        conflicts = state.detect_conflicts()
        self.assertEqual(len(conflicts), 1)
        self.assertAlmostEqual(state.uncertainty, 0.9, places=5) # Penalty for conflict
        
        generator = DeterministicGenerator()
        response = generator.generate_from_state("Status", state)
        
        self.assertIn("conflicts detected", response)
        self.assertIn("insufficient", response.lower())
        print("  [OK] Detected conflicts and generated valid insufficient-data response")

    def test_06_vector_store_persistence(self):
        """Verify ParadoxVectorStore persistence."""
        print("\n[Test 06] Verifying Store Persistence...")
        self.vector_store.save()
        
        # Load in a new store
        new_store = ParadoxVectorStore(dimension=self.dim)
        new_store.load(self.test_dir)
        
        self.assertEqual(new_store.size(), self.vector_store.size())
        
        # Test search on loaded store
        query_vec = np.random.rand(self.dim).astype(np.float32)
        d1, i1, _ = self.vector_store.search(query_vec, k=1)
        d2, i2, _ = new_store.search(query_vec, k=1)
        
        self.assertEqual(i1[0], i2[0])
        self.assertAlmostEqual(d1[0], d2[0], places=5)
        print("  [OK] Vector store saved and reloaded with identical state")

if __name__ == '__main__':
    unittest.main()
