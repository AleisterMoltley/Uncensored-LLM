"""
Tests for Knowledge Memory System
"""

import tempfile
import shutil
from pathlib import Path
import unittest

from knowledge_memory import KnowledgeMemory

# Test constants
MAX_EXPECTED_INSIGHTS_AFTER_COMPRESSION = 100  # Upper bound for insights after compression


class TestKnowledgeMemory(unittest.TestCase):
    """Test cases for KnowledgeMemory class."""
    
    def setUp(self):
        """Create a temporary directory for testing."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.km = KnowledgeMemory(self.temp_dir)
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test that KnowledgeMemory initializes correctly."""
        stats = self.km.get_statistics()
        self.assertEqual(stats["total_pdfs_processed"], 0)
        self.assertEqual(stats["total_insights"], 0)
        self.assertEqual(stats["total_arguments"], 0)
    
    def test_add_core_belief(self):
        """Test adding a core belief."""
        self.km.add_core_belief("Logic should guide decisions", "test", 5)
        stats = self.km.get_statistics()
        self.assertEqual(stats["total_core_beliefs"], 1)
    
    def test_extract_knowledge_from_chunks(self):
        """Test extracting knowledge from text chunks."""
        chunks = [
            "Philosophy is the study of fundamental questions about existence. "
            "It is important to understand the nature of knowledge.",
            "Science relies on empirical evidence. Research shows that "
            "experiments must be reproducible to be valid.",
            "Technology has transformed how we communicate. Digital tools "
            "have become essential in modern society."
        ]
        
        result = self.km.extract_knowledge_from_chunks(
            chunks,
            "test_document.pdf",
            "abc123hash"
        )
        
        self.assertEqual(result["status"], "processed")
        self.assertGreater(result["insights_extracted"], 0)
        
        stats = self.km.get_statistics()
        self.assertEqual(stats["total_pdfs_processed"], 1)
        self.assertGreater(stats["total_insights"], 0)
    
    def test_duplicate_pdf_detection(self):
        """Test that duplicate PDFs are not processed twice."""
        chunks = ["Some important text about science."]
        
        result1 = self.km.extract_knowledge_from_chunks(chunks, "doc.pdf", "hash123")
        result2 = self.km.extract_knowledge_from_chunks(chunks, "doc.pdf", "hash123")
        
        self.assertEqual(result1["status"], "processed")
        self.assertEqual(result2["status"], "already_processed")
        
        stats = self.km.get_statistics()
        self.assertEqual(stats["total_pdfs_processed"], 1)
    
    def test_get_relevant_knowledge(self):
        """Test retrieving relevant knowledge for a query."""
        # Add some knowledge first
        chunks = [
            "Artificial intelligence is transforming industries. AI systems "
            "can learn from data and make predictions.",
            "Machine learning is a subset of AI. It uses algorithms to "
            "identify patterns in data."
        ]
        self.km.extract_knowledge_from_chunks(chunks, "ai_book.pdf", "aihash")
        
        # Query for relevant knowledge
        knowledge = self.km.get_relevant_knowledge("artificial intelligence")
        
        self.assertIn("insights", knowledge)
        self.assertIn("arguments", knowledge)
        self.assertIn("core_beliefs", knowledge)
    
    def test_format_knowledge_for_prompt(self):
        """Test formatting knowledge for prompt injection."""
        self.km.add_core_belief("Be rational and logical", "test", 10)
        
        chunks = ["Science is fundamental to understanding. Research shows that "
                  "evidence-based thinking leads to better decisions."]
        self.km.extract_knowledge_from_chunks(chunks, "doc.pdf", "hash456")
        
        prompt_context = self.km.format_knowledge_for_prompt("science")
        
        # Should contain some formatted knowledge
        self.assertIsInstance(prompt_context, str)
        if prompt_context:  # If there's relevant knowledge
            self.assertIn("LEARNED KNOWLEDGE", prompt_context)
    
    def test_compare_arguments(self):
        """Test comparing arguments about a topic."""
        chunks = [
            "Democracy is effective because it allows citizens to participate. "
            "Therefore, democratic systems tend to be more stable.",
            "Some argue that democracy can be slow because of the need for "
            "consensus. Consequently, urgent decisions may be delayed."
        ]
        self.km.extract_knowledge_from_chunks(chunks, "politics.pdf", "polhash")
        
        arguments = self.km.compare_arguments("democracy")
        self.assertIsInstance(arguments, list)
    
    def test_export_import(self):
        """Test exporting and importing knowledge."""
        self.km.add_core_belief("Test belief", "test", 5)
        chunks = ["Important information about history."]
        self.km.extract_knowledge_from_chunks(chunks, "history.pdf", "histhash")
        
        # Export
        exported = self.km.export_knowledge()
        self.assertIn("topics", exported)
        self.assertIn("core_beliefs", exported)
        
        # Create new memory and import
        km2 = KnowledgeMemory(self.temp_dir / "km2")
        km2.import_knowledge(exported)
        
        stats2 = km2.get_statistics()
        self.assertGreater(stats2["total_core_beliefs"], 0)
    
    def test_clear(self):
        """Test clearing knowledge memory."""
        self.km.add_core_belief("Test", "test", 1)
        self.km.clear()
        
        stats = self.km.get_statistics()
        self.assertEqual(stats["total_pdfs_processed"], 0)
        self.assertEqual(stats["total_core_beliefs"], 0)
    
    def test_persistence(self):
        """Test that knowledge persists across instances."""
        self.km.add_core_belief("Persistent belief", "test", 5)
        
        # Create new instance with same directory
        km2 = KnowledgeMemory(self.temp_dir)
        stats = km2.get_statistics()
        
        self.assertEqual(stats["total_core_beliefs"], 1)
    
    def test_compression_on_many_insights(self):
        """Test that compression happens when insights exceed threshold."""
        # Create many insights
        for i in range(30):
            chunks = [f"Important fact number {i}. This is significant because "
                      f"it demonstrates point {i}."]
            self.km.extract_knowledge_from_chunks(
                chunks,
                f"doc_{i}.pdf",
                f"hash_{i}"
            )
        
        stats = self.km.get_statistics()
        # Should have processed all PDFs
        self.assertEqual(stats["total_pdfs_processed"], 30)
        # Insights should be bounded due to compression
        self.assertLessEqual(stats["total_insights"], MAX_EXPECTED_INSIGHTS_AFTER_COMPRESSION)


class TestKnowledgeMemorySize(unittest.TestCase):
    """Test memory size management."""
    
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        # Configure with small max size for testing
        config = {"max_knowledge_memory_mb": 0.001}  # 1 KB limit
        self.km = KnowledgeMemory(self.temp_dir, config)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_compression_triggered_on_size_limit(self):
        """Test that compression is triggered when size limit is exceeded."""
        # Add lots of content
        for i in range(20):
            chunks = [
                f"This is a long text chunk number {i} with lots of content "
                f"that should eventually trigger compression. The system should "
                f"automatically compress when the memory file exceeds the "
                f"configured size limit. This text is intentionally verbose "
                f"to add more bytes to the memory file. Fact {i} is important."
            ]
            self.km.extract_knowledge_from_chunks(
                chunks,
                f"large_doc_{i}.pdf",
                f"large_hash_{i}"
            )
        
        stats = self.km.get_statistics()
        # Compression should have been triggered
        self.assertGreater(stats["compressions_performed"], 0)


if __name__ == "__main__":
    unittest.main()
