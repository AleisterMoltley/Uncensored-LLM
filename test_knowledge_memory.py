"""
Tests for Knowledge Memory System
"""

import tempfile
import shutil
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock

from knowledge_memory import (
    KnowledgeMemory,
    InsightModel,
    ArgumentModel,
    CoreBeliefModel,
    StatisticsModel,
    MemoryModel,
    KnowledgeMemoryConfig,
    ExtractionResult,
    RelevantKnowledgeResult,
    StatisticsResult,
    EmbeddingCache,
    RedisConfig,
)
from pydantic import ValidationError

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


class TestPydanticModels(unittest.TestCase):
    """Test cases for Pydantic model validation."""
    
    def test_insight_model_valid(self):
        """Test valid InsightModel creation."""
        insight = InsightModel(
            content="This is a valid insight",
            source="test_source",
            weight=3
        )
        self.assertEqual(insight.content, "This is a valid insight")
        self.assertEqual(insight.source, "test_source")
        self.assertEqual(insight.weight, 3)
    
    def test_insight_model_whitespace_stripping(self):
        """Test that InsightModel strips whitespace from content."""
        insight = InsightModel(content="  test content  ", source="test")
        self.assertEqual(insight.content, "test content")
    
    def test_insight_model_invalid_empty_content(self):
        """Test that InsightModel rejects empty content."""
        with self.assertRaises(ValidationError):
            InsightModel(content="", source="test")
    
    def test_insight_model_invalid_zero_weight(self):
        """Test that InsightModel rejects weight less than 1 (zero)."""
        with self.assertRaises(ValidationError):
            InsightModel(content="valid", weight=0)
    
    def test_insight_model_invalid_negative_weight(self):
        """Test that InsightModel rejects negative weight."""
        with self.assertRaises(ValidationError):
            InsightModel(content="valid", weight=-1)
    
    def test_argument_model_valid(self):
        """Test valid ArgumentModel creation."""
        argument = ArgumentModel(
            claim="This is because of reason X",
            source="test_pdf",
            strength=2,
            context=["Previous sentence", "Next sentence"]
        )
        self.assertEqual(argument.claim, "This is because of reason X")
        self.assertEqual(argument.source, "test_pdf")
        self.assertEqual(argument.strength, 2)
        self.assertEqual(len(argument.context), 2)
    
    def test_argument_model_whitespace_stripping(self):
        """Test that ArgumentModel strips whitespace from claim."""
        argument = ArgumentModel(claim="  test claim  ", source="test")
        self.assertEqual(argument.claim, "test claim")
    
    def test_core_belief_model_valid(self):
        """Test valid CoreBeliefModel creation."""
        belief = CoreBeliefModel(
            content="Logic should guide all decisions",
            source="user",
            weight=10
        )
        self.assertEqual(belief.content, "Logic should guide all decisions")
        self.assertEqual(belief.source, "user")
        self.assertEqual(belief.weight, 10)
    
    def test_statistics_model_defaults(self):
        """Test StatisticsModel default values."""
        stats = StatisticsModel()
        self.assertEqual(stats.total_pdfs_processed, 0)
        self.assertEqual(stats.total_insights_extracted, 0)
        self.assertEqual(stats.compressions_performed, 0)
    
    def test_statistics_model_invalid_negative(self):
        """Test that StatisticsModel rejects negative values."""
        with self.assertRaises(ValidationError):
            StatisticsModel(total_pdfs_processed=-1)
    
    def test_memory_model_defaults(self):
        """Test MemoryModel default values."""
        memory = MemoryModel()
        self.assertEqual(memory.version, "1.0")
        self.assertEqual(len(memory.topics), 0)
        self.assertEqual(len(memory.core_beliefs), 0)
        self.assertEqual(len(memory.arguments), 0)
        self.assertEqual(len(memory.source_hashes), 0)
    
    def test_memory_model_to_dict(self):
        """Test MemoryModel to_dict conversion."""
        memory = MemoryModel(
            version="1.0",
            statistics=StatisticsModel(total_pdfs_processed=5),
            core_beliefs=[
                CoreBeliefModel(content="Test belief", source="user", weight=5)
            ]
        )
        
        data = memory.to_dict()
        self.assertIsInstance(data, dict)
        self.assertEqual(data["version"], "1.0")
        self.assertEqual(data["statistics"]["total_pdfs_processed"], 5)
        self.assertEqual(len(data["core_beliefs"]), 1)
    
    def test_memory_model_from_dict(self):
        """Test MemoryModel from_dict parsing."""
        data = {
            "version": "1.0",
            "created": "2024-01-01T00:00:00",
            "updated": "2024-01-01T00:00:00",
            "statistics": {
                "total_pdfs_processed": 3,
                "total_insights_extracted": 10,
                "compressions_performed": 1
            },
            "topics": {
                "science": [
                    {"content": "Science is important", "source": "test", "weight": 2}
                ]
            },
            "core_beliefs": [
                {"content": "Be logical", "source": "user", "weight": 5}
            ],
            "arguments": [
                {"claim": "Because of X", "source": "doc", "type": "logical_argument", "strength": 1}
            ],
            "source_hashes": ["hash123"]
        }
        
        memory = MemoryModel.from_dict(data)
        self.assertEqual(memory.version, "1.0")
        self.assertEqual(memory.statistics.total_pdfs_processed, 3)
        self.assertEqual(len(memory.topics["science"]), 1)
        self.assertEqual(len(memory.core_beliefs), 1)
        self.assertEqual(len(memory.arguments), 1)
        self.assertEqual(len(memory.source_hashes), 1)
    
    def test_memory_model_roundtrip(self):
        """Test MemoryModel can roundtrip through to_dict and from_dict."""
        original = MemoryModel(
            version="1.0",
            statistics=StatisticsModel(total_pdfs_processed=5, compressions_performed=2),
            topics={
                "philosophy": [
                    InsightModel(content="Philosophy is the study of fundamental questions", source="book1"),
                    InsightModel(content="Ethics explores moral principles", source="book2", weight=3)
                ]
            },
            core_beliefs=[
                CoreBeliefModel(content="Reason is essential", source="user", weight=8)
            ],
            arguments=[
                ArgumentModel(claim="Therefore we must conclude", source="paper1", strength=2)
            ],
            source_hashes=["abc123", "def456"]
        )
        
        # Convert to dict and back
        data = original.to_dict()
        restored = MemoryModel.from_dict(data)
        
        # Verify all fields match
        self.assertEqual(original.version, restored.version)
        self.assertEqual(original.statistics.total_pdfs_processed, restored.statistics.total_pdfs_processed)
        self.assertEqual(len(original.topics), len(restored.topics))
        self.assertEqual(len(original.core_beliefs), len(restored.core_beliefs))
        self.assertEqual(len(original.arguments), len(restored.arguments))
        self.assertEqual(original.source_hashes, restored.source_hashes)


class TestTypedDicts(unittest.TestCase):
    """Test cases for TypedDict return types."""
    
    def setUp(self):
        """Create a temporary directory for testing."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.km = KnowledgeMemory(self.temp_dir)
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_extraction_result_type(self):
        """Test that extract_knowledge_from_chunks returns proper TypedDict."""
        chunks = ["This is important information about science and research."]
        result = self.km.extract_knowledge_from_chunks(
            chunks, "test.pdf", "test_hash"
        )
        
        self.assertIn("status", result)
        self.assertIn("filename", result)
        self.assertEqual(result["status"], "processed")
        self.assertEqual(result["filename"], "test.pdf")
        self.assertIn("insights_extracted", result)
        self.assertIn("arguments_extracted", result)
    
    def test_already_processed_result_type(self):
        """Test AlreadyProcessedResult when duplicate PDF is processed."""
        chunks = ["Test content about science."]
        self.km.extract_knowledge_from_chunks(chunks, "test.pdf", "hash123")
        
        # Second call with same hash
        result = self.km.extract_knowledge_from_chunks(chunks, "test.pdf", "hash123")
        
        self.assertEqual(result["status"], "already_processed")
        self.assertEqual(result["filename"], "test.pdf")
    
    def test_statistics_result_type(self):
        """Test that get_statistics returns proper StatisticsResult."""
        stats = self.km.get_statistics()
        
        self.assertIn("version", stats)
        self.assertIn("created", stats)
        self.assertIn("updated", stats)
        self.assertIn("total_pdfs_processed", stats)
        self.assertIn("total_insights", stats)
        self.assertIn("total_arguments", stats)
        self.assertIn("total_core_beliefs", stats)
        self.assertIn("topics_count", stats)
        self.assertIn("topics", stats)
        self.assertIn("compressions_performed", stats)
        self.assertIn("file_size_bytes", stats)
        self.assertIn("file_size_mb", stats)
    
    def test_relevant_knowledge_result_type(self):
        """Test that get_relevant_knowledge returns proper RelevantKnowledgeResult."""
        self.km.add_core_belief("Reason is important", "test")
        result = self.km.get_relevant_knowledge("reason")
        
        self.assertIn("core_beliefs", result)
        self.assertIn("insights", result)
        self.assertIn("arguments", result)
        self.assertIn("topics_matched", result)
        self.assertIsInstance(result["core_beliefs"], list)
        self.assertIsInstance(result["insights"], list)
        self.assertIsInstance(result["arguments"], list)
        self.assertIsInstance(result["topics_matched"], list)
    
    def test_config_typed_dict(self):
        """Test KnowledgeMemoryConfig TypedDict usage."""
        config: KnowledgeMemoryConfig = {
            "max_knowledge_memory_mb": 5.0,
            "max_insights_per_topic": 25,
            "summary_threshold": 10
        }
        km = KnowledgeMemory(self.temp_dir / "custom_config", config)
        
        self.assertEqual(km.max_size_mb, 5.0)
        self.assertEqual(km.max_insights_per_topic, 25)
        self.assertEqual(km.summary_threshold, 10)


class TestValidateMemory(unittest.TestCase):
    """Test cases for memory validation."""
    
    def setUp(self):
        """Create a temporary directory for testing."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.km = KnowledgeMemory(self.temp_dir)
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_validate_memory_valid(self):
        """Test validation of a valid memory state."""
        self.km.add_core_belief("Test belief", "test")
        self.assertTrue(self.km._validate_memory())
    
    def test_validate_memory_after_extraction(self):
        """Test validation after knowledge extraction."""
        chunks = ["Science is fundamental to progress. Research shows important results."]
        self.km.extract_knowledge_from_chunks(chunks, "doc.pdf", "hash123")
        self.assertTrue(self.km._validate_memory())


class TestEmbeddingCache(unittest.TestCase):
    """Test cases for EmbeddingCache class."""
    
    def test_cache_disabled_by_default(self):
        """Test that cache is disabled when no config provided."""
        cache = EmbeddingCache()
        self.assertFalse(cache.enabled)
        self.assertFalse(cache.is_available())
    
    def test_cache_disabled_explicitly(self):
        """Test that cache respects enabled=False config."""
        config: RedisConfig = {"enabled": False, "url": "redis://localhost:6379/0"}
        cache = EmbeddingCache(config)
        self.assertFalse(cache.enabled)
        self.assertFalse(cache.is_available())
    
    def test_get_returns_none_when_disabled(self):
        """Test that get returns None when cache is disabled."""
        cache = EmbeddingCache()
        result = cache.get("test text")
        self.assertIsNone(result)
    
    def test_set_returns_false_when_disabled(self):
        """Test that set returns False when cache is disabled."""
        cache = EmbeddingCache()
        result = cache.set("test text", [1.0, 2.0, 3.0])
        self.assertFalse(result)
    
    def test_get_batch_returns_none_when_disabled(self):
        """Test that get_batch returns None for all texts when disabled."""
        cache = EmbeddingCache()
        texts = ["text1", "text2"]
        results = cache.get_batch(texts)
        self.assertEqual(results, {"text1": None, "text2": None})
    
    def test_set_batch_returns_zero_when_disabled(self):
        """Test that set_batch returns 0 when disabled."""
        cache = EmbeddingCache()
        embeddings = {"text1": [1.0, 2.0], "text2": [3.0, 4.0]}
        result = cache.set_batch(embeddings)
        self.assertEqual(result, 0)
    
    def test_clear_returns_zero_when_disabled(self):
        """Test that clear returns 0 when disabled."""
        cache = EmbeddingCache()
        result = cache.clear()
        self.assertEqual(result, 0)
    
    def test_get_stats_when_disabled(self):
        """Test that get_stats returns correct info when disabled."""
        cache = EmbeddingCache()
        stats = cache.get_stats()
        self.assertFalse(stats["enabled"])
        self.assertFalse(stats["available"])
        self.assertEqual(stats["cached_embeddings"], 0)
    
    def test_text_to_cache_key_deterministic(self):
        """Test that cache key generation is deterministic."""
        cache = EmbeddingCache()
        key1 = cache._text_to_cache_key("test text")
        key2 = cache._text_to_cache_key("test text")
        self.assertEqual(key1, key2)
    
    def test_text_to_cache_key_different_texts(self):
        """Test that different texts produce different keys."""
        cache = EmbeddingCache()
        key1 = cache._text_to_cache_key("text one")
        key2 = cache._text_to_cache_key("text two")
        self.assertNotEqual(key1, key2)
    
    def test_cache_key_format(self):
        """Test that cache key has correct prefix."""
        cache = EmbeddingCache()
        key = cache._text_to_cache_key("test")
        self.assertTrue(key.startswith("embedding:"))
    
    def test_redis_connection_success_mock(self):
        """Test successful Redis connection with mock."""
        # Create a cache with enabled=True
        config: RedisConfig = {"enabled": True, "url": "redis://localhost:6379/0"}
        cache = EmbeddingCache(config)
        
        # Manually set up the mock state
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_client.get.return_value = '[1.0, 2.0, 3.0]'
        
        cache._redis_client = mock_client
        cache._redis_available = True
        
        # Now test that operations work
        self.assertTrue(cache.is_available())
    
    def test_redis_connection_failure_handled(self):
        """Test Redis connection failure is handled gracefully."""
        # When Redis isn't installed or connection fails, cache should be disabled
        config: RedisConfig = {"enabled": True, "url": "redis://localhost:6379/0"}
        cache = EmbeddingCache(config)
        
        # The _connect_redis method should have been called and failed
        # (either because redis isn't installed or connection failed)
        # The cache should gracefully handle this
        self.assertFalse(cache.is_available())
    
    def test_cache_get_with_mock_client(self):
        """Test successful cache get operation with mock client."""
        config: RedisConfig = {"enabled": True, "url": "redis://localhost:6379/0"}
        cache = EmbeddingCache(config)
        
        # Set up mock client
        mock_client = MagicMock()
        mock_client.get.return_value = '[1.0, 2.0, 3.0]'
        cache._redis_client = mock_client
        cache._redis_available = True
        
        result = cache.get("test text")
        self.assertEqual(result, [1.0, 2.0, 3.0])
    
    def test_cache_set_with_mock_client(self):
        """Test successful cache set operation with mock client."""
        config: RedisConfig = {"enabled": True, "url": "redis://localhost:6379/0"}
        cache = EmbeddingCache(config)
        
        # Set up mock client
        mock_client = MagicMock()
        cache._redis_client = mock_client
        cache._redis_available = True
        
        result = cache.set("test text", [1.0, 2.0, 3.0])
        self.assertTrue(result)
        mock_client.setex.assert_called_once()


class TestKnowledgeMemoryWithEmbeddingCache(unittest.TestCase):
    """Test cases for KnowledgeMemory with EmbeddingCache integration."""
    
    def setUp(self):
        """Create a temporary directory for testing."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_embedding_cache_initialized(self):
        """Test that embedding cache is initialized."""
        km = KnowledgeMemory(self.temp_dir)
        cache = km.get_embedding_cache()
        self.assertIsInstance(cache, EmbeddingCache)
    
    def test_embedding_cache_with_config(self):
        """Test embedding cache with Redis config."""
        config: KnowledgeMemoryConfig = {
            "redis": {
                "enabled": False,
                "url": "redis://custom:6379/0",
                "embedding_cache_ttl": 3600
            }
        }
        km = KnowledgeMemory(self.temp_dir, config)
        cache = km.get_embedding_cache()
        
        self.assertFalse(cache.enabled)
        self.assertEqual(cache.redis_url, "redis://custom:6379/0")
        self.assertEqual(cache.ttl, 3600)
    
    def test_get_cached_embedding_when_disabled(self):
        """Test get_cached_embedding returns None when cache disabled."""
        km = KnowledgeMemory(self.temp_dir)
        result = km.get_cached_embedding("test text")
        self.assertIsNone(result)
    
    def test_cache_embedding_when_disabled(self):
        """Test cache_embedding returns False when cache disabled."""
        km = KnowledgeMemory(self.temp_dir)
        result = km.cache_embedding("test text", [1.0, 2.0])
        self.assertFalse(result)
    
    def test_get_embedding_cache_stats(self):
        """Test get_embedding_cache_stats returns stats dict."""
        km = KnowledgeMemory(self.temp_dir)
        stats = km.get_embedding_cache_stats()
        
        self.assertIn("enabled", stats)
        self.assertIn("available", stats)
        self.assertIn("cached_embeddings", stats)
    
    def test_statistics_includes_embedding_cache(self):
        """Test that statistics includes embedding cache info."""
        km = KnowledgeMemory(self.temp_dir)
        stats = km.get_statistics()
        
        self.assertIn("embedding_cache", stats)
        self.assertIsInstance(stats["embedding_cache"], dict)


class TestChunkSizeConfiguration(unittest.TestCase):
    """Test cases for chunk size configuration."""
    
    def setUp(self):
        """Create a temporary directory for testing."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_default_chunk_size(self):
        """Test default chunk size when not configured."""
        km = KnowledgeMemory(self.temp_dir)
        self.assertEqual(km.chunk_size, 600)
        self.assertEqual(km.chunk_overlap, 100)
    
    def test_custom_chunk_size(self):
        """Test custom chunk size from config."""
        config: KnowledgeMemoryConfig = {
            "chunk_size": 1000,
            "chunk_overlap": 200
        }
        km = KnowledgeMemory(self.temp_dir, config)
        
        self.assertEqual(km.chunk_size, 1000)
        self.assertEqual(km.chunk_overlap, 200)
    
    def test_statistics_includes_chunk_config(self):
        """Test that statistics includes chunk configuration."""
        config: KnowledgeMemoryConfig = {
            "chunk_size": 800,
            "chunk_overlap": 150
        }
        km = KnowledgeMemory(self.temp_dir, config)
        stats = km.get_statistics()
        
        self.assertEqual(stats["chunk_size"], 800)
        self.assertEqual(stats["chunk_overlap"], 150)


if __name__ == "__main__":
    unittest.main()


class TestLargePDFProcessing(unittest.TestCase):
    """Test cases for large PDF processing with caching."""
    
    def setUp(self):
        """Create a temporary directory for testing."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_large_chunks_processing(self):
        """Test processing many chunks similar to a large PDF."""
        config: KnowledgeMemoryConfig = {
            "chunk_size": 500,
            "chunk_overlap": 50
        }
        km = KnowledgeMemory(self.temp_dir, config)
        
        # Simulate a large PDF with many chunks
        num_chunks = 100
        chunks = []
        for i in range(num_chunks):
            chunk = (
                f"This is chunk number {i} containing important information. "
                f"The science shows that fact {i} is significant because "
                f"it demonstrates point {i} about the technology."
            )
            chunks.append(chunk)
        
        result = km.extract_knowledge_from_chunks(
            chunks, "large_document.pdf", "largehash123"
        )
        
        self.assertEqual(result["status"], "processed")
        self.assertGreater(result["insights_extracted"], 0)
        
        stats = km.get_statistics()
        self.assertEqual(stats["total_pdfs_processed"], 1)
        self.assertEqual(stats["chunk_size"], 500)
        self.assertEqual(stats["chunk_overlap"], 50)
    
    def test_multiple_large_pdfs(self):
        """Test processing multiple large PDFs."""
        km = KnowledgeMemory(self.temp_dir)
        
        # Process multiple "large" PDFs
        for pdf_idx in range(5):
            chunks = []
            for i in range(50):
                chunk = f"Document {pdf_idx} - Section {i}: Important content about topic {i}."
                chunks.append(chunk)
            
            result = km.extract_knowledge_from_chunks(
                chunks, f"document_{pdf_idx}.pdf", f"hash_{pdf_idx}"
            )
            self.assertEqual(result["status"], "processed")
        
        stats = km.get_statistics()
        self.assertEqual(stats["total_pdfs_processed"], 5)
    
    def test_embedding_cache_integration_mock(self):
        """Test that embedding cache is properly integrated."""
        config: KnowledgeMemoryConfig = {
            "redis": {
                "enabled": True,
                "url": "redis://localhost:6379/0",
                "embedding_cache_ttl": 3600
            }
        }
        km = KnowledgeMemory(self.temp_dir, config)
        
        # Get the cache and set up mock state
        cache = km.get_embedding_cache()
        mock_client = MagicMock()
        cache._redis_client = mock_client
        cache._redis_available = True
        
        # Test caching operation
        text = "This is test content for embedding."
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        success = km.cache_embedding(text, embedding)
        self.assertTrue(success)
        
        # Verify setex was called
        mock_client.setex.assert_called()
