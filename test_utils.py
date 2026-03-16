"""
Tests for the utils module, specifically the PersistentStorage class.
"""

import tempfile
import shutil
from pathlib import Path
import unittest

from utils import PersistentStorage


class TestPersistentStorage(unittest.TestCase):
    """Test cases for PersistentStorage class."""
    
    def setUp(self):
        """Create a temporary directory for testing."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.storage_file = self.temp_dir / "test_data.json.gz"
        self.storage = PersistentStorage(self.storage_file)
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_nonexistent_file_returns_default(self):
        """Test that loading a nonexistent file returns the default value."""
        result = self.storage.load(default={"key": "default_value"})
        self.assertEqual(result, {"key": "default_value"})
    
    def test_load_nonexistent_file_returns_empty_dict_without_default(self):
        """Test that loading a nonexistent file returns empty dict when no default."""
        result = self.storage.load()
        self.assertEqual(result, {})
    
    def test_save_and_load(self):
        """Test saving and loading data."""
        data = {"name": "test", "value": 123, "nested": {"a": 1, "b": 2}}
        self.storage.save(data)
        
        # Create new storage instance to load fresh
        storage2 = PersistentStorage(self.storage_file)
        loaded = storage2.load()
        
        self.assertEqual(loaded, data)
    
    def test_save_returns_true_on_success(self):
        """Test that save returns True on success."""
        result = self.storage.save({"test": "data"})
        self.assertTrue(result)
    
    def test_exists_returns_false_for_new_storage(self):
        """Test exists() returns False for new storage."""
        self.assertFalse(self.storage.exists())
    
    def test_exists_returns_true_after_save(self):
        """Test exists() returns True after saving."""
        self.storage.save({"test": "data"})
        self.assertTrue(self.storage.exists())
    
    def test_get_size_mb_returns_zero_for_nonexistent(self):
        """Test get_size_mb() returns 0 for nonexistent file."""
        self.assertEqual(self.storage.get_size_mb(), 0.0)
    
    def test_get_size_mb_returns_positive_after_save(self):
        """Test get_size_mb() returns positive value after saving."""
        self.storage.save({"test": "data"})
        self.assertGreater(self.storage.get_size_mb(), 0.0)
    
    def test_delete_removes_file(self):
        """Test delete() removes the file."""
        self.storage.save({"test": "data"})
        self.assertTrue(self.storage.exists())
        
        result = self.storage.delete()
        
        self.assertTrue(result)
        self.assertFalse(self.storage.exists())
    
    def test_delete_returns_false_for_nonexistent(self):
        """Test delete() returns False for nonexistent file."""
        result = self.storage.delete()
        self.assertFalse(result)
    
    def test_unicode_data(self):
        """Test saving and loading Unicode data."""
        data = {
            "german": "Übung macht den Meister",
            "japanese": "こんにちは",
            "emoji": "🎉🚀"
        }
        self.storage.save(data)
        loaded = self.storage.load()
        
        self.assertEqual(loaded, data)
    
    def test_large_data(self):
        """Test saving and loading large data."""
        data = {"items": [f"item_{i}" for i in range(1000)]}
        self.storage.save(data)
        loaded = self.storage.load()
        
        self.assertEqual(len(loaded["items"]), 1000)
    
    def test_compression_callback_triggered_on_size_exceeded(self):
        """Test that compression callback is triggered when size exceeds limit."""
        compression_called = {"count": 0}
        
        def compress_callback(data):
            compression_called["count"] += 1
            # Return compressed version (just remove most items)
            return {"items": data["items"][:10]}
        
        storage = PersistentStorage(
            self.storage_file,
            max_size_mb=0.0001,  # Very small limit to trigger callback
            on_size_exceeded=compress_callback
        )
        
        # Save large data that exceeds the limit
        large_data = {"items": [f"item_{i}" * 100 for i in range(100)]}
        storage.save(large_data)
        
        self.assertGreater(compression_called["count"], 0)
    
    def test_compression_callback_not_triggered_within_limit(self):
        """Test that compression callback is not triggered when within limit."""
        compression_called = {"count": 0}
        
        def compress_callback(data):
            compression_called["count"] += 1
            return data
        
        storage = PersistentStorage(
            self.storage_file,
            max_size_mb=10,  # Large limit
            on_size_exceeded=compress_callback
        )
        
        # Save small data that won't exceed limit
        storage.save({"test": "data"})
        
        self.assertEqual(compression_called["count"], 0)
    
    def test_corrupted_file_returns_default(self):
        """Test that a corrupted file returns the default value."""
        # Write corrupted data
        with open(self.storage_file, 'wb') as f:
            f.write(b"not valid gzip data")
        
        result = self.storage.load(default={"fallback": True})
        
        self.assertEqual(result, {"fallback": True})


class TestPersistentStorageEdgeCases(unittest.TestCase):
    """Edge case tests for PersistentStorage."""
    
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_empty_dict(self):
        """Test saving and loading an empty dict."""
        storage = PersistentStorage(self.temp_dir / "empty.json.gz")
        storage.save({})
        loaded = storage.load()
        
        self.assertEqual(loaded, {})
    
    def test_nested_structures(self):
        """Test saving and loading deeply nested structures."""
        data = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": ["a", "b", "c"]
                    }
                }
            }
        }
        storage = PersistentStorage(self.temp_dir / "nested.json.gz")
        storage.save(data)
        loaded = storage.load()
        
        self.assertEqual(loaded, data)
    
    def test_special_characters_in_path(self):
        """Test storage with special characters in path (if filesystem supports it)."""
        storage = PersistentStorage(self.temp_dir / "test-file_2024.json.gz")
        storage.save({"test": True})
        loaded = storage.load()
        
        self.assertEqual(loaded, {"test": True})


if __name__ == "__main__":
    unittest.main()
