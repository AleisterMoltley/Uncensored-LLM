"""
Tests for Taboo Manager System
"""

import tempfile
import shutil
from pathlib import Path
import unittest

from server import TabooManager


class TestTabooManager(unittest.TestCase):
    """Test cases for TabooManager class."""
    
    def setUp(self):
        """Create a temporary directory for testing."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.tm = TabooManager(self.temp_dir)
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test that TabooManager initializes correctly."""
        stats = self.tm.get_statistics()
        self.assertEqual(stats["total_taboos"], 0)
        self.assertEqual(stats["active_taboos"], 0)
    
    def test_add_taboo(self):
        """Test adding a taboo."""
        taboo = self.tm.add_taboo("No personal questions", "topic")
        
        self.assertIsNotNone(taboo)
        self.assertIn("id", taboo)
        self.assertEqual(taboo["description"], "No personal questions")
        self.assertEqual(taboo["category"], "topic")
        self.assertTrue(taboo["active"])
        
        stats = self.tm.get_statistics()
        self.assertEqual(stats["total_taboos"], 1)
        self.assertEqual(stats["active_taboos"], 1)
    
    def test_add_taboo_default_category(self):
        """Test adding a taboo with default category."""
        taboo = self.tm.add_taboo("No explicit content")
        self.assertEqual(taboo["category"], "general")
    
    def test_list_taboos(self):
        """Test listing all taboos."""
        self.tm.add_taboo("Taboo 1", "topic")
        self.tm.add_taboo("Taboo 2", "content")
        self.tm.add_taboo("Taboo 3", "behavior")
        
        taboos = self.tm.list_taboos()
        self.assertEqual(len(taboos), 3)
    
    def test_list_taboos_active_only(self):
        """Test listing only active taboos."""
        t1 = self.tm.add_taboo("Taboo 1", "topic")
        self.tm.add_taboo("Taboo 2", "content")
        
        # Deactivate first taboo
        self.tm.toggle_taboo(t1["id"])
        
        active_taboos = self.tm.list_taboos(active_only=True)
        self.assertEqual(len(active_taboos), 1)
        self.assertEqual(active_taboos[0]["description"], "Taboo 2")
    
    def test_remove_taboo(self):
        """Test removing a taboo."""
        taboo = self.tm.add_taboo("To be removed", "topic")
        
        result = self.tm.remove_taboo(taboo["id"])
        self.assertTrue(result)
        
        taboos = self.tm.list_taboos()
        self.assertEqual(len(taboos), 0)
    
    def test_remove_nonexistent_taboo(self):
        """Test removing a taboo that doesn't exist."""
        result = self.tm.remove_taboo("nonexistent-id")
        self.assertFalse(result)
    
    def test_toggle_taboo(self):
        """Test toggling a taboo's active status."""
        taboo = self.tm.add_taboo("Toggle me", "topic")
        
        # First toggle - should become inactive
        result = self.tm.toggle_taboo(taboo["id"])
        self.assertTrue(result)
        
        taboos = self.tm.list_taboos()
        self.assertFalse(taboos[0]["active"])
        
        # Second toggle - should become active again
        result = self.tm.toggle_taboo(taboo["id"])
        self.assertTrue(result)
        
        taboos = self.tm.list_taboos()
        self.assertTrue(taboos[0]["active"])
    
    def test_toggle_nonexistent_taboo(self):
        """Test toggling a taboo that doesn't exist."""
        result = self.tm.toggle_taboo("nonexistent-id")
        self.assertFalse(result)
    
    def test_get_active_taboos_for_prompt_empty(self):
        """Test getting prompt format with no taboos."""
        prompt = self.tm.get_active_taboos_for_prompt()
        self.assertEqual(prompt, "")
    
    def test_get_active_taboos_for_prompt(self):
        """Test getting prompt format with taboos."""
        self.tm.add_taboo("No family discussions", "topic")
        self.tm.add_taboo("No violent content", "content")
        
        prompt = self.tm.get_active_taboos_for_prompt()
        
        self.assertIn("WICHTIGE EINSCHRÄNKUNGEN", prompt)
        self.assertIn("No family discussions", prompt)
        self.assertIn("No violent content", prompt)
        self.assertIn("AUSDRÜCKLICH VERBOTEN", prompt)
    
    def test_get_active_taboos_for_prompt_excludes_inactive(self):
        """Test that inactive taboos are excluded from prompt."""
        t1 = self.tm.add_taboo("Active taboo", "topic")
        t2 = self.tm.add_taboo("Inactive taboo", "topic")
        
        self.tm.toggle_taboo(t2["id"])
        
        prompt = self.tm.get_active_taboos_for_prompt()
        
        self.assertIn("Active taboo", prompt)
        self.assertNotIn("Inactive taboo", prompt)
    
    def test_clear_all(self):
        """Test clearing all taboos."""
        self.tm.add_taboo("Taboo 1", "topic")
        self.tm.add_taboo("Taboo 2", "content")
        self.tm.add_taboo("Taboo 3", "behavior")
        
        self.tm.clear_all()
        
        taboos = self.tm.list_taboos()
        self.assertEqual(len(taboos), 0)
    
    def test_persistence(self):
        """Test that taboos persist after reload."""
        self.tm.add_taboo("Persistent taboo", "topic")
        
        # Create new instance pointing to same directory
        tm2 = TabooManager(self.temp_dir)
        
        taboos = tm2.list_taboos()
        self.assertEqual(len(taboos), 1)
        self.assertEqual(taboos[0]["description"], "Persistent taboo")
    
    def test_statistics(self):
        """Test getting statistics."""
        t1 = self.tm.add_taboo("Taboo 1", "topic")
        self.tm.add_taboo("Taboo 2", "content")
        self.tm.add_taboo("Taboo 3", "behavior")
        
        self.tm.toggle_taboo(t1["id"])
        
        stats = self.tm.get_statistics()
        
        self.assertEqual(stats["total_taboos"], 3)
        self.assertEqual(stats["active_taboos"], 2)
        self.assertEqual(stats["inactive_taboos"], 1)
        self.assertIn("last_updated", stats)


if __name__ == "__main__":
    unittest.main()
