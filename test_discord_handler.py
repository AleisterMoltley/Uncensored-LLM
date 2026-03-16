"""
Tests for Discord Handler and User Memory System
"""

import tempfile
import shutil
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

from discord_handler import DiscordHandler, UserMemory


class TestUserMemory(unittest.TestCase):
    """Test cases for UserMemory class."""
    
    def setUp(self):
        """Create a temporary directory for testing."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.memory_file = self.temp_dir / "test_memories.json.gz"
        self.user_memory = UserMemory(self.memory_file)
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test that UserMemory initializes correctly."""
        stats = self.user_memory.get_statistics()
        self.assertEqual(stats["total_users"], 0)
        self.assertEqual(stats["total_conversations"], 0)
        self.assertEqual(stats["total_facts"], 0)
    
    def test_get_user_memory_creates_new_entry(self):
        """Test that getting memory for new user creates entry."""
        user_mem = self.user_memory.get_user_memory(12345, 67890)
        
        self.assertEqual(user_mem["user_id"], 12345)
        self.assertEqual(user_mem["guild_id"], 67890)
        self.assertEqual(user_mem["interaction_count"], 0)
        self.assertEqual(len(user_mem["facts"]), 0)
        self.assertEqual(len(user_mem["conversations"]), 0)
    
    def test_update_user_info(self):
        """Test updating user information."""
        self.user_memory.update_user_info(
            12345, 67890,
            username="testuser",
            display_name="Test User",
            discriminator="1234"
        )
        
        user_mem = self.user_memory.get_user_memory(12345, 67890)
        self.assertEqual(user_mem["username"], "testuser")
        self.assertEqual(user_mem["display_name"], "Test User")
        self.assertEqual(user_mem["discriminator"], "1234")
    
    def test_add_conversation(self):
        """Test adding a conversation to user memory."""
        self.user_memory.add_conversation(
            12345, 67890,
            "Hello bot!",
            "Hello! How can I help you?"
        )
        
        user_mem = self.user_memory.get_user_memory(12345, 67890)
        self.assertEqual(user_mem["interaction_count"], 1)
        self.assertEqual(len(user_mem["conversations"]), 1)
        self.assertEqual(user_mem["conversations"][0]["user_message"], "Hello bot!")
        self.assertEqual(user_mem["conversations"][0]["bot_response"], "Hello! How can I help you?")
    
    def test_add_fact(self):
        """Test adding a fact about a user."""
        self.user_memory.add_fact(12345, 67890, "User likes Python")
        self.user_memory.add_fact(12345, 67890, "User lives in Berlin")
        
        user_mem = self.user_memory.get_user_memory(12345, 67890)
        self.assertEqual(len(user_mem["facts"]), 2)
        self.assertIn("User likes Python", user_mem["facts"])
        self.assertIn("User lives in Berlin", user_mem["facts"])
    
    def test_add_duplicate_fact(self):
        """Test that duplicate facts are not added."""
        self.user_memory.add_fact(12345, 67890, "User likes Python")
        self.user_memory.add_fact(12345, 67890, "User likes Python")  # duplicate
        
        user_mem = self.user_memory.get_user_memory(12345, 67890)
        self.assertEqual(len(user_mem["facts"]), 1)
    
    def test_set_preference(self):
        """Test setting a user preference."""
        self.user_memory.set_preference(12345, 67890, "language", "German")
        
        user_mem = self.user_memory.get_user_memory(12345, 67890)
        self.assertEqual(user_mem["preferences"]["language"], "German")
    
    def test_format_memory_for_prompt(self):
        """Test formatting memory for prompt injection."""
        self.user_memory.update_user_info(
            12345, 67890,
            username="testuser",
            display_name="Alice"
        )
        self.user_memory.add_fact(12345, 67890, "Likes AI research")
        self.user_memory.add_conversation(
            12345, 67890,
            "What is machine learning?",
            "Machine learning is a subset of AI..."
        )
        
        prompt = self.user_memory.format_memory_for_prompt(12345, 67890)
        
        self.assertIn("USER MEMORY", prompt)
        self.assertIn("Alice", prompt)
        self.assertIn("testuser", prompt)
        self.assertIn("Likes AI research", prompt)
    
    def test_format_empty_memory(self):
        """Test formatting memory for new user with no interactions."""
        # New user with no name or facts should still get basic info
        prompt = self.user_memory.format_memory_for_prompt(99999, 88888)
        # A new user will have basic interaction info but no name/facts
        self.assertIn("Interactions: 0", prompt)
    
    def test_get_all_users(self):
        """Test getting list of all users."""
        self.user_memory.update_user_info(111, 222, username="user1")
        self.user_memory.update_user_info(333, 444, username="user2")
        
        users = self.user_memory.get_all_users()
        self.assertEqual(len(users), 2)
    
    def test_clear_user(self):
        """Test clearing memory for a specific user."""
        self.user_memory.add_fact(12345, 67890, "Some fact")
        self.user_memory.clear_user(12345, 67890)
        
        stats = self.user_memory.get_statistics()
        self.assertEqual(stats["total_users"], 0)
    
    def test_clear_all(self):
        """Test clearing all user memories."""
        self.user_memory.add_fact(111, 222, "Fact 1")
        self.user_memory.add_fact(333, 444, "Fact 2")
        self.user_memory.clear_all()
        
        stats = self.user_memory.get_statistics()
        self.assertEqual(stats["total_users"], 0)
    
    def test_persistence(self):
        """Test that memories persist across instances."""
        self.user_memory.add_fact(12345, 67890, "Persistent fact")
        
        # Create new instance with same file
        user_memory2 = UserMemory(self.memory_file)
        user_mem = user_memory2.get_user_memory(12345, 67890)
        
        self.assertIn("Persistent fact", user_mem["facts"])
    
    def test_conversation_limit(self):
        """Test that conversations are limited to prevent unbounded growth."""
        # Add more than 50 conversations
        for i in range(60):
            self.user_memory.add_conversation(
                12345, 67890,
                f"Message {i}",
                f"Response {i}"
            )
        
        user_mem = self.user_memory.get_user_memory(12345, 67890)
        self.assertEqual(len(user_mem["conversations"]), 50)
        # Should keep the most recent conversations
        self.assertEqual(user_mem["conversations"][-1]["user_message"], "Message 59")
    
    def test_facts_limit(self):
        """Test that facts are limited to prevent unbounded growth."""
        # Add more than 20 facts
        for i in range(25):
            self.user_memory.add_fact(12345, 67890, f"Fact {i}")
        
        user_mem = self.user_memory.get_user_memory(12345, 67890)
        self.assertEqual(len(user_mem["facts"]), 20)


class TestDiscordHandler(unittest.TestCase):
    """Test cases for DiscordHandler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "discord": {
                "bot_token": "test_token_123",
                "respond_to_mentions": True,
                "respond_to_direct": True,
                "task": "Be helpful and friendly"
            }
        }
        self.llm_callback = MagicMock(return_value="This is a test response from the LLM.")
        
        # Use temp directory for data
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Patch the data directories
        with patch('discord_handler.DISCORD_DIR', self.temp_dir):
            with patch('discord_handler.HISTORY_FILE', self.temp_dir / "history.json"):
                with patch('discord_handler.USER_MEMORY_FILE', self.temp_dir / "memories.json.gz"):
                    self.handler = DiscordHandler(self.config, self.llm_callback)
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test that handler initializes correctly."""
        status = self.handler.get_status()
        self.assertTrue(status["configured"])
        self.assertFalse(status["running"])
    
    def test_configure(self):
        """Test reconfiguring the handler."""
        new_config = {
            "bot_token": "new_token",
            "task": "New task"
        }
        
        status = self.handler.configure(new_config)
        self.assertEqual(self.handler.config["task"], "New task")
    
    def test_extract_message_without_mention(self):
        """Test extracting message content without bot mention."""
        result = self.handler._extract_message_without_mention("<@123456789> What is AI?", 123456789)
        self.assertEqual(result, "What is AI?")
        
        result = self.handler._extract_message_without_mention("<@!123456789> How are you?", 123456789)
        self.assertEqual(result, "How are you?")
    
    def test_generate_response(self):
        """Test generating a response."""
        response = self.handler.generate_response(
            "What is AI?",
            12345, 67890,
            username="testuser",
            display_name="Alice"
        )
        
        self.assertEqual(response, "This is a test response from the LLM.")
        self.llm_callback.assert_called_once()
        
        # Check that user info was stored
        user_mem = self.handler.user_memory.get_user_memory(12345, 67890)
        self.assertEqual(user_mem["username"], "testuser")
        self.assertEqual(user_mem["display_name"], "Alice")
    
    def test_process_message_dm(self):
        """Test processing a direct message."""
        message_data = {
            "message_id": 1,
            "channel_id": 67890,
            "guild_id": 0,
            "user_id": 12345,
            "text": "Hello!",
            "is_dm": True,
            "is_mentioned": False,
            "username": "testuser",
            "display_name": "Alice",
            "discriminator": None,
            "bot_user_id": 99999
        }
        
        result = self.handler.process_message(message_data)
        
        self.assertTrue(result["should_respond"])
        self.assertIn("generated_response", result)
    
    def test_process_message_guild_without_mention(self):
        """Test processing a guild message without bot mention."""
        message_data = {
            "message_id": 1,
            "channel_id": 67890,
            "guild_id": 11111,
            "user_id": 12345,
            "text": "Hello everyone!",
            "is_dm": False,
            "is_mentioned": False,
            "username": "testuser",
            "display_name": "Alice",
            "discriminator": None,
            "bot_user_id": 99999
        }
        
        result = self.handler.process_message(message_data)
        
        self.assertFalse(result["should_respond"])
    
    def test_process_message_guild_with_mention(self):
        """Test processing a guild message with bot mention."""
        message_data = {
            "message_id": 1,
            "channel_id": 67890,
            "guild_id": 11111,
            "user_id": 12345,
            "text": "<@99999> What is AI?",
            "is_dm": False,
            "is_mentioned": True,
            "username": "testuser",
            "display_name": "Alice",
            "discriminator": None,
            "bot_user_id": 99999
        }
        
        result = self.handler.process_message(message_data)
        
        self.assertTrue(result["should_respond"])
        self.assertIn("generated_response", result)
    
    def test_extract_and_store_facts(self):
        """Test automatic fact extraction from messages."""
        # Initialize user memory
        self.handler.user_memory.get_user_memory(12345, 67890)
        
        # Test fact extraction with a pattern we know matches
        self.handler._extract_and_store_facts(
            12345, 67890,
            "I work at Google",
            "That's great!"
        )
        
        user_mem = self.handler.user_memory.get_user_memory(12345, 67890)
        # Should have extracted at least one fact
        self.assertGreater(len(user_mem.get("facts", [])), 0)
        # At least one fact should contain Google
        self.assertTrue(any("Google" in fact for fact in user_mem["facts"]))
    
    def test_get_history(self):
        """Test getting message history."""
        # Process a few messages
        for i in range(5):
            self.handler.process_message({
                "message_id": i,
                "channel_id": 67890,
                "guild_id": 0,
                "user_id": 12345,
                "text": f"Message {i}",
                "is_dm": True,
                "is_mentioned": False,
                "username": "testuser",
                "display_name": "Alice",
                "discriminator": None,
                "bot_user_id": 99999
            })
        
        history = self.handler.get_history(limit=3)
        self.assertEqual(len(history), 3)
    
    def test_clear_history(self):
        """Test clearing message history."""
        self.handler.process_message({
            "message_id": 1,
            "channel_id": 67890,
            "guild_id": 0,
            "user_id": 12345,
            "text": "Test",
            "is_dm": True,
            "is_mentioned": False,
            "username": "testuser",
            "display_name": "Alice",
            "discriminator": None,
            "bot_user_id": 99999
        })
        
        self.handler.clear_history()
        history = self.handler.get_history()
        self.assertEqual(len(history), 0)
    
    def test_get_user_memories(self):
        """Test getting user memory summaries."""
        # Clear any existing memories first
        self.handler.user_memory.clear_all()
        
        # Create some user memories
        self.handler.user_memory.update_user_info(111, 222, username="user1")
        self.handler.user_memory.update_user_info(333, 444, username="user2")
        
        memories = self.handler.get_user_memories()
        self.assertEqual(len(memories), 2)
    
    def test_clear_user_memory(self):
        """Test clearing memory for a specific user."""
        # Clear all first to ensure clean state
        self.handler.user_memory.clear_all()
        
        self.handler.user_memory.add_fact(12345, 67890, "Some fact")
        result = self.handler.clear_user_memory(12345, 67890)
        
        self.assertTrue(result["success"])
        stats = self.handler.user_memory.get_statistics()
        self.assertEqual(stats["total_users"], 0)
    
    def test_clear_all_memories(self):
        """Test clearing all user memories."""
        self.handler.user_memory.add_fact(111, 222, "Fact 1")
        self.handler.user_memory.add_fact(333, 444, "Fact 2")
        
        result = self.handler.clear_all_memories()
        self.assertTrue(result["success"])
        
        stats = self.handler.user_memory.get_statistics()
        self.assertEqual(stats["total_users"], 0)
    
    def test_add_user_fact(self):
        """Test manually adding a fact about a user."""
        result = self.handler.add_user_fact(12345, 67890, "Manually added fact")
        
        self.assertTrue(result["success"])
        user_mem = self.handler.user_memory.get_user_memory(12345, 67890)
        self.assertIn("Manually added fact", user_mem["facts"])


class TestRateLimiter(unittest.TestCase):
    """Test cases for RateLimiter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        from discord_handler import RateLimiter
        self.rate_limiter = RateLimiter({
            "enabled": True,
            "messages_per_second": 2.0,
            "messages_per_minute": 10,
            "messages_per_channel_per_minute": 5,
            "cooldown_seconds": 5.0,
            "max_retries": 3
        })
    
    def test_initial_state(self):
        """Test initial state of rate limiter."""
        stats = self.rate_limiter.get_statistics()
        self.assertEqual(stats["messages_sent"], 0)
        self.assertEqual(stats["messages_delayed"], 0)
        self.assertEqual(stats["messages_blocked"], 0)
        self.assertEqual(stats["rate_limit_hits"], 0)
        self.assertFalse(stats["in_cooldown"])
    
    def test_can_send_initially(self):
        """Test that messages can be sent initially."""
        can_send, wait_time = self.rate_limiter.can_send(12345)
        self.assertTrue(can_send)
        self.assertEqual(wait_time, 0.0)
    
    def test_record_send(self):
        """Test recording a sent message."""
        self.rate_limiter.record_send(12345)
        stats = self.rate_limiter.get_statistics()
        self.assertEqual(stats["messages_sent"], 1)
    
    def test_trigger_cooldown(self):
        """Test triggering cooldown."""
        self.rate_limiter.trigger_cooldown(10.0)
        stats = self.rate_limiter.get_statistics()
        self.assertTrue(stats["in_cooldown"])
        self.assertGreater(stats["cooldown_remaining"], 0)
        self.assertEqual(stats["rate_limit_hits"], 1)
    
    def test_reset_statistics(self):
        """Test resetting statistics."""
        self.rate_limiter.record_send(12345)
        self.rate_limiter.record_delay()
        self.rate_limiter.record_blocked()
        
        self.rate_limiter.reset_statistics()
        
        stats = self.rate_limiter.get_statistics()
        self.assertEqual(stats["messages_sent"], 0)
        self.assertEqual(stats["messages_delayed"], 0)
        self.assertEqual(stats["messages_blocked"], 0)
    
    def test_disabled_rate_limiting(self):
        """Test that rate limiting can be disabled."""
        from discord_handler import RateLimiter
        disabled_limiter = RateLimiter({"enabled": False})
        
        # Should always allow sending when disabled
        can_send, wait_time = disabled_limiter.can_send(12345)
        self.assertTrue(can_send)
        
        # wait_if_needed should always return True when disabled
        result = disabled_limiter.wait_if_needed(12345)
        self.assertTrue(result)
    
    def test_configure(self):
        """Test reconfiguring rate limiter."""
        self.rate_limiter.configure({
            "messages_per_second": 5.0,
            "cooldown_seconds": 10.0
        })
        
        self.assertEqual(self.rate_limiter.messages_per_second, 5.0)
        self.assertEqual(self.rate_limiter.cooldown_seconds, 10.0)


if __name__ == "__main__":
    unittest.main()
