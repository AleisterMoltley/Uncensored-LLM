"""
Tests for Telegram Handler and User Memory System
"""

import tempfile
import shutil
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

from telegram_handler import TelegramHandler, UserMemory


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
        self.assertEqual(user_mem["chat_id"], 67890)
        self.assertEqual(user_mem["interaction_count"], 0)
        self.assertEqual(len(user_mem["facts"]), 0)
        self.assertEqual(len(user_mem["conversations"]), 0)
    
    def test_update_user_info(self):
        """Test updating user information."""
        self.user_memory.update_user_info(
            12345, 67890,
            username="testuser",
            first_name="Test",
            last_name="User"
        )
        
        user_mem = self.user_memory.get_user_memory(12345, 67890)
        self.assertEqual(user_mem["username"], "testuser")
        self.assertEqual(user_mem["first_name"], "Test")
        self.assertEqual(user_mem["last_name"], "User")
    
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
            first_name="Alice"
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


class TestTelegramHandler(unittest.TestCase):
    """Test cases for TelegramHandler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "telegram": {
                "bot_token": "test_token_123",
                "bot_username": "test_bot",
                "respond_to_mentions": True,
                "respond_to_direct": True,
                "task": "Be helpful and friendly"
            }
        }
        self.llm_callback = MagicMock(return_value="This is a test response from the LLM.")
        
        # Use temp directory for data
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Patch the data directories
        with patch('telegram_handler.TELEGRAM_DIR', self.temp_dir):
            with patch('telegram_handler.HISTORY_FILE', self.temp_dir / "history.json"):
                with patch('telegram_handler.USER_MEMORY_FILE', self.temp_dir / "memories.json.gz"):
                    self.handler = TelegramHandler(self.config, self.llm_callback)
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test that handler initializes correctly."""
        status = self.handler.get_status()
        self.assertTrue(status["configured"])
        self.assertFalse(status["running"])
        self.assertEqual(status["bot_username"], "test_bot")
    
    def test_configure(self):
        """Test reconfiguring the handler."""
        new_config = {
            "bot_token": "new_token",
            "bot_username": "new_bot",
            "task": "New task"
        }
        
        status = self.handler.configure(new_config)
        self.assertEqual(self.handler.config["bot_username"], "new_bot")
    
    def test_is_bot_mentioned(self):
        """Test bot mention detection."""
        self.assertTrue(self.handler._is_bot_mentioned("Hey @test_bot what's up?"))
        self.assertTrue(self.handler._is_bot_mentioned("@test_bot"))
        self.assertTrue(self.handler._is_bot_mentioned("Hello @TEST_BOT world"))  # case insensitive
        
        self.assertFalse(self.handler._is_bot_mentioned("No mention here"))
        self.assertFalse(self.handler._is_bot_mentioned("@other_bot"))
    
    def test_extract_message_without_mention(self):
        """Test extracting message content without bot mention."""
        result = self.handler._extract_message_without_mention("@test_bot What is AI?")
        self.assertEqual(result, "What is AI?")
        
        result = self.handler._extract_message_without_mention("Hello @test_bot how are you?")
        self.assertEqual(result, "Hello how are you?")
    
    def test_generate_response(self):
        """Test generating a response."""
        response = self.handler.generate_response(
            "What is AI?",
            12345, 67890,
            username="testuser",
            first_name="Alice"
        )
        
        self.assertEqual(response, "This is a test response from the LLM.")
        self.llm_callback.assert_called_once()
        
        # Check that user info was stored
        user_mem = self.handler.user_memory.get_user_memory(12345, 67890)
        self.assertEqual(user_mem["username"], "testuser")
        self.assertEqual(user_mem["first_name"], "Alice")
    
    def test_process_message_private_chat(self):
        """Test processing a private chat message."""
        message_data = {
            "message_id": 1,
            "chat_id": 67890,
            "user_id": 12345,
            "text": "Hello!",
            "chat_type": "private",
            "username": "testuser",
            "first_name": "Alice",
            "last_name": None
        }
        
        result = self.handler.process_message(message_data)
        
        self.assertTrue(result["should_respond"])
        self.assertIn("generated_response", result)
    
    def test_process_message_group_without_mention(self):
        """Test processing a group message without bot mention."""
        message_data = {
            "message_id": 1,
            "chat_id": 67890,
            "user_id": 12345,
            "text": "Hello everyone!",
            "chat_type": "group",
            "username": "testuser",
            "first_name": "Alice",
            "last_name": None
        }
        
        result = self.handler.process_message(message_data)
        
        self.assertFalse(result["should_respond"])
    
    def test_process_message_group_with_mention(self):
        """Test processing a group message with bot mention."""
        message_data = {
            "message_id": 1,
            "chat_id": 67890,
            "user_id": 12345,
            "text": "@test_bot What is AI?",
            "chat_type": "group",
            "username": "testuser",
            "first_name": "Alice",
            "last_name": None
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
                "chat_id": 67890,
                "user_id": 12345,
                "text": f"Message {i}",
                "chat_type": "private",
                "username": "testuser",
                "first_name": "Alice",
                "last_name": None
            })
        
        history = self.handler.get_history(limit=3)
        self.assertEqual(len(history), 3)
    
    def test_clear_history(self):
        """Test clearing message history."""
        self.handler.process_message({
            "message_id": 1,
            "chat_id": 67890,
            "user_id": 12345,
            "text": "Test",
            "chat_type": "private",
            "username": "testuser",
            "first_name": "Alice",
            "last_name": None
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
    
    def test_add_user_fact(self):
        """Test manually adding a fact about a user."""
        result = self.handler.add_user_fact(12345, 67890, "Works in tech")
        
        self.assertTrue(result["success"])
        
        user_mem = self.handler.user_memory.get_user_memory(12345, 67890)
        self.assertIn("Works in tech", user_mem["facts"])
    
    def test_history_limit(self):
        """Test that history is limited to prevent unbounded growth."""
        # Process more than 500 messages
        with patch.object(self.handler, 'generate_response', return_value="Response"):
            for i in range(510):
                self.handler.process_message({
                    "message_id": i,
                    "chat_id": 67890,
                    "user_id": 12345,
                    "text": f"Message {i}",
                    "chat_type": "private",
                    "username": "testuser",
                    "first_name": "Alice",
                    "last_name": None
                })
        
        # History should be limited to 500
        self.assertEqual(len(self.handler._history), 500)


class TestTelegramHandlerDisabledResponses(unittest.TestCase):
    """Test cases for disabled response settings."""
    
    def test_respond_to_direct_disabled(self):
        """Test that direct messages are ignored when disabled."""
        config = {
            "telegram": {
                "bot_token": "test_token",
                "bot_username": "test_bot",
                "respond_to_direct": False,
                "respond_to_mentions": True
            }
        }
        llm_callback = MagicMock(return_value="Response")
        
        temp_dir = Path(tempfile.mkdtemp())
        try:
            with patch('telegram_handler.TELEGRAM_DIR', temp_dir):
                with patch('telegram_handler.HISTORY_FILE', temp_dir / "history.json"):
                    with patch('telegram_handler.USER_MEMORY_FILE', temp_dir / "memories.json.gz"):
                        handler = TelegramHandler(config, llm_callback)
            
            result = handler.process_message({
                "message_id": 1,
                "chat_id": 67890,
                "user_id": 12345,
                "text": "Hello!",
                "chat_type": "private",
                "username": "testuser",
                "first_name": "Alice",
                "last_name": None
            })
            
            self.assertFalse(result["should_respond"])
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_respond_to_mentions_disabled(self):
        """Test that mentions are ignored when disabled."""
        config = {
            "telegram": {
                "bot_token": "test_token",
                "bot_username": "test_bot",
                "respond_to_direct": True,
                "respond_to_mentions": False
            }
        }
        llm_callback = MagicMock(return_value="Response")
        
        temp_dir = Path(tempfile.mkdtemp())
        try:
            with patch('telegram_handler.TELEGRAM_DIR', temp_dir):
                with patch('telegram_handler.HISTORY_FILE', temp_dir / "history.json"):
                    with patch('telegram_handler.USER_MEMORY_FILE', temp_dir / "memories.json.gz"):
                        handler = TelegramHandler(config, llm_callback)
            
            result = handler.process_message({
                "message_id": 1,
                "chat_id": 67890,
                "user_id": 12345,
                "text": "@test_bot Hello!",
                "chat_type": "group",
                "username": "testuser",
                "first_name": "Alice",
                "last_name": None
            })
            
            self.assertFalse(result["should_respond"])
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestRateLimiter(unittest.TestCase):
    """Test cases for RateLimiter class."""
    
    def test_initialization_default(self):
        """Test that RateLimiter initializes with default values."""
        from telegram_handler import RateLimiter
        
        limiter = RateLimiter()
        
        self.assertTrue(limiter.enabled)
        self.assertEqual(limiter.messages_per_second, 1.0)
        self.assertEqual(limiter.messages_per_minute, 20)
        self.assertEqual(limiter.messages_per_chat_per_minute, 3)
        self.assertEqual(limiter.cooldown_seconds, 5.0)
        self.assertEqual(limiter.max_retries, 3)
    
    def test_initialization_custom(self):
        """Test that RateLimiter initializes with custom values."""
        from telegram_handler import RateLimiter
        
        config = {
            "enabled": False,
            "messages_per_second": 2.0,
            "messages_per_minute": 30,
            "messages_per_chat_per_minute": 5,
            "cooldown_seconds": 10.0,
            "max_retries": 5
        }
        
        limiter = RateLimiter(config)
        
        self.assertFalse(limiter.enabled)
        self.assertEqual(limiter.messages_per_second, 2.0)
        self.assertEqual(limiter.messages_per_minute, 30)
        self.assertEqual(limiter.messages_per_chat_per_minute, 5)
        self.assertEqual(limiter.cooldown_seconds, 10.0)
        self.assertEqual(limiter.max_retries, 5)
    
    def test_can_send_when_disabled(self):
        """Test that can_send always returns True when disabled."""
        from telegram_handler import RateLimiter
        
        limiter = RateLimiter({"enabled": False})
        
        can_send, wait_time = limiter.can_send(12345)
        self.assertTrue(can_send)
        self.assertEqual(wait_time, 0.0)
    
    def test_can_send_initial(self):
        """Test that can_send returns True initially."""
        from telegram_handler import RateLimiter
        
        limiter = RateLimiter({"enabled": True})
        
        can_send, wait_time = limiter.can_send(12345)
        self.assertTrue(can_send)
        self.assertEqual(wait_time, 0.0)
    
    def test_record_send_updates_stats(self):
        """Test that record_send updates statistics."""
        from telegram_handler import RateLimiter
        
        limiter = RateLimiter()
        
        initial_stats = limiter.get_statistics()
        self.assertEqual(initial_stats["messages_sent"], 0)
        
        limiter.record_send(12345)
        
        updated_stats = limiter.get_statistics()
        self.assertEqual(updated_stats["messages_sent"], 1)
    
    def test_record_delay_updates_stats(self):
        """Test that record_delay updates statistics."""
        from telegram_handler import RateLimiter
        
        limiter = RateLimiter()
        limiter.record_delay()
        
        stats = limiter.get_statistics()
        self.assertEqual(stats["messages_delayed"], 1)
    
    def test_record_blocked_updates_stats(self):
        """Test that record_blocked updates statistics."""
        from telegram_handler import RateLimiter
        
        limiter = RateLimiter()
        limiter.record_blocked()
        
        stats = limiter.get_statistics()
        self.assertEqual(stats["messages_blocked"], 1)
    
    def test_trigger_cooldown(self):
        """Test that trigger_cooldown sets cooldown state."""
        from telegram_handler import RateLimiter
        
        limiter = RateLimiter()
        limiter.trigger_cooldown(2.0)
        
        stats = limiter.get_statistics()
        self.assertTrue(stats["in_cooldown"])
        self.assertGreater(stats["cooldown_remaining"], 0)
        self.assertEqual(stats["rate_limit_hits"], 1)
    
    def test_configure(self):
        """Test that configure updates settings."""
        from telegram_handler import RateLimiter
        
        limiter = RateLimiter()
        self.assertTrue(limiter.enabled)
        
        limiter.configure({"enabled": False, "messages_per_minute": 50})
        
        self.assertFalse(limiter.enabled)
        self.assertEqual(limiter.messages_per_minute, 50)
    
    def test_reset_statistics(self):
        """Test that reset_statistics clears stats."""
        from telegram_handler import RateLimiter
        
        limiter = RateLimiter()
        limiter.record_send(12345)
        limiter.record_delay()
        
        limiter.reset_statistics()
        
        stats = limiter.get_statistics()
        self.assertEqual(stats["messages_sent"], 0)
        self.assertEqual(stats["messages_delayed"], 0)
    
    def test_get_statistics_includes_config(self):
        """Test that get_statistics includes config info."""
        from telegram_handler import RateLimiter
        
        limiter = RateLimiter({"messages_per_minute": 25})
        
        stats = limiter.get_statistics()
        
        self.assertIn("config", stats)
        self.assertEqual(stats["config"]["messages_per_minute"], 25)


class TestTelegramHandlerRateLimiting(unittest.TestCase):
    """Test cases for TelegramHandler rate limiting integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "telegram": {
                "bot_token": "test_token_123",
                "bot_username": "test_bot",
                "respond_to_mentions": True,
                "respond_to_direct": True,
                "task": "Be helpful and friendly",
                "rate_limit": {
                    "enabled": True,
                    "messages_per_minute": 10
                }
            }
        }
        self.llm_callback = MagicMock(return_value="Test response")
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_handler_has_rate_limiter(self):
        """Test that TelegramHandler initializes with rate limiter."""
        with patch('telegram_handler.TELEGRAM_DIR', self.temp_dir):
            with patch('telegram_handler.HISTORY_FILE', self.temp_dir / "history.json"):
                with patch('telegram_handler.USER_MEMORY_FILE', self.temp_dir / "memories.json.gz"):
                    handler = TelegramHandler(self.config, self.llm_callback)
        
        self.assertIsNotNone(handler.rate_limiter)
        self.assertTrue(handler.rate_limiter.enabled)
    
    def test_status_includes_rate_limiter(self):
        """Test that get_status includes rate limiter info."""
        with patch('telegram_handler.TELEGRAM_DIR', self.temp_dir):
            with patch('telegram_handler.HISTORY_FILE', self.temp_dir / "history.json"):
                with patch('telegram_handler.USER_MEMORY_FILE', self.temp_dir / "memories.json.gz"):
                    handler = TelegramHandler(self.config, self.llm_callback)
        
        status = handler.get_status()
        self.assertIn("rate_limiter", status)
    
    def test_configure_updates_rate_limiter(self):
        """Test that configure updates rate limiter config."""
        with patch('telegram_handler.TELEGRAM_DIR', self.temp_dir):
            with patch('telegram_handler.HISTORY_FILE', self.temp_dir / "history.json"):
                with patch('telegram_handler.USER_MEMORY_FILE', self.temp_dir / "memories.json.gz"):
                    handler = TelegramHandler(self.config, self.llm_callback)
        
        new_config = {
            "bot_token": "new_token",
            "rate_limit": {
                "messages_per_minute": 50
            }
        }
        
        handler.configure(new_config)
        
        self.assertEqual(handler.rate_limiter.messages_per_minute, 50)
    
    def test_enable_rate_limiting(self):
        """Test enable_rate_limiting method."""
        with patch('telegram_handler.TELEGRAM_DIR', self.temp_dir):
            with patch('telegram_handler.HISTORY_FILE', self.temp_dir / "history.json"):
                with patch('telegram_handler.USER_MEMORY_FILE', self.temp_dir / "memories.json.gz"):
                    handler = TelegramHandler(self.config, self.llm_callback)
        
        result = handler.enable_rate_limiting(False)
        
        self.assertTrue(result["success"])
        self.assertFalse(result["enabled"])
        self.assertFalse(handler.rate_limiter.enabled)
    
    def test_get_rate_limit_statistics(self):
        """Test get_rate_limit_statistics method."""
        with patch('telegram_handler.TELEGRAM_DIR', self.temp_dir):
            with patch('telegram_handler.HISTORY_FILE', self.temp_dir / "history.json"):
                with patch('telegram_handler.USER_MEMORY_FILE', self.temp_dir / "memories.json.gz"):
                    handler = TelegramHandler(self.config, self.llm_callback)
        
        stats = handler.get_rate_limit_statistics()
        
        self.assertIn("messages_sent", stats)
        self.assertIn("messages_delayed", stats)
        self.assertIn("config", stats)


if __name__ == "__main__":
    unittest.main()
