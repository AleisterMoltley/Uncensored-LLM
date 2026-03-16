"""
Discord Integration Handler for LocalLLM
Responds to messages in Discord channels when the bot is mentioned.
Maintains persistent user memory for personalized interactions.
Includes rate limiting to avoid Discord API bans.
"""

import json
import logging
import threading
import time
import re
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, List, Any, Callable, TypedDict, Tuple

from utils import PersistentStorage


# Logger for Discord handler
logger = logging.getLogger("llm_servant.discord")


class RateLimitConfig(TypedDict, total=False):
    """Configuration for rate limiting Discord messages."""
    # Messages per second limit (default: 1)
    messages_per_second: float
    # Messages per minute limit (default: 20)
    messages_per_minute: int
    # Messages per channel per minute (default: 5)
    messages_per_channel_per_minute: int
    # Cooldown after hitting limit in seconds (default: 5)
    cooldown_seconds: float
    # Maximum retry attempts (default: 3)
    max_retries: int
    # Enable rate limiting (default: True)
    enabled: bool


class RateLimiter:
    """
    Rate limiter for Discord bot messages to avoid API bans.
    
    Discord has rate limits:
    - 5 messages per 5 seconds per channel
    - 50 messages per second globally
    
    This limiter is conservative to avoid hitting any limits.
    """
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        """
        Initialize the rate limiter.
        
        Args:
            config: Rate limit configuration
        """
        self.config = config or {}
        self._lock = threading.Lock()
        
        # Global message timestamps (for per-second limiting)
        self._global_timestamps: deque = deque(maxlen=100)
        
        # Per-channel message timestamps (for per-channel limiting)
        self._channel_timestamps: Dict[int, deque] = {}
        
        # Track if currently in cooldown
        self._cooldown_until: float = 0
        
        # Statistics
        self._stats = {
            "messages_sent": 0,
            "messages_delayed": 0,
            "messages_blocked": 0,
            "rate_limit_hits": 0
        }
    
    @property
    def enabled(self) -> bool:
        """Check if rate limiting is enabled."""
        return self.config.get("enabled", True)
    
    @property
    def messages_per_second(self) -> float:
        """Get messages per second limit."""
        return self.config.get("messages_per_second", 1.0)
    
    @property
    def messages_per_minute(self) -> int:
        """Get messages per minute limit."""
        return self.config.get("messages_per_minute", 20)
    
    @property
    def messages_per_channel_per_minute(self) -> int:
        """Get messages per channel per minute limit."""
        return self.config.get("messages_per_channel_per_minute", 5)
    
    @property
    def cooldown_seconds(self) -> float:
        """Get cooldown duration after hitting limit."""
        return self.config.get("cooldown_seconds", 5.0)
    
    @property
    def max_retries(self) -> int:
        """Get maximum retry attempts."""
        return self.config.get("max_retries", 3)
    
    def configure(self, config: RateLimitConfig):
        """Update rate limiter configuration."""
        self.config.update(config)
    
    def _clean_old_timestamps(self, timestamps: deque, max_age_seconds: float = 60.0):
        """Remove timestamps older than max_age_seconds."""
        now = time.time()
        while timestamps and (now - timestamps[0]) > max_age_seconds:
            timestamps.popleft()
    
    def _get_channel_timestamps(self, channel_id: int) -> deque:
        """Get or create timestamp deque for a channel."""
        if channel_id not in self._channel_timestamps:
            self._channel_timestamps[channel_id] = deque(maxlen=100)
        return self._channel_timestamps[channel_id]
    
    def can_send(self, channel_id: int) -> Tuple[bool, float]:
        """
        Check if a message can be sent now.
        
        Args:
            channel_id: The Discord channel ID
            
        Returns:
            Tuple of (can_send: bool, wait_time: float)
        """
        if not self.enabled:
            return True, 0.0
        
        now = time.time()
        
        with self._lock:
            # Check if in cooldown
            if now < self._cooldown_until:
                wait_time = self._cooldown_until - now
                return False, wait_time
            
            # Clean old timestamps
            self._clean_old_timestamps(self._global_timestamps, 60.0)
            
            channel_timestamps = self._get_channel_timestamps(channel_id)
            self._clean_old_timestamps(channel_timestamps, 60.0)
            
            # Check global per-second limit
            recent_global = sum(1 for ts in self._global_timestamps if now - ts < 1.0)
            if recent_global >= self.messages_per_second:
                # Calculate wait time until next message can be sent
                if self._global_timestamps:
                    wait_time = 1.0 - (now - self._global_timestamps[-1])
                else:
                    wait_time = 1.0
                return False, max(0.1, wait_time)
            
            # Check global per-minute limit
            if len(self._global_timestamps) >= self.messages_per_minute:
                oldest = self._global_timestamps[0]
                wait_time = 60.0 - (now - oldest)
                if wait_time > 0:
                    return False, wait_time
            
            # Check per-channel per-minute limit
            if len(channel_timestamps) >= self.messages_per_channel_per_minute:
                oldest = channel_timestamps[0]
                wait_time = 60.0 - (now - oldest)
                if wait_time > 0:
                    return False, wait_time
            
            return True, 0.0
    
    def record_send(self, channel_id: int):
        """Record that a message was sent."""
        now = time.time()
        with self._lock:
            self._global_timestamps.append(now)
            self._get_channel_timestamps(channel_id).append(now)
            self._stats["messages_sent"] += 1
    
    def record_delay(self):
        """Record that a message was delayed."""
        with self._lock:
            self._stats["messages_delayed"] += 1
    
    def record_blocked(self):
        """Record that a message was blocked."""
        with self._lock:
            self._stats["messages_blocked"] += 1
    
    def trigger_cooldown(self, duration: Optional[float] = None):
        """
        Trigger a cooldown period (usually after hitting a rate limit).
        
        Args:
            duration: Cooldown duration in seconds (uses config default if None)
        """
        if duration is None:
            duration = self.cooldown_seconds
        
        with self._lock:
            self._cooldown_until = time.time() + duration
            self._stats["rate_limit_hits"] += 1
            logger.warning("Rate limit triggered, cooldown for %.1f seconds", duration)
    
    def wait_if_needed(self, channel_id: int, max_wait: float = 30.0) -> bool:
        """
        Wait if necessary to respect rate limits.
        
        Args:
            channel_id: The Discord channel ID
            max_wait: Maximum time to wait in seconds
            
        Returns:
            True if message can be sent, False if blocked after max_wait
        """
        if not self.enabled:
            return True
        
        total_waited = 0.0
        
        while total_waited < max_wait:
            can_send, wait_time = self.can_send(channel_id)
            
            if can_send:
                return True
            
            # Limit wait time to remaining max_wait
            actual_wait = min(wait_time, max_wait - total_waited)
            if actual_wait <= 0:
                break
            
            self.record_delay()
            logger.debug("Rate limiting: waiting %.2f seconds before sending to channel %d", 
                        actual_wait, channel_id)
            time.sleep(actual_wait)
            total_waited += actual_wait
        
        self.record_blocked()
        logger.warning("Message blocked after waiting %.1f seconds (channel %d)", total_waited, channel_id)
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        with self._lock:
            return {
                **self._stats,
                "in_cooldown": time.time() < self._cooldown_until,
                "cooldown_remaining": max(0, self._cooldown_until - time.time()),
                "config": {
                    "enabled": self.enabled,
                    "messages_per_second": self.messages_per_second,
                    "messages_per_minute": self.messages_per_minute,
                    "messages_per_channel_per_minute": self.messages_per_channel_per_minute,
                    "cooldown_seconds": self.cooldown_seconds
                }
            }
    
    def reset_statistics(self):
        """Reset rate limiter statistics."""
        with self._lock:
            self._stats = {
                "messages_sent": 0,
                "messages_delayed": 0,
                "messages_blocked": 0,
                "rate_limit_hits": 0
            }


# Discord data persistence
DISCORD_DIR = Path(__file__).parent / "discord_data"
DISCORD_DIR.mkdir(exist_ok=True)
HISTORY_FILE = DISCORD_DIR / "message_history.json"
USER_MEMORY_FILE = DISCORD_DIR / "user_memories.json.gz"


class UserMemory:
    """
    Persistent memory system for Discord users.
    Stores conversation summaries and user facts to enable
    personalized responses even after days or weeks.
    """
    
    def __init__(self, memory_file: Path = USER_MEMORY_FILE, max_size_mb: float = 10.0):
        """
        Initialize user memory system.
        
        Args:
            memory_file: Path to the compressed memory file
            max_size_mb: Maximum memory file size in MB
        """
        self.memory_file = memory_file
        self.max_size_mb = max_size_mb
        self.memories: Dict[str, Dict[str, Any]] = {}
        
        # Initialize persistent storage with compression callback
        self._storage = PersistentStorage(
            memory_file,
            max_size_mb=max_size_mb,
            on_size_exceeded=self._compress_callback
        )
        self._load()
    
    def _compress_callback(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Callback for PersistentStorage when size limit is exceeded.
        Compresses old entries and returns the compressed data.
        """
        self.memories = data
        self._compress_old_entries()
        return self.memories
    
    def _load(self):
        """Load memories from compressed file."""
        self.memories = self._storage.load(default={})
    
    def _save(self):
        """Save memories to compressed file with size check."""
        self._storage.save(self.memories)
    
    def _compress_old_entries(self):
        """Remove oldest conversation entries to reduce file size."""
        for user_id, user_data in self.memories.items():
            conversations = user_data.get("conversations", [])
            # Keep only last 20 conversations per user
            if len(conversations) > 20:
                user_data["conversations"] = conversations[-20:]
    
    def _get_user_key(self, user_id: int, guild_id: int) -> str:
        """Generate a unique key for user in specific guild."""
        return f"{user_id}_{guild_id}"
    
    def get_user_memory(self, user_id: int, guild_id: int) -> Dict[str, Any]:
        """
        Get memory for a specific user in a guild.
        
        Args:
            user_id: Discord user ID
            guild_id: Discord guild (server) ID
            
        Returns:
            User memory dictionary
        """
        key = self._get_user_key(user_id, guild_id)
        if key not in self.memories:
            self.memories[key] = {
                "user_id": user_id,
                "guild_id": guild_id,
                "first_seen": datetime.now(timezone.utc).isoformat(),
                "last_seen": datetime.now(timezone.utc).isoformat(),
                "username": None,
                "display_name": None,
                "discriminator": None,
                "facts": [],  # Known facts about the user
                "preferences": {},  # User preferences
                "conversations": [],  # Recent conversation summaries
                "interaction_count": 0
            }
        return self.memories[key]
    
    def update_user_info(self, user_id: int, guild_id: int, 
                         username: Optional[str] = None,
                         display_name: Optional[str] = None,
                         discriminator: Optional[str] = None):
        """Update basic user information."""
        user_mem = self.get_user_memory(user_id, guild_id)
        if username:
            user_mem["username"] = username
        if display_name:
            user_mem["display_name"] = display_name
        if discriminator:
            user_mem["discriminator"] = discriminator
        user_mem["last_seen"] = datetime.now(timezone.utc).isoformat()
        self._save()
    
    def add_conversation(self, user_id: int, guild_id: int, 
                         user_message: str, bot_response: str):
        """
        Add a conversation exchange to user memory.
        
        Args:
            user_id: Discord user ID
            guild_id: Discord guild ID
            user_message: The user's message
            bot_response: The bot's response
        """
        user_mem = self.get_user_memory(user_id, guild_id)
        user_mem["conversations"].append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_message": user_message[:500],  # Limit message length
            "bot_response": bot_response[:500]
        })
        user_mem["interaction_count"] += 1
        user_mem["last_seen"] = datetime.now(timezone.utc).isoformat()
        
        # Keep only last 50 conversations per user
        if len(user_mem["conversations"]) > 50:
            user_mem["conversations"] = user_mem["conversations"][-50:]
        
        self._save()
    
    def add_fact(self, user_id: int, guild_id: int, fact: str):
        """
        Add a learned fact about the user.
        
        Args:
            user_id: Discord user ID
            guild_id: Discord guild ID
            fact: A fact learned about the user
        """
        user_mem = self.get_user_memory(user_id, guild_id)
        
        # Avoid duplicate facts
        if fact not in user_mem["facts"]:
            user_mem["facts"].append(fact)
            # Keep only last 20 facts
            if len(user_mem["facts"]) > 20:
                user_mem["facts"] = user_mem["facts"][-20:]
            self._save()
    
    def set_preference(self, user_id: int, guild_id: int, key: str, value: Any):
        """Set a user preference."""
        user_mem = self.get_user_memory(user_id, guild_id)
        user_mem["preferences"][key] = value
        self._save()
    
    def format_memory_for_prompt(self, user_id: int, guild_id: int, 
                                  max_conversations: int = 5) -> str:
        """
        Format user memory as context for the LLM prompt.
        
        Args:
            user_id: Discord user ID
            guild_id: Discord guild ID
            max_conversations: Maximum number of recent conversations to include
            
        Returns:
            Formatted string for inclusion in prompt
        """
        user_mem = self.get_user_memory(user_id, guild_id)
        parts = []
        
        # User info
        if user_mem.get("display_name"):
            parts.append(f"Display name: {user_mem['display_name']}")
        if user_mem.get("username"):
            parts.append(f"Username: {user_mem['username']}")
        
        # Interaction stats
        parts.append(f"Interactions: {user_mem.get('interaction_count', 0)}")
        if user_mem.get("first_seen"):
            parts.append(f"First seen: {user_mem['first_seen'][:10]}")
        
        # Known facts
        if user_mem.get("facts"):
            parts.append(f"Known facts about user: {'; '.join(user_mem['facts'][-5:])}")
        
        # Recent conversations
        conversations = user_mem.get("conversations", [])[-max_conversations:]
        if conversations:
            conv_parts = []
            for conv in conversations:
                conv_parts.append(f"- User: {conv['user_message'][:100]}")
                conv_parts.append(f"  Bot: {conv['bot_response'][:100]}")
            parts.append(f"Recent conversation history:\n" + "\n".join(conv_parts))
        
        if not parts:
            return ""
        
        return "USER MEMORY (remember this person):\n" + "\n".join(parts)
    
    def get_all_users(self) -> List[Dict[str, Any]]:
        """Get summary of all users in memory."""
        users = []
        for key, data in self.memories.items():
            users.append({
                "user_id": data.get("user_id"),
                "guild_id": data.get("guild_id"),
                "username": data.get("username"),
                "display_name": data.get("display_name"),
                "discriminator": data.get("discriminator"),
                "interaction_count": data.get("interaction_count", 0),
                "last_seen": data.get("last_seen"),
                "facts_count": len(data.get("facts", []))
            })
        return sorted(users, key=lambda x: x.get("last_seen", ""), reverse=True)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics."""
        total_users = len(self.memories)
        total_conversations = sum(
            len(u.get("conversations", [])) for u in self.memories.values()
        )
        total_facts = sum(
            len(u.get("facts", [])) for u in self.memories.values()
        )
        
        file_size_mb = 0
        if self.memory_file.exists():
            file_size_mb = self.memory_file.stat().st_size / (1024 * 1024)
        
        return {
            "total_users": total_users,
            "total_conversations": total_conversations,
            "total_facts": total_facts,
            "file_size_mb": round(file_size_mb, 3)
        }
    
    def clear_user(self, user_id: int, guild_id: int):
        """Clear memory for a specific user."""
        key = self._get_user_key(user_id, guild_id)
        if key in self.memories:
            del self.memories[key]
            self._save()
    
    def clear_all(self):
        """Clear all user memories."""
        self.memories = {}
        self._save()


class DiscordHandler:
    """
    Handles Discord integration for the LocalLLM system.
    - Listens for messages where the bot is mentioned
    - Responds automatically using the LLM (no approval needed)
    - Maintains persistent user memory
    - Uses active personality from dashboard
    - Includes rate limiting to avoid Discord API bans
    """
    
    def __init__(self, config: Dict, llm_callback: Callable[[str], str],
                 personality_prompt_builder: Optional[Callable[[str], str]] = None):
        """
        Initialize the Discord handler.
        
        Args:
            config: Full configuration from config.json
            llm_callback: Function to generate LLM responses
            personality_prompt_builder: Optional function to build prompts using active personality
        """
        self.full_config = config
        self.config = config.get("discord", {})
        self.llm_callback = llm_callback
        self.personality_prompt_builder = personality_prompt_builder
        self.client = None
        self.bot_info = None
        self.user_memory = UserMemory()
        self._bot_thread: Optional[threading.Thread] = None
        self._bot_running = False
        self._history: List[Dict] = []
        self._load_history()
        
        # Initialize rate limiter
        rate_limit_config = self.config.get("rate_limit", {})
        self.rate_limiter = RateLimiter(rate_limit_config)
    
    def _load_history(self):
        """Load message history from disk."""
        try:
            if HISTORY_FILE.exists():
                with open(HISTORY_FILE) as f:
                    self._history = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️ Could not load Discord history: {e}")
            self._history = []
    
    def _save_history(self):
        """Save message history to disk."""
        try:
            with open(HISTORY_FILE, "w") as f:
                json.dump(self._history, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"⚠️ Could not save Discord history: {e}")
    
    def configure(self, discord_config: Dict):
        """
        Configure Discord bot settings.
        
        Args:
            discord_config: Dictionary containing:
                - bot_token: Discord Bot Token
                - respond_to_mentions: Whether to respond when mentioned in channels
                - respond_to_direct: Whether to respond to direct messages
                - task: Description of how the bot should behave
                - rate_limit: Optional rate limit configuration
        """
        self.config = discord_config
        
        # Update rate limiter configuration if provided
        if "rate_limit" in discord_config:
            self.rate_limiter.configure(discord_config["rate_limit"])
        
        return self.get_status()
    
    def get_status(self) -> Dict:
        """Get current Discord handler status including rate limiter info."""
        return {
            "configured": bool(self.config.get("bot_token")),
            "running": self._bot_running,
            "respond_to_mentions": self.config.get("respond_to_mentions", True),
            "respond_to_direct": self.config.get("respond_to_direct", True),
            "task": self.config.get("task", ""),
            "messages_total": len(self._history),
            "messages_replied": len([h for h in self._history if h.get("replied")]),
            "user_memory": self.user_memory.get_statistics(),
            "rate_limiter": self.rate_limiter.get_statistics()
        }
    
    def _is_bot_mentioned(self, message, bot_user_id: int) -> bool:
        """Check if the bot is mentioned in the message."""
        # Check if bot's user ID is in message mentions
        if hasattr(message, 'mentions'):
            for user in message.mentions:
                if user.id == bot_user_id:
                    return True
        return False
    
    def _extract_message_without_mention(self, text: str, bot_user_id: int) -> str:
        """Remove bot mention from message text."""
        # Remove Discord mention format: <@USER_ID> or <@!USER_ID>
        pattern = rf"<@!?{bot_user_id}>\s*"
        return re.sub(pattern, "", text).strip()
    
    def generate_response(self, message: str, user_id: int, guild_id: int,
                          username: Optional[str] = None,
                          display_name: Optional[str] = None,
                          discriminator: Optional[str] = None) -> str:
        """
        Generate a response to a Discord message using the LLM.
        Responds automatically without requiring user approval.
        
        Args:
            message: The user's message
            user_id: Discord user ID
            guild_id: Discord guild ID
            username: User's Discord username
            display_name: User's display name
            discriminator: User's discriminator (legacy, may be None)
            
        Returns:
            Generated response text
        """
        # Update user info in memory
        self.user_memory.update_user_info(
            user_id, guild_id, username, display_name, discriminator
        )
        
        # Get user memory context
        memory_context = self.user_memory.format_memory_for_prompt(user_id, guild_id)
        
        task = self.config.get("task", "respond helpfully and engagingly")
        
        # Build the user query for Discord
        user_display = display_name or username or "User"
        user_query_parts = [
            f"You are a Discord Bot assistant. Your task is: {task}",
            "",
            "IMPORTANT: You have memory of past interactions with this user. "
            "Use this memory to give personalized, contextual responses. "
            "Reference past conversations when relevant.",
            ""
        ]
        
        if memory_context:
            user_query_parts.append(memory_context)
            user_query_parts.append("")
        
        user_query_parts.append(f"Current message from {user_display}:")
        user_query_parts.append(f'"{message}"')
        user_query_parts.append("")
        user_query_parts.append("Generate a helpful, engaging response. "
                               "If the user mentions something personal, "
                               "remember it for future conversations.")
        
        user_query = "\n".join(user_query_parts)
        
        try:
            # Use personality-based prompt if available
            if self.personality_prompt_builder:
                prompt = self.personality_prompt_builder(user_query)
                response = self.llm_callback(prompt)
            else:
                response = self.llm_callback(user_query)
            return response.strip()
        except Exception as e:
            print(f"⚠️ LLM response error: {e}")
            return ""
    
    def process_message(self, message_data: Dict) -> Dict:
        """
        Process a Discord message and generate a response.
        
        Args:
            message_data: Dictionary containing message information
            
        Returns:
            Processing result dictionary
        """
        message_id = message_data.get("message_id")
        channel_id = message_data.get("channel_id")
        guild_id = message_data.get("guild_id", 0)  # 0 for DMs
        user_id = message_data.get("user_id")
        text = message_data.get("text", "")
        is_dm = message_data.get("is_dm", False)
        is_mentioned = message_data.get("is_mentioned", False)
        username = message_data.get("username")
        display_name = message_data.get("display_name")
        discriminator = message_data.get("discriminator")
        
        # Check if we should respond
        should_respond = False
        clean_message = text
        
        if is_dm:
            # Always respond in DMs if enabled
            should_respond = self.config.get("respond_to_direct", True)
        else:
            # In guild channels, only respond if mentioned
            if self.config.get("respond_to_mentions", True) and is_mentioned:
                should_respond = True
                bot_user_id = message_data.get("bot_user_id", 0)
                clean_message = self._extract_message_without_mention(text, bot_user_id)
        
        result = {
            "message_id": message_id,
            "channel_id": channel_id,
            "guild_id": guild_id,
            "user_id": user_id,
            "username": username,
            "display_name": display_name,
            "text": text,
            "is_dm": is_dm,
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "should_respond": should_respond,
            "replied": False
        }
        
        if should_respond and clean_message:
            # Generate response
            response = self.generate_response(
                clean_message, user_id, guild_id,
                username, display_name, discriminator
            )
            result["generated_response"] = response
            
            if response:
                # Store conversation in user memory
                self.user_memory.add_conversation(
                    user_id, guild_id, clean_message, response
                )
                
                # Try to extract and store facts from the conversation
                self._extract_and_store_facts(
                    user_id, guild_id, clean_message, response
                )
        
        # Add to history
        self._history.append(result)
        # Keep only last 500 messages
        if len(self._history) > 500:
            self._history = self._history[-500:]
        self._save_history()
        
        return result
    
    def _extract_and_store_facts(self, user_id: int, guild_id: int,
                                  user_message: str, bot_response: str):
        """
        Try to extract factual information about the user from the conversation.
        This is a simple heuristic-based extraction.
        """
        # Simple patterns to detect self-disclosed information
        patterns = [
            (r"(?:ich bin|i am|i'm)\s+(\w+)", "User might be: {}"),
            (r"(?:ich arbeite|i work)\s+(?:als|as|at)\s+(.+?)(?:\.|$)", "Works as/at: {}"),
            (r"(?:ich mag|i like|i love)\s+(.+?)(?:\.|$)", "Likes: {}"),
            (r"(?:ich hasse|i hate|i don't like)\s+(.+?)(?:\.|$)", "Dislikes: {}"),
            (r"(?:ich wohne|i live)\s+(?:in|at)\s+(.+?)(?:\.|$)", "Lives in: {}"),
            (r"(?:mein name ist|my name is)\s+(\w+)", "Name: {}"),
        ]
        
        for pattern, fact_template in patterns:
            match = re.search(pattern, user_message, re.IGNORECASE)
            if match:
                fact = fact_template.format(match.group(1).strip())
                self.user_memory.add_fact(user_id, guild_id, fact)
    
    def start_bot(self):
        """Start the Discord bot in a background thread."""
        if self._bot_running:
            return {"success": False, "message": "Bot already running"}
        
        if not self.config.get("bot_token"):
            return {"success": False, "message": "Bot token not configured"}
        
        self._bot_running = True
        self._bot_thread = threading.Thread(target=self._bot_loop, daemon=True)
        self._bot_thread.start()
        
        return {"success": True, "message": "Bot started"}
    
    def stop_bot(self):
        """Stop the Discord bot."""
        self._bot_running = False
        return {"success": True, "message": "Bot stopped"}
    
    def _bot_loop(self):
        """Main bot event loop using discord.py with rate limiting."""
        try:
            import asyncio
            import discord
            from discord import Intents
            
            bot_token = self.config.get("bot_token")
            rate_limiter = self.rate_limiter  # Reference to rate limiter
            handler = self  # Reference to handler
            
            intents = Intents.default()
            intents.message_content = True
            intents.guilds = True
            intents.dm_messages = True
            
            client = discord.Client(intents=intents)
            self.client = client
            
            @client.event
            async def on_ready():
                """Called when bot is ready."""
                handler.bot_info = {
                    "id": client.user.id,
                    "username": client.user.name,
                    "discriminator": client.user.discriminator
                }
                print(f"🎮 Discord bot {client.user.name} connected")
            
            @client.event
            async def on_message(message):
                """Handle incoming messages with rate limiting."""
                # Ignore messages from self
                if message.author == client.user:
                    return
                
                # Check if this is a DM or guild message
                is_dm = message.guild is None
                guild_id = message.guild.id if message.guild else 0
                channel_id = message.channel.id
                
                # Check if bot is mentioned
                is_mentioned = handler._is_bot_mentioned(message, client.user.id)
                
                # Prepare message data
                message_data = {
                    "message_id": message.id,
                    "channel_id": channel_id,
                    "guild_id": guild_id,
                    "user_id": message.author.id,
                    "username": message.author.name,
                    "display_name": message.author.display_name,
                    "discriminator": getattr(message.author, 'discriminator', None),
                    "text": message.content,
                    "is_dm": is_dm,
                    "is_mentioned": is_mentioned,
                    "bot_user_id": client.user.id
                }
                
                # Process message
                result = handler.process_message(message_data)
                
                # Send response if one was generated
                if result.get("should_respond") and result.get("generated_response"):
                    # Apply rate limiting before sending
                    can_send = rate_limiter.wait_if_needed(channel_id)
                    
                    if not can_send:
                        logger.warning("Message blocked by rate limiter for channel %d", channel_id)
                        # Update history to mark as rate limited
                        for entry in handler._history:
                            if entry.get("message_id") == message.id:
                                entry["rate_limited"] = True
                                break
                        handler._save_history()
                        return
                    
                    # Try to send with retry logic for rate limit errors
                    max_retries = rate_limiter.max_retries
                    for attempt in range(max_retries):
                        try:
                            # Split long messages if needed (Discord has 2000 char limit)
                            response_text = result["generated_response"]
                            if len(response_text) > 2000:
                                response_text = response_text[:1997] + "..."
                            
                            await message.reply(response_text)
                            # Record successful send
                            rate_limiter.record_send(channel_id)
                            # Update history to mark as replied
                            for entry in handler._history:
                                if entry.get("message_id") == message.id:
                                    entry["replied"] = True
                                    break
                            handler._save_history()
                            break
                            
                        except discord.errors.HTTPException as e:
                            if e.status == 429:  # Rate limited
                                wait_time = e.retry_after if hasattr(e, 'retry_after') else 5
                                logger.warning("Discord rate limit hit, retry after %d seconds", wait_time)
                                rate_limiter.trigger_cooldown(wait_time + 1)
                                
                                if attempt < max_retries - 1:
                                    await asyncio.sleep(wait_time + 1)
                                else:
                                    logger.error("Max retries reached for message %d", message.id)
                                    for entry in handler._history:
                                        if entry.get("message_id") == message.id:
                                            entry["rate_limited"] = True
                                            break
                                    handler._save_history()
                            else:
                                logger.error("Discord HTTP error: %s", str(e))
                                break
                                
                        except Exception as e:
                            logger.error("Failed to send Discord response: %s", str(e))
                            print(f"⚠️ Failed to send Discord response: {e}")
                            break
            
            # Run the bot
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                print("🎮 Discord bot starting...")
                loop.run_until_complete(client.start(bot_token))
            except KeyboardInterrupt:
                pass
            finally:
                if not client.is_closed():
                    loop.run_until_complete(client.close())
                loop.close()
                
        except ImportError:
            print("⚠️ discord.py not installed. Run: pip install discord.py")
            self._bot_running = False
        except Exception as e:
            print(f"⚠️ Discord bot error: {e}")
            self._bot_running = False
    
    def get_history(self, limit: int = 50) -> List[Dict]:
        """
        Get message processing history.
        
        Args:
            limit: Maximum number of history entries to return
            
        Returns:
            List of history entries (most recent first)
        """
        return sorted(
            self._history,
            key=lambda x: x.get("processed_at", ""),
            reverse=True
        )[:limit]
    
    def clear_history(self):
        """Clear message processing history."""
        self._history = []
        self._save_history()
        return {"success": True, "message": "History cleared"}
    
    def get_user_memories(self, limit: int = 50) -> List[Dict]:
        """Get list of users with their memory summaries."""
        return self.user_memory.get_all_users()[:limit]
    
    def get_user_memory_detail(self, user_id: int, guild_id: int) -> Dict:
        """Get detailed memory for a specific user."""
        return self.user_memory.get_user_memory(user_id, guild_id)
    
    def clear_user_memory(self, user_id: int, guild_id: int):
        """Clear memory for a specific user."""
        self.user_memory.clear_user(user_id, guild_id)
        return {"success": True, "message": f"Memory cleared for user {user_id}"}
    
    def clear_all_memories(self):
        """Clear all user memories."""
        self.user_memory.clear_all()
        return {"success": True, "message": "All user memories cleared"}
    
    def add_user_fact(self, user_id: int, guild_id: int, fact: str):
        """Manually add a fact about a user."""
        self.user_memory.add_fact(user_id, guild_id, fact)
        return {"success": True, "message": "Fact added"}
    
    # Rate limiter management methods
    
    def get_rate_limit_config(self) -> RateLimitConfig:
        """Get the current rate limit configuration."""
        return self.rate_limiter.config
    
    def set_rate_limit_config(self, config: RateLimitConfig) -> Dict:
        """
        Update the rate limit configuration.
        
        Args:
            config: New rate limit settings
            
        Returns:
            Updated status with new configuration
        """
        self.rate_limiter.configure(config)
        self.config["rate_limit"] = config
        return {
            "success": True,
            "rate_limit": self.rate_limiter.get_statistics()
        }
    
    def enable_rate_limiting(self, enabled: bool = True) -> Dict:
        """
        Enable or disable rate limiting.
        
        Args:
            enabled: Whether to enable rate limiting
            
        Returns:
            Updated status
        """
        self.rate_limiter.configure({"enabled": enabled})
        return {
            "success": True,
            "enabled": enabled,
            "rate_limit": self.rate_limiter.get_statistics()
        }
    
    def get_rate_limit_statistics(self) -> Dict:
        """Get rate limiter statistics."""
        return self.rate_limiter.get_statistics()
    
    def reset_rate_limit_statistics(self) -> Dict:
        """Reset rate limiter statistics."""
        self.rate_limiter.reset_statistics()
        return {
            "success": True,
            "message": "Rate limiter statistics reset"
        }
