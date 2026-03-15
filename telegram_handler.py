"""
Telegram Integration Handler for LocalLLM
Responds to messages in Telegram chats when the bot is mentioned.
Maintains persistent user memory for personalized interactions.
"""

import json
import gzip
import threading
import time
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, List, Any, Callable


# Telegram data persistence
TELEGRAM_DIR = Path(__file__).parent / "telegram_data"
TELEGRAM_DIR.mkdir(exist_ok=True)
HISTORY_FILE = TELEGRAM_DIR / "message_history.json"
USER_MEMORY_FILE = TELEGRAM_DIR / "user_memories.json.gz"


class UserMemory:
    """
    Persistent memory system for Telegram users.
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
        self._load()
    
    def _load(self):
        """Load memories from compressed file."""
        if self.memory_file.exists():
            try:
                with gzip.open(self.memory_file, 'rt', encoding='utf-8') as f:
                    self.memories = json.load(f)
            except (json.JSONDecodeError, IOError, gzip.BadGzipFile) as e:
                print(f"⚠️ Could not load user memories: {e}")
                self.memories = {}
    
    def _save(self):
        """Save memories to compressed file with size check."""
        json_data = json.dumps(self.memories, ensure_ascii=False, indent=None)
        size_bytes = len(json_data.encode('utf-8'))
        size_mb = size_bytes / (1024 * 1024)
        
        # If too large, compress old entries
        if size_mb > self.max_size_mb:
            self._compress_old_entries()
            json_data = json.dumps(self.memories, ensure_ascii=False, indent=None)
        
        try:
            with gzip.open(self.memory_file, 'wt', encoding='utf-8') as f:
                f.write(json_data)
        except IOError as e:
            print(f"⚠️ Could not save user memories: {e}")
    
    def _compress_old_entries(self):
        """Remove oldest conversation entries to reduce file size."""
        for user_id, user_data in self.memories.items():
            conversations = user_data.get("conversations", [])
            # Keep only last 20 conversations per user
            if len(conversations) > 20:
                user_data["conversations"] = conversations[-20:]
    
    def _get_user_key(self, user_id: int, chat_id: int) -> str:
        """Generate a unique key for user in specific chat."""
        return f"{user_id}_{chat_id}"
    
    def get_user_memory(self, user_id: int, chat_id: int) -> Dict[str, Any]:
        """
        Get memory for a specific user in a chat.
        
        Args:
            user_id: Telegram user ID
            chat_id: Telegram chat ID
            
        Returns:
            User memory dictionary
        """
        key = self._get_user_key(user_id, chat_id)
        if key not in self.memories:
            self.memories[key] = {
                "user_id": user_id,
                "chat_id": chat_id,
                "first_seen": datetime.now(timezone.utc).isoformat(),
                "last_seen": datetime.now(timezone.utc).isoformat(),
                "username": None,
                "first_name": None,
                "last_name": None,
                "facts": [],  # Known facts about the user
                "preferences": {},  # User preferences
                "conversations": [],  # Recent conversation summaries
                "interaction_count": 0
            }
        return self.memories[key]
    
    def update_user_info(self, user_id: int, chat_id: int, 
                         username: Optional[str] = None,
                         first_name: Optional[str] = None,
                         last_name: Optional[str] = None):
        """Update basic user information."""
        user_mem = self.get_user_memory(user_id, chat_id)
        if username:
            user_mem["username"] = username
        if first_name:
            user_mem["first_name"] = first_name
        if last_name:
            user_mem["last_name"] = last_name
        user_mem["last_seen"] = datetime.now(timezone.utc).isoformat()
        self._save()
    
    def add_conversation(self, user_id: int, chat_id: int, 
                         user_message: str, bot_response: str):
        """
        Add a conversation exchange to user memory.
        
        Args:
            user_id: Telegram user ID
            chat_id: Telegram chat ID
            user_message: The user's message
            bot_response: The bot's response
        """
        user_mem = self.get_user_memory(user_id, chat_id)
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
    
    def add_fact(self, user_id: int, chat_id: int, fact: str):
        """
        Add a learned fact about the user.
        
        Args:
            user_id: Telegram user ID
            chat_id: Telegram chat ID
            fact: A fact learned about the user
        """
        user_mem = self.get_user_memory(user_id, chat_id)
        
        # Avoid duplicate facts
        if fact not in user_mem["facts"]:
            user_mem["facts"].append(fact)
            # Keep only last 20 facts
            if len(user_mem["facts"]) > 20:
                user_mem["facts"] = user_mem["facts"][-20:]
            self._save()
    
    def set_preference(self, user_id: int, chat_id: int, key: str, value: Any):
        """Set a user preference."""
        user_mem = self.get_user_memory(user_id, chat_id)
        user_mem["preferences"][key] = value
        self._save()
    
    def format_memory_for_prompt(self, user_id: int, chat_id: int, 
                                  max_conversations: int = 5) -> str:
        """
        Format user memory as context for the LLM prompt.
        
        Args:
            user_id: Telegram user ID
            chat_id: Telegram chat ID
            max_conversations: Maximum number of recent conversations to include
            
        Returns:
            Formatted string for inclusion in prompt
        """
        user_mem = self.get_user_memory(user_id, chat_id)
        parts = []
        
        # User info
        name_parts = []
        if user_mem.get("first_name"):
            name_parts.append(user_mem["first_name"])
        if user_mem.get("last_name"):
            name_parts.append(user_mem["last_name"])
        if name_parts:
            parts.append(f"User name: {' '.join(name_parts)}")
        if user_mem.get("username"):
            parts.append(f"Username: @{user_mem['username']}")
        
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
                "chat_id": data.get("chat_id"),
                "username": data.get("username"),
                "first_name": data.get("first_name"),
                "last_name": data.get("last_name"),
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
    
    def clear_user(self, user_id: int, chat_id: int):
        """Clear memory for a specific user."""
        key = self._get_user_key(user_id, chat_id)
        if key in self.memories:
            del self.memories[key]
            self._save()
    
    def clear_all(self):
        """Clear all user memories."""
        self.memories = {}
        self._save()


class TelegramHandler:
    """
    Handles Telegram integration for the LocalLLM system.
    - Listens for messages where the bot is mentioned
    - Responds automatically using the LLM (no approval needed)
    - Maintains persistent user memory
    - Uses active personality from dashboard
    """
    
    def __init__(self, config: Dict, llm_callback: Callable[[str], str],
                 personality_prompt_builder: Optional[Callable[[str], str]] = None):
        """
        Initialize the Telegram handler.
        
        Args:
            config: Full configuration from config.json
            llm_callback: Function to generate LLM responses
            personality_prompt_builder: Optional function to build prompts using active personality
        """
        self.full_config = config
        self.config = config.get("telegram", {})
        self.llm_callback = llm_callback
        self.personality_prompt_builder = personality_prompt_builder
        self.bot = None
        self.bot_info = None
        self.user_memory = UserMemory()
        self._bot_thread: Optional[threading.Thread] = None
        self._bot_running = False
        self._history: List[Dict] = []
        self._load_history()
    
    def _load_history(self):
        """Load message history from disk."""
        try:
            if HISTORY_FILE.exists():
                with open(HISTORY_FILE) as f:
                    self._history = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️ Could not load Telegram history: {e}")
            self._history = []
    
    def _save_history(self):
        """Save message history to disk."""
        try:
            with open(HISTORY_FILE, "w") as f:
                json.dump(self._history, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"⚠️ Could not save Telegram history: {e}")
    
    def configure(self, telegram_config: Dict):
        """
        Configure Telegram bot settings.
        
        Args:
            telegram_config: Dictionary containing:
                - bot_token: Telegram Bot Token from BotFather
                - bot_username: Bot's username (without @)
                - respond_to_mentions: Whether to respond when mentioned in groups
                - respond_to_direct: Whether to respond to direct messages
                - task: Description of how the bot should behave
        """
        self.config = telegram_config
        return self.get_status()
    
    def _init_bot(self) -> bool:
        """Initialize Telegram bot client."""
        try:
            from telegram import Bot
            from telegram.ext import Application, MessageHandler, filters
            
            bot_token = self.config.get("bot_token", "")
            
            if not bot_token:
                return False
            
            # Create bot instance for basic verification
            self.bot = Bot(token=bot_token)
            return True
            
        except ImportError:
            print("⚠️ python-telegram-bot not installed. Run: pip install python-telegram-bot")
            return False
        except Exception as e:
            print(f"⚠️ Telegram bot init error: {e}")
            return False
    
    def get_status(self) -> Dict:
        """Get current Telegram handler status."""
        return {
            "configured": bool(self.config.get("bot_token")),
            "running": self._bot_running,
            "bot_username": self.config.get("bot_username", ""),
            "respond_to_mentions": self.config.get("respond_to_mentions", True),
            "respond_to_direct": self.config.get("respond_to_direct", True),
            "task": self.config.get("task", ""),
            "messages_total": len(self._history),
            "messages_replied": len([h for h in self._history if h.get("replied")]),
            "user_memory": self.user_memory.get_statistics()
        }
    
    def _is_bot_mentioned(self, text: str) -> bool:
        """Check if the bot is mentioned in the message."""
        bot_username = self.config.get("bot_username", "")
        if not bot_username:
            return False
        
        # Check for @username mention
        mention_pattern = rf"@{re.escape(bot_username)}\b"
        return bool(re.search(mention_pattern, text, re.IGNORECASE))
    
    def _extract_message_without_mention(self, text: str) -> str:
        """Remove bot mention from message text."""
        bot_username = self.config.get("bot_username", "")
        if not bot_username:
            return text
        
        # Remove @username mention
        pattern = rf"@{re.escape(bot_username)}\s*"
        return re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
    
    def generate_response(self, message: str, user_id: int, chat_id: int,
                          username: Optional[str] = None,
                          first_name: Optional[str] = None,
                          last_name: Optional[str] = None) -> str:
        """
        Generate a response to a Telegram message using the LLM.
        Responds automatically without requiring user approval.
        
        Args:
            message: The user's message
            user_id: Telegram user ID
            chat_id: Telegram chat ID
            username: User's Telegram username
            first_name: User's first name
            last_name: User's last name
            
        Returns:
            Generated response text
        """
        # Update user info in memory
        self.user_memory.update_user_info(
            user_id, chat_id, username, first_name, last_name
        )
        
        # Get user memory context
        memory_context = self.user_memory.format_memory_for_prompt(user_id, chat_id)
        
        task = self.config.get("task", "respond helpfully and engagingly")
        
        # Build the user query for Telegram
        display_name = first_name or username or "User"
        user_query_parts = [
            f"Du bist ein Telegram Bot Assistent. Deine Aufgabe ist: {task}",
            "",
            "WICHTIG: Du hast ein Gedächtnis über vergangene Interaktionen mit diesem Nutzer. "
            "Nutze dieses Gedächtnis, um personalisierte, kontextuelle Antworten zu geben. "
            "Beziehe dich auf vergangene Gespräche, wenn relevant.",
            ""
        ]
        
        if memory_context:
            user_query_parts.append(memory_context)
            user_query_parts.append("")
        
        user_query_parts.append(f"Aktuelle Nachricht von {display_name}:")
        user_query_parts.append(f'"{message}"')
        user_query_parts.append("")
        user_query_parts.append("Generiere eine hilfreiche, engagierte Antwort. "
                               "Wenn der Nutzer etwas Persönliches erwähnt, "
                               "merke es dir für zukünftige Gespräche.")
        
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
        Process a Telegram message and generate a response.
        
        Args:
            message_data: Dictionary containing message information
            
        Returns:
            Processing result dictionary
        """
        message_id = message_data.get("message_id")
        chat_id = message_data.get("chat_id")
        user_id = message_data.get("user_id")
        text = message_data.get("text", "")
        chat_type = message_data.get("chat_type", "private")
        username = message_data.get("username")
        first_name = message_data.get("first_name")
        last_name = message_data.get("last_name")
        
        # Check if we should respond
        should_respond = False
        clean_message = text
        
        if chat_type == "private":
            # Always respond in private chats if enabled
            should_respond = self.config.get("respond_to_direct", True)
        else:
            # In group chats, only respond if mentioned
            if self.config.get("respond_to_mentions", True) and self._is_bot_mentioned(text):
                should_respond = True
                clean_message = self._extract_message_without_mention(text)
        
        result = {
            "message_id": message_id,
            "chat_id": chat_id,
            "user_id": user_id,
            "username": username,
            "first_name": first_name,
            "text": text,
            "chat_type": chat_type,
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "should_respond": should_respond,
            "replied": False
        }
        
        if should_respond and clean_message:
            # Generate response
            response = self.generate_response(
                clean_message, user_id, chat_id,
                username, first_name, last_name
            )
            result["generated_response"] = response
            
            if response:
                # Store conversation in user memory
                self.user_memory.add_conversation(
                    user_id, chat_id, clean_message, response
                )
                
                # Try to extract and store facts from the conversation
                self._extract_and_store_facts(
                    user_id, chat_id, clean_message, response
                )
        
        # Add to history
        self._history.append(result)
        # Keep only last 500 messages
        if len(self._history) > 500:
            self._history = self._history[-500:]
        self._save_history()
        
        return result
    
    def _extract_and_store_facts(self, user_id: int, chat_id: int,
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
                self.user_memory.add_fact(user_id, chat_id, fact)
    
    def start_bot(self):
        """Start the Telegram bot in a background thread."""
        if self._bot_running:
            return {"success": False, "message": "Bot already running"}
        
        if not self.config.get("bot_token"):
            return {"success": False, "message": "Bot token not configured"}
        
        self._bot_running = True
        self._bot_thread = threading.Thread(target=self._bot_loop, daemon=True)
        self._bot_thread.start()
        
        return {"success": True, "message": "Bot started"}
    
    def stop_bot(self):
        """Stop the Telegram bot."""
        self._bot_running = False
        return {"success": True, "message": "Bot stopped"}
    
    def _bot_loop(self):
        """Main bot polling loop using python-telegram-bot."""
        try:
            import asyncio
            from telegram import Update, Bot
            from telegram.ext import Application, MessageHandler, filters, ContextTypes
            
            bot_token = self.config.get("bot_token")
            
            async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
                """Handle incoming messages."""
                if not update.message or not update.message.text:
                    return
                
                message = update.message
                
                # Prepare message data
                message_data = {
                    "message_id": message.message_id,
                    "chat_id": message.chat.id,
                    "user_id": message.from_user.id if message.from_user else 0,
                    "username": message.from_user.username if message.from_user else None,
                    "first_name": message.from_user.first_name if message.from_user else None,
                    "last_name": message.from_user.last_name if message.from_user else None,
                    "text": message.text,
                    "chat_type": message.chat.type
                }
                
                # Process message
                result = self.process_message(message_data)
                
                # Send response if one was generated
                if result.get("should_respond") and result.get("generated_response"):
                    try:
                        await message.reply_text(result["generated_response"])
                        # Update history to mark as replied
                        for entry in self._history:
                            if entry.get("message_id") == message.message_id:
                                entry["replied"] = True
                                break
                        self._save_history()
                    except Exception as e:
                        print(f"⚠️ Failed to send Telegram response: {e}")
            
            # Create application
            app = Application.builder().token(bot_token).build()
            
            # Add message handler for text messages
            app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
            
            # Get bot info
            async def get_bot_info():
                bot = Bot(token=bot_token)
                me = await bot.get_me()
                self.bot_info = {
                    "id": me.id,
                    "username": me.username,
                    "first_name": me.first_name
                }
                # Store username in config for mention detection
                if me.username and not self.config.get("bot_username"):
                    self.config["bot_username"] = me.username
                print(f"🤖 Telegram bot @{me.username} connected")
            
            # Run the bot
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Get bot info first
                loop.run_until_complete(get_bot_info())
                
                # Start polling
                print("📱 Telegram bot starting polling...")
                loop.run_until_complete(app.initialize())
                loop.run_until_complete(app.start())
                
                # Run updater
                updater_task = loop.create_task(app.updater.start_polling())
                
                # Keep running while bot should be running
                while self._bot_running:
                    loop.run_until_complete(asyncio.sleep(1))
                
                # Cleanup
                loop.run_until_complete(app.updater.stop())
                loop.run_until_complete(app.stop())
                loop.run_until_complete(app.shutdown())
                
            finally:
                loop.close()
                
        except ImportError:
            print("⚠️ python-telegram-bot not installed. Run: pip install python-telegram-bot")
            self._bot_running = False
        except Exception as e:
            print(f"⚠️ Telegram bot error: {e}")
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
    
    def get_user_memory_detail(self, user_id: int, chat_id: int) -> Dict:
        """Get detailed memory for a specific user."""
        return self.user_memory.get_user_memory(user_id, chat_id)
    
    def clear_user_memory(self, user_id: int, chat_id: int):
        """Clear memory for a specific user."""
        self.user_memory.clear_user(user_id, chat_id)
        return {"success": True, "message": f"Memory cleared for user {user_id}"}
    
    def clear_all_memories(self):
        """Clear all user memories."""
        self.user_memory.clear_all()
        return {"success": True, "message": "All user memories cleared"}
    
    def add_user_fact(self, user_id: int, chat_id: int, fact: str):
        """Manually add a fact about a user."""
        self.user_memory.add_fact(user_id, chat_id, fact)
        return {"success": True, "message": "Fact added"}
