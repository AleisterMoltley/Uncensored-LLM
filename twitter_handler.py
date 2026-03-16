"""
Twitter Integration Handler for LocalLLM
Scans Twitter for tweets matching assigned tasks and responds using the LLM.
Supports both thread-based and Celery-based background scanning.
Extended with advanced Tweepy v2 filters for precise tweet filtering.
"""

import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Dict, List, Any, Callable, TypedDict

from utils import PersistentStorage


class TwitterV2Filters(TypedDict, total=False):
    """
    Configuration for Twitter API v2 search filters.
    These filters extend the basic keyword search with advanced operators.
    """
    # Content type filters
    has_media: bool  # Only tweets with media (images/videos)
    has_links: bool  # Only tweets with links
    has_hashtags: bool  # Only tweets with hashtags
    has_mentions: bool  # Only tweets with @mentions
    has_images: bool  # Only tweets with images
    has_videos: bool  # Only tweets with videos
    has_geo: bool  # Only tweets with geo data
    
    # Exclusion filters
    exclude_retweets: bool  # Exclude retweets (default True)
    exclude_replies: bool  # Exclude replies (default True)
    exclude_quotes: bool  # Exclude quote tweets
    exclude_nullcast: bool  # Exclude promoted tweets
    
    # Engagement filters
    min_retweets: int  # Minimum retweet count
    min_likes: int  # Minimum like count
    min_replies: int  # Minimum reply count
    
    # Content filters
    language: str  # Tweet language (ISO 639-1, e.g., 'en', 'de')
    is_verified: bool  # Only from verified accounts
    is_not_nullcast: bool  # Exclude promoted content
    
    # Context filters (requires Academic Research access for some)
    conversation_id: str  # Only tweets in specific conversation
    context_entity_ids: List[str]  # Twitter context entity IDs
    
    # Time filters
    max_age_hours: int  # Maximum age in hours (default 3)

# Logger for Twitter handler
logger = logging.getLogger("llm_servant.twitter")

# Twitter history persistence
TWITTER_DIR = Path(__file__).parent / "twitter_data"
TWITTER_DIR.mkdir(exist_ok=True)
HISTORY_FILE = TWITTER_DIR / "tweet_history.json.gz"


class TwitterHandler:
    """
    Handles Twitter integration for the LocalLLM system.
    - Scans Twitter for tweets matching assigned tasks
    - Filters tweets not older than 3 hours
    - Generates responses using the LLM
    - Posts replies to tweets (automatically, no approval needed)
    """
    
    def __init__(self, config: Dict, llm_callback: Callable[[str], str],
                 personality_prompt_builder: Optional[Callable[[str], str]] = None):
        """
        Initialize the Twitter handler.
        
        Args:
            config: Full configuration from config.json
            llm_callback: Function to generate LLM responses
            personality_prompt_builder: Optional function to build prompts using active personality
        """
        self.full_config = config
        self.config = config.get("twitter", {})
        self.llm_callback = llm_callback
        self.personality_prompt_builder = personality_prompt_builder
        self.api = None
        self.client = None
        self._scanner_thread: Optional[threading.Thread] = None
        self._scanner_running = False
        self._celery_task_id: Optional[str] = None
        self._use_celery = self._check_celery_available()
        self._history: List[Dict] = []
        
        # Initialize persistent storage for history
        self._storage = PersistentStorage(HISTORY_FILE)
        self._load_history()
    
    def _check_celery_available(self) -> bool:
        """
        Check if Celery is available and enabled for background tasks.
        
        Returns:
            bool: True if Celery should be used for background tasks
        """
        try:
            from celery_app import is_celery_available, get_celery_config
            
            config = get_celery_config()
            if not config.get("enabled", False):
                return False
            
            return is_celery_available()
        except ImportError:
            return False
        except Exception as e:
            logger.debug("Celery availability check failed: %s", e)
            return False
        
    def _load_history(self):
        """Load tweet history from disk."""
        loaded = self._storage.load(default={"history": []})
        self._history = loaded.get("history", [])
    
    def _save_history(self):
        """Save tweet history to disk."""
        self._storage.save({"history": self._history})
    
    def configure(self, twitter_config: Dict):
        """
        Configure Twitter API credentials and settings.
        
        Args:
            twitter_config: Dictionary containing:
                - api_key: Twitter API Key
                - api_secret: Twitter API Secret
                - access_token: Twitter Access Token
                - access_token_secret: Twitter Access Token Secret
                - bearer_token: Twitter Bearer Token (for v2 API)
                - task: Description of what tweets to look for
                - search_keywords: Keywords to search for
                - scan_interval_minutes: How often to scan (default: 5)
                - auto_reply: Whether to automatically reply (default: False)
        """
        self.config = twitter_config
        self._init_api()
        return self.get_status()
    
    def _init_api(self) -> bool:
        """Initialize Twitter API client."""
        try:
            import tweepy
            
            api_key = self.config.get("api_key", "")
            api_secret = self.config.get("api_secret", "")
            access_token = self.config.get("access_token", "")
            access_token_secret = self.config.get("access_token_secret", "")
            bearer_token = self.config.get("bearer_token", "")
            
            if not all([api_key, api_secret, access_token, access_token_secret]):
                return False
            
            # OAuth 1.0a for posting tweets
            auth = tweepy.OAuth1UserHandler(
                api_key, api_secret,
                access_token, access_token_secret
            )
            self.api = tweepy.API(auth, wait_on_rate_limit=True)
            
            # OAuth 2.0 Client for searching (v2 API)
            if bearer_token:
                self.client = tweepy.Client(
                    bearer_token=bearer_token,
                    consumer_key=api_key,
                    consumer_secret=api_secret,
                    access_token=access_token,
                    access_token_secret=access_token_secret,
                    wait_on_rate_limit=True
                )
            
            # Verify credentials
            self.api.verify_credentials()
            return True
            
        except ImportError:
            print("⚠️ tweepy not installed. Run: pip install tweepy")
            return False
        except Exception as e:
            print(f"⚠️ Twitter API init error: {e}")
            return False
    
    def get_status(self) -> Dict:
        """Get current Twitter handler status including background task info and v2 filters."""
        status = {
            "configured": bool(self.api),
            "scanning": self._scanner_running or self._celery_task_id is not None,
            "task": self.config.get("task", ""),
            "search_keywords": self.config.get("search_keywords", []),
            "scan_interval_minutes": self.config.get("scan_interval_minutes", 5),
            "auto_reply": self.config.get("auto_reply", False),
            "tweets_found_total": len(self._history),
            "tweets_replied_total": len([h for h in self._history if h.get("replied")]),
            "backend": "celery" if self._use_celery else "thread",
            "v2_filters": self.config.get("v2_filters", {})
        }
        
        # Add Celery task info if applicable
        if self._celery_task_id:
            status["celery_task_id"] = self._celery_task_id
            try:
                from background_tasks import get_task_status
                task_status = get_task_status(self._celery_task_id)
                status["celery_task_status"] = task_status.get("status", "unknown")
            except ImportError:
                pass
        
        return status
    
    def _build_v2_query(self, keywords: List[str], 
                         v2_filters: Optional[TwitterV2Filters] = None) -> str:
        """
        Build a Twitter API v2 search query with advanced filters.
        
        Args:
            keywords: List of search keywords
            v2_filters: Optional TwitterV2Filters configuration
            
        Returns:
            Formatted query string for Twitter API v2
        """
        if not v2_filters:
            v2_filters = self.config.get("v2_filters", {})
        
        # Build keyword part
        query_parts = []
        for kw in keywords:
            if " " in kw:
                query_parts.append(f'"{kw}"')
            else:
                query_parts.append(kw)
        
        query = f"({' OR '.join(query_parts)})"
        
        # Add content type filters (has: operators)
        if v2_filters.get("has_media"):
            query += " has:media"
        if v2_filters.get("has_links"):
            query += " has:links"
        if v2_filters.get("has_hashtags"):
            query += " has:hashtags"
        if v2_filters.get("has_mentions"):
            query += " has:mentions"
        if v2_filters.get("has_images"):
            query += " has:images"
        if v2_filters.get("has_videos"):
            query += " has:videos"
        if v2_filters.get("has_geo"):
            query += " has:geo"
        
        # Add exclusion filters (is: and -is: operators)
        exclude_retweets = v2_filters.get("exclude_retweets", True)
        if exclude_retweets:
            query += " -is:retweet"
        
        exclude_replies = v2_filters.get("exclude_replies", True)
        if exclude_replies:
            query += " -is:reply"
        
        if v2_filters.get("exclude_quotes"):
            query += " -is:quote"
        
        if v2_filters.get("exclude_nullcast") or v2_filters.get("is_not_nullcast"):
            query += " -is:nullcast"
        
        # Add verified filter
        if v2_filters.get("is_verified"):
            query += " is:verified"
        
        # Add language filter (use v2_filters or fallback to default)
        language = v2_filters.get("language", "en")
        if language:
            query += f" lang:{language}"
        
        # Add conversation filter
        conversation_id = v2_filters.get("conversation_id")
        if conversation_id:
            query += f" conversation_id:{conversation_id}"
        
        # Add context entity filters
        context_entities = v2_filters.get("context_entity_ids", [])
        for entity_id in context_entities:
            query += f" context:{entity_id}"
        
        return query
    
    def _filter_by_engagement(self, tweets: List[Dict], 
                               v2_filters: Optional[TwitterV2Filters] = None) -> List[Dict]:
        """
        Filter tweets by engagement metrics (applied post-search).
        
        Args:
            tweets: List of tweet dictionaries
            v2_filters: Optional TwitterV2Filters configuration
            
        Returns:
            Filtered list of tweets
        """
        if not v2_filters:
            v2_filters = self.config.get("v2_filters", {})
        
        min_retweets = v2_filters.get("min_retweets", 0)
        min_likes = v2_filters.get("min_likes", 0)
        min_replies = v2_filters.get("min_replies", 0)
        
        # If no engagement filters, return all tweets
        if not (min_retweets or min_likes or min_replies):
            return tweets
        
        filtered = []
        for tweet in tweets:
            metrics = tweet.get("metrics", {})
            retweets = metrics.get("retweet_count", 0)
            likes = metrics.get("like_count", 0)
            replies = metrics.get("reply_count", 0)
            
            if retweets >= min_retweets and likes >= min_likes and replies >= min_replies:
                filtered.append(tweet)
        
        return filtered
    
    def get_v2_filters(self) -> TwitterV2Filters:
        """
        Get the current v2 filter configuration.
        
        Returns:
            Current v2 filter settings
        """
        return self.config.get("v2_filters", {})
    
    def set_v2_filters(self, filters: TwitterV2Filters) -> Dict:
        """
        Update the v2 filter configuration.
        
        Args:
            filters: New filter settings to apply
            
        Returns:
            Updated status
        """
        self.config["v2_filters"] = filters
        return {
            "success": True,
            "v2_filters": filters
        }
    
    def search_tweets(self, max_results: int = 20, 
                      v2_filters: Optional[TwitterV2Filters] = None) -> List[Dict]:
        """
        Search for tweets matching the configured task with extended v2 filters.
        Uses Twitter API v2 operators for precise filtering.
        
        Args:
            max_results: Maximum number of tweets to return
            v2_filters: Optional v2 filters (uses config if not provided)
            
        Returns:
            List of tweet dictionaries
        """
        if not self.client:
            return []
        
        keywords = self.config.get("search_keywords", [])
        if not keywords:
            return []
        
        # Use provided filters or get from config
        if v2_filters is None:
            v2_filters = self.config.get("v2_filters", {})
        
        # Build the v2 query with advanced filters
        query = self._build_v2_query(keywords, v2_filters)
        
        # Calculate time window (use max_age_hours from filters or default 3)
        max_age_hours = v2_filters.get("max_age_hours", 3)
        start_time = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        
        try:
            # Extended tweet fields for v2 API
            tweet_fields = [
                "created_at", 
                "author_id", 
                "public_metrics", 
                "lang",
                "context_annotations",  # Topic/entity context
                "entities",  # URLs, mentions, hashtags, etc.
                "geo",  # Geographic information
                "possibly_sensitive",  # Content sensitivity flag
                "source"  # Client used to post
            ]
            
            # Extended user fields
            user_fields = [
                "username", 
                "name",
                "verified",  # Verification status
                "description",  # User bio
                "public_metrics"  # Follower/following counts
            ]
            
            # Expanded data inclusions
            expansions = [
                "author_id",
                "geo.place_id",  # Geographic data
                "entities.mentions.username"  # Mentioned users
            ]
            
            # Search using Twitter API v2
            response = self.client.search_recent_tweets(
                query=query,
                max_results=min(max_results, 100),
                start_time=start_time,
                tweet_fields=tweet_fields,
                user_fields=user_fields,
                expansions=expansions
            )
            
            tweets = []
            users = {}
            
            # Build user lookup
            if response.includes and "users" in response.includes:
                for user in response.includes["users"]:
                    users[user.id] = {
                        "username": user.username,
                        "name": user.name,
                        "verified": getattr(user, "verified", False),
                        "description": getattr(user, "description", ""),
                        "metrics": getattr(user, "public_metrics", {})
                    }
            
            # Process tweets
            if response.data:
                for tweet in response.data:
                    tweet_age = datetime.now(timezone.utc) - tweet.created_at
                    max_age_seconds = max_age_hours * 3600
                    
                    if tweet_age.total_seconds() <= max_age_seconds:
                        user_info = users.get(tweet.author_id, {})
                        
                        # Extract context annotations if available
                        context = []
                        if hasattr(tweet, "context_annotations") and tweet.context_annotations:
                            for annotation in tweet.context_annotations:
                                domain = annotation.get("domain", {})
                                entity = annotation.get("entity", {})
                                context.append({
                                    "domain": domain.get("name", ""),
                                    "entity": entity.get("name", "")
                                })
                        
                        # Extract entities if available
                        entities = {}
                        if hasattr(tweet, "entities") and tweet.entities:
                            entities = {
                                "hashtags": [h.get("tag", "") for h in tweet.entities.get("hashtags", [])],
                                "mentions": [m.get("username", "") for m in tweet.entities.get("mentions", [])],
                                "urls": [u.get("expanded_url", "") for u in tweet.entities.get("urls", [])]
                            }
                        
                        tweets.append({
                            "id": str(tweet.id),
                            "text": tweet.text,
                            "author_id": str(tweet.author_id),
                            "author_username": user_info.get("username", "unknown"),
                            "author_name": user_info.get("name", "Unknown"),
                            "author_verified": user_info.get("verified", False),
                            "author_description": user_info.get("description", ""),
                            "author_metrics": user_info.get("metrics", {}),
                            "created_at": tweet.created_at.isoformat(),
                            "age_minutes": int(tweet_age.total_seconds() / 60),
                            "metrics": tweet.public_metrics if tweet.public_metrics else {},
                            "lang": getattr(tweet, "lang", ""),
                            "context_annotations": context,
                            "entities": entities,
                            "possibly_sensitive": getattr(tweet, "possibly_sensitive", False),
                            "source": getattr(tweet, "source", "")
                        })
            
            # Apply post-search engagement filters
            tweets = self._filter_by_engagement(tweets, v2_filters)
            
            return tweets
            
        except Exception as e:
            logger.error("Twitter search error: %s", e)
            print(f"⚠️ Twitter search error: {e}")
            return []
    
    def generate_response(self, tweet: Dict) -> str:
        """
        Generate a response to a tweet using the LLM with active personality.
        Runs autonomously without requiring user approval.
        
        Args:
            tweet: Tweet dictionary
            
        Returns:
            Generated response text
        """
        task = self.config.get("task", "respond helpfully and professionally")
        
        # Build the user query for the tweet
        user_query = f"""Du antwortest auf einen Tweet auf Twitter. Deine Aufgabe ist: {task}

Tweet von @{tweet.get('author_username', 'user')}:
"{tweet.get('text', '')}"

Generiere eine kurze, passende Antwort (max 280 Zeichen). Sei hilfreich und engagiert.
Gib nur den Antworttext aus, nichts anderes."""

        try:
            # Use personality-based prompt if available
            if self.personality_prompt_builder:
                prompt = self.personality_prompt_builder(user_query)
                response = self.llm_callback(prompt)
            else:
                response = self.llm_callback(user_query)
            
            # Truncate to Twitter's character limit
            if len(response) > 280:
                response = response[:277] + "..."
            return response.strip()
        except Exception as e:
            print(f"⚠️ LLM response error: {e}")
            return ""
    
    def reply_to_tweet(self, tweet_id: str, response_text: str) -> Dict:
        """
        Post a reply to a tweet.
        
        Args:
            tweet_id: ID of the tweet to reply to
            response_text: Text of the reply
            
        Returns:
            Result dictionary with success status
        """
        if not self.client:
            return {"success": False, "error": "Twitter API not configured"}
        
        if not response_text:
            return {"success": False, "error": "Empty response text"}
        
        try:
            # Post reply using v2 API
            result = self.client.create_tweet(
                text=response_text,
                in_reply_to_tweet_id=tweet_id
            )
            
            return {
                "success": True,
                "reply_tweet_id": str(result.data["id"]),
                "text": response_text
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def process_tweet(self, tweet: Dict, auto_reply: bool = False) -> Dict:
        """
        Process a single tweet: generate response and optionally reply.
        
        Args:
            tweet: Tweet dictionary
            auto_reply: Whether to automatically post the reply
            
        Returns:
            Processing result dictionary
        """
        # Check if we already processed this tweet
        processed_ids = {h["tweet_id"] for h in self._history}
        if tweet["id"] in processed_ids:
            return {"skipped": True, "reason": "already processed"}
        
        # Generate response
        response = self.generate_response(tweet)
        
        result = {
            "tweet_id": tweet["id"],
            "tweet_text": tweet["text"],
            "author": tweet.get("author_username", "unknown"),
            "generated_response": response,
            "replied": False,
            "processed_at": datetime.now(timezone.utc).isoformat()
        }
        
        # Auto-reply if enabled
        if auto_reply and response:
            reply_result = self.reply_to_tweet(tweet["id"], response)
            result["replied"] = reply_result.get("success", False)
            result["reply_tweet_id"] = reply_result.get("reply_tweet_id")
            result["reply_error"] = reply_result.get("error")
        
        # Add to history
        self._history.append(result)
        self._save_history()
        
        return result
    
    def scan_and_process(self) -> List[Dict]:
        """
        Scan for new tweets and process them.
        
        Returns:
            List of processing results
        """
        tweets = self.search_tweets()
        auto_reply = self.config.get("auto_reply", False)
        
        results = []
        for tweet in tweets:
            result = self.process_tweet(tweet, auto_reply=auto_reply)
            if not result.get("skipped"):
                results.append(result)
        
        return results
    
    def start_scanner(self) -> Dict[str, Any]:
        """
        Start the background tweet scanner.
        Uses Celery if available and enabled, otherwise falls back to threading.
        
        Returns:
            dict: Result with success status and backend used
        """
        if self._scanner_running or self._celery_task_id:
            return {"success": False, "message": "Scanner already running"}
        
        if not self.client:
            return {"success": False, "message": "Twitter API not configured"}
        
        # Check if Celery is available and should be used
        self._use_celery = self._check_celery_available()
        
        if self._use_celery:
            return self._start_celery_scanner()
        else:
            return self._start_thread_scanner()
    
    def _start_celery_scanner(self) -> Dict[str, Any]:
        """
        Start scanner using Celery for background processing.
        
        Returns:
            dict: Result with task ID
        """
        try:
            from background_tasks import schedule_twitter_scan
            
            result = schedule_twitter_scan()
            
            if result.get("scheduled"):
                self._celery_task_id = result.get("task_id")
                logger.info("Twitter scanner started via Celery, task_id: %s", self._celery_task_id)
                return {
                    "success": True,
                    "message": "Scanner started (Celery)",
                    "backend": "celery",
                    "task_id": self._celery_task_id
                }
            else:
                # Celery scheduling failed, fall back to thread
                logger.warning("Celery scheduling failed, falling back to thread")
                return self._start_thread_scanner()
                
        except ImportError as e:
            logger.warning("Celery import failed: %s, using thread scanner", e)
            return self._start_thread_scanner()
        except Exception as e:
            logger.error("Celery scanner start failed: %s", e)
            return self._start_thread_scanner()
    
    def _start_thread_scanner(self) -> Dict[str, Any]:
        """
        Start scanner using traditional threading.
        
        Returns:
            dict: Result
        """
        self._scanner_running = True
        self._scanner_thread = threading.Thread(target=self._scanner_loop, daemon=True)
        self._scanner_thread.start()
        
        logger.info("Twitter scanner started via threading")
        return {
            "success": True,
            "message": "Scanner started (Thread)",
            "backend": "thread"
        }
    
    def stop_scanner(self) -> Dict[str, Any]:
        """
        Stop the background tweet scanner.
        Handles both Celery and thread-based scanners.
        
        Returns:
            dict: Result
        """
        stopped = False
        
        # Stop Celery task if active
        if self._celery_task_id:
            try:
                from background_tasks import revoke_task
                revoke_task(self._celery_task_id, terminate=True)
                logger.info("Celery task revoked: %s", self._celery_task_id)
                stopped = True
            except Exception as e:
                logger.warning("Failed to revoke Celery task: %s", e)
            finally:
                self._celery_task_id = None
        
        # Stop thread-based scanner
        if self._scanner_running:
            self._scanner_running = False
            stopped = True
            logger.info("Thread scanner stopped")
        
        if stopped:
            return {"success": True, "message": "Scanner stopped"}
        else:
            return {"success": True, "message": "Scanner was not running"}
    
    def _scanner_loop(self):
        """Background scanner loop (thread-based)."""
        interval = self.config.get("scan_interval_minutes", 5) * 60
        
        while self._scanner_running:
            try:
                results = self.scan_and_process()
                if results:
                    logger.info("Twitter: Processed %d new tweets", len(results))
            except Exception as e:
                logger.error("Scanner error: %s", e)
            
            # Sleep in small intervals to allow quick shutdown
            for _ in range(int(interval)):
                if not self._scanner_running:
                    break
                time.sleep(1)
    
    def get_history(self, limit: int = 50) -> List[Dict]:
        """
        Get tweet processing history.
        
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
        """Clear tweet processing history."""
        self._history = []
        self._save_history()
        return {"success": True, "message": "History cleared"}
    
    def scan_async(self) -> Dict[str, Any]:
        """
        Trigger an asynchronous Twitter scan.
        Uses Celery if available, otherwise runs synchronously.
        
        Returns:
            dict: Result with task ID if async, or scan results if sync
        """
        if not self.client:
            return {"success": False, "error": "Twitter API not configured"}
        
        # Try to use Celery for async processing
        if self._check_celery_available():
            try:
                from background_tasks import schedule_twitter_scan
                
                result = schedule_twitter_scan()
                if result.get("scheduled"):
                    return {
                        "success": True,
                        "async": True,
                        "task_id": result.get("task_id"),
                        "message": "Scan scheduled via Celery"
                    }
            except Exception as e:
                logger.warning("Async scan failed, falling back to sync: %s", e)
        
        # Fall back to synchronous scan
        try:
            results = self.scan_and_process()
            return {
                "success": True,
                "async": False,
                "tweets_processed": len(results),
                "results": results
            }
        except Exception as e:
            logger.error("Sync scan failed: %s", e)
            return {
                "success": False,
                "error": str(e)
            }
    
    def process_tweet_async(self, tweet: Dict) -> Dict[str, Any]:
        """
        Process a single tweet asynchronously.
        Uses Celery if available, otherwise processes synchronously.
        
        Args:
            tweet: Tweet dictionary to process
            
        Returns:
            dict: Result with task ID if async, or processing result if sync
        """
        if self._check_celery_available():
            try:
                from background_tasks import twitter_process_tweet
                
                task = twitter_process_tweet.delay(tweet)
                return {
                    "success": True,
                    "async": True,
                    "task_id": task.id,
                    "message": "Tweet processing scheduled"
                }
            except Exception as e:
                logger.warning("Async tweet processing failed: %s", e)
        
        # Fall back to synchronous processing
        auto_reply = self.config.get("auto_reply", False)
        result = self.process_tweet(tweet, auto_reply=auto_reply)
        return {
            "success": True,
            "async": False,
            "result": result
        }
    
    def manual_reply(self, tweet_id: str, response_text: str) -> Dict:
        """
        Manually reply to a tweet from history.
        
        Args:
            tweet_id: ID of the tweet to reply to
            response_text: Text of the reply
            
        Returns:
            Result dictionary
        """
        result = self.reply_to_tweet(tweet_id, response_text)
        
        # Update history
        for entry in self._history:
            if entry["tweet_id"] == tweet_id:
                entry["replied"] = result.get("success", False)
                entry["reply_tweet_id"] = result.get("reply_tweet_id")
                entry["manual_reply"] = True
                break
        
        self._save_history()
        return result
