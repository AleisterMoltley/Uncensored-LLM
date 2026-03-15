"""
Twitter Integration Handler for LocalLLM
Scans Twitter for tweets matching assigned tasks and responds using the LLM.
"""

import json
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Dict, List, Any, Callable

# Twitter history persistence
TWITTER_DIR = Path(__file__).parent / "twitter_data"
TWITTER_DIR.mkdir(exist_ok=True)
HISTORY_FILE = TWITTER_DIR / "tweet_history.json"


class TwitterHandler:
    """
    Handles Twitter integration for the LocalLLM system.
    - Scans Twitter for tweets matching assigned tasks
    - Filters tweets not older than 3 hours
    - Generates responses using the LLM
    - Posts replies to tweets
    """
    
    def __init__(self, config: Dict, llm_callback: Callable[[str], str]):
        """
        Initialize the Twitter handler.
        
        Args:
            config: Twitter configuration from config.json
            llm_callback: Function to generate LLM responses
        """
        self.config = config.get("twitter", {})
        self.llm_callback = llm_callback
        self.api = None
        self.client = None
        self._scanner_thread: Optional[threading.Thread] = None
        self._scanner_running = False
        self._history: List[Dict] = []
        self._load_history()
        
    def _load_history(self):
        """Load tweet history from disk."""
        try:
            if HISTORY_FILE.exists():
                with open(HISTORY_FILE) as f:
                    self._history = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            # History file may be corrupted or unreadable, start fresh
            print(f"⚠️ Could not load Twitter history: {e}")
            self._history = []
    
    def _save_history(self):
        """Save tweet history to disk."""
        try:
            with open(HISTORY_FILE, "w") as f:
                json.dump(self._history, f, indent=2, ensure_ascii=False)
        except IOError as e:
            # Log the error but don't fail the operation
            print(f"⚠️ Could not save Twitter history: {e}")
    
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
        """Get current Twitter handler status."""
        return {
            "configured": bool(self.api),
            "scanning": self._scanner_running,
            "task": self.config.get("task", ""),
            "search_keywords": self.config.get("search_keywords", []),
            "scan_interval_minutes": self.config.get("scan_interval_minutes", 5),
            "auto_reply": self.config.get("auto_reply", False),
            "tweets_found_total": len(self._history),
            "tweets_replied_total": len([h for h in self._history if h.get("replied")])
        }
    
    def search_tweets(self, max_results: int = 20) -> List[Dict]:
        """
        Search for tweets matching the configured task.
        Only returns tweets from the last 3 hours.
        
        Args:
            max_results: Maximum number of tweets to return
            
        Returns:
            List of tweet dictionaries
        """
        if not self.client:
            return []
        
        keywords = self.config.get("search_keywords", [])
        if not keywords:
            return []
        
        # Build search query from keywords
        query_parts = []
        for kw in keywords:
            if " " in kw:
                query_parts.append(f'"{kw}"')
            else:
                query_parts.append(kw)
        
        # Exclude retweets and replies to get original content
        query = f"({' OR '.join(query_parts)}) -is:retweet -is:reply lang:en"
        
        # Calculate time 3 hours ago
        three_hours_ago = datetime.now(timezone.utc) - timedelta(hours=3)
        
        try:
            # Search using Twitter API v2
            response = self.client.search_recent_tweets(
                query=query,
                max_results=min(max_results, 100),
                start_time=three_hours_ago,
                tweet_fields=["created_at", "author_id", "public_metrics", "lang"],
                user_fields=["username", "name"],
                expansions=["author_id"]
            )
            
            tweets = []
            users = {}
            
            # Build user lookup
            if response.includes and "users" in response.includes:
                for user in response.includes["users"]:
                    users[user.id] = {
                        "username": user.username,
                        "name": user.name
                    }
            
            # Process tweets
            if response.data:
                for tweet in response.data:
                    tweet_age = datetime.now(timezone.utc) - tweet.created_at
                    if tweet_age.total_seconds() <= 3 * 3600:  # 3 hours in seconds
                        user_info = users.get(tweet.author_id, {})
                        tweets.append({
                            "id": str(tweet.id),
                            "text": tweet.text,
                            "author_id": str(tweet.author_id),
                            "author_username": user_info.get("username", "unknown"),
                            "author_name": user_info.get("name", "Unknown"),
                            "created_at": tweet.created_at.isoformat(),
                            "age_minutes": int(tweet_age.total_seconds() / 60),
                            "metrics": tweet.public_metrics if tweet.public_metrics else {}
                        })
            
            return tweets
            
        except Exception as e:
            print(f"⚠️ Twitter search error: {e}")
            return []
    
    def generate_response(self, tweet: Dict) -> str:
        """
        Generate a response to a tweet using the LLM.
        
        Args:
            tweet: Tweet dictionary
            
        Returns:
            Generated response text
        """
        task = self.config.get("task", "respond helpfully and professionally")
        
        prompt = f"""You are responding to a tweet on Twitter. Your task is: {task}

Tweet from @{tweet.get('author_username', 'user')}:
"{tweet.get('text', '')}"

Generate a concise, appropriate response (max 280 characters). Be helpful, professional, and engaging.
Only output the response text, nothing else."""

        try:
            response = self.llm_callback(prompt)
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
    
    def start_scanner(self):
        """Start the background tweet scanner."""
        if self._scanner_running:
            return {"success": False, "message": "Scanner already running"}
        
        if not self.client:
            return {"success": False, "message": "Twitter API not configured"}
        
        self._scanner_running = True
        self._scanner_thread = threading.Thread(target=self._scanner_loop, daemon=True)
        self._scanner_thread.start()
        
        return {"success": True, "message": "Scanner started"}
    
    def stop_scanner(self):
        """Stop the background tweet scanner."""
        self._scanner_running = False
        return {"success": True, "message": "Scanner stopped"}
    
    def _scanner_loop(self):
        """Background scanner loop."""
        interval = self.config.get("scan_interval_minutes", 5) * 60
        
        while self._scanner_running:
            try:
                results = self.scan_and_process()
                if results:
                    print(f"🐦 Twitter: Processed {len(results)} new tweets")
            except Exception as e:
                print(f"⚠️ Scanner error: {e}")
            
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
