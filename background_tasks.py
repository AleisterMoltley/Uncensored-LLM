"""
Background Tasks Module for LocalLLM
Defines Celery tasks for Twitter scanning and other background operations.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional

from logging_config import get_logger

# Setup logging with file rotation
logger = get_logger("tasks")

# Path to config file
CONFIG_PATH = Path(__file__).parent / "config.json"


def load_config() -> Dict[str, Any]:
    """Load configuration from config.json."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    return {}


def get_celery_app():
    """
    Get the Celery app instance.
    This is done lazily to avoid circular imports.
    
    Returns:
        Celery app instance or None if not available
    """
    try:
        from celery_app import celery_app
        return celery_app
    except ImportError:
        return None


# Try to get the Celery app
_celery_app = get_celery_app()

# Define tasks only if Celery is available
if _celery_app is not None:
    
    @_celery_app.task(bind=True, name="background_tasks.twitter_scan_task")
    def twitter_scan_task(self) -> Dict[str, Any]:
        """
        Background task to scan Twitter for tweets and process them.
        This task is designed to run periodically via Celery Beat.
        
        Returns:
            dict: Results of the scan including tweets processed
        """
        logger.info("Starting Twitter scan task")
        
        try:
            # Load fresh config
            config = load_config()
            twitter_config = config.get("twitter", {})
            
            # Check if Twitter is configured
            if not twitter_config.get("api_key"):
                logger.debug("Twitter not configured, skipping scan")
                return {
                    "success": False,
                    "error": "Twitter not configured",
                    "tweets_processed": 0,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            
            # Create TwitterHandler instance for this task
            from twitter_handler import TwitterHandler
            
            def llm_callback(prompt: str) -> str:
                """Generate LLM response for tweets."""
                # Import here to avoid circular imports
                from langchain_ollama import OllamaLLM
                
                llm = OllamaLLM(
                    model=config.get("model", "dolphin-llama3:8b"),
                    temperature=config.get("temperature", 0.5),
                    num_ctx=config.get("num_ctx", 2048),
                    num_predict=512,
                    repeat_penalty=1.1,
                    top_k=40,
                    top_p=0.9,
                )
                return llm.invoke(prompt)
            
            handler = TwitterHandler(config, llm_callback)
            handler.configure(twitter_config)
            
            # Perform the scan
            results = handler.scan_and_process()
            
            logger.info("Twitter scan completed: %d tweets processed", len(results))
            
            return {
                "success": True,
                "tweets_processed": len(results),
                "results": results,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except ImportError as e:
            logger.warning("Required module not available: %s", e)
            return {
                "success": False,
                "error": f"Module not available: {str(e)}",
                "tweets_processed": 0,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            logger.error("Twitter scan task failed: %s", e)
            # Celery will handle retries based on configuration
            raise self.retry(exc=e, countdown=60, max_retries=3)
    
    
    @_celery_app.task(bind=True, name="background_tasks.twitter_process_tweet")
    def twitter_process_tweet(self, tweet_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Background task to process a single tweet.
        
        Args:
            tweet_data: Tweet data dictionary
            
        Returns:
            dict: Processing result
        """
        logger.info("Processing tweet: %s", tweet_data.get("id", "unknown"))
        
        try:
            config = load_config()
            twitter_config = config.get("twitter", {})
            
            from twitter_handler import TwitterHandler
            
            def llm_callback(prompt: str) -> str:
                from langchain_ollama import OllamaLLM
                llm = OllamaLLM(
                    model=config.get("model", "dolphin-llama3:8b"),
                    temperature=config.get("temperature", 0.5),
                    num_ctx=config.get("num_ctx", 2048),
                    num_predict=512,
                )
                return llm.invoke(prompt)
            
            handler = TwitterHandler(config, llm_callback)
            handler.configure(twitter_config)
            
            auto_reply = twitter_config.get("auto_reply", False)
            result = handler.process_tweet(tweet_data, auto_reply=auto_reply)
            
            return result
            
        except Exception as e:
            logger.error("Tweet processing failed: %s", e)
            raise self.retry(exc=e, countdown=30, max_retries=3)
    
    
    @_celery_app.task(name="background_tasks.llm_generate_response")
    def llm_generate_response(prompt: str, config_overrides: Optional[Dict] = None) -> str:
        """
        Background task to generate an LLM response.
        Useful for offloading heavy LLM work to background workers.
        
        Args:
            prompt: The prompt to send to the LLM
            config_overrides: Optional config overrides
            
        Returns:
            str: Generated response
        """
        try:
            config = load_config()
            if config_overrides:
                config.update(config_overrides)
            
            from langchain_ollama import OllamaLLM
            
            llm = OllamaLLM(
                model=config.get("model", "dolphin-llama3:8b"),
                temperature=config.get("temperature", 0.5),
                num_ctx=config.get("num_ctx", 2048),
                num_predict=512,
            )
            
            return llm.invoke(prompt)
            
        except Exception as e:
            logger.error("LLM response generation failed: %s", e)
            raise


# Fallback functions for when Celery is not available
def twitter_scan_sync() -> Dict[str, Any]:
    """
    Synchronous Twitter scan for when Celery is not available.
    This is a wrapper around the TwitterHandler's scan method.
    
    Returns:
        dict: Scan results
    """
    try:
        config = load_config()
        twitter_config = config.get("twitter", {})
        
        if not twitter_config.get("api_key"):
            return {
                "success": False,
                "error": "Twitter not configured",
                "tweets_processed": 0
            }
        
        from twitter_handler import TwitterHandler
        
        def llm_callback(prompt: str) -> str:
            from langchain_ollama import OllamaLLM
            llm = OllamaLLM(
                model=config.get("model", "dolphin-llama3:8b"),
                temperature=config.get("temperature", 0.5),
                num_ctx=config.get("num_ctx", 2048),
                num_predict=512,
            )
            return llm.invoke(prompt)
        
        handler = TwitterHandler(config, llm_callback)
        handler.configure(twitter_config)
        results = handler.scan_and_process()
        
        return {
            "success": True,
            "tweets_processed": len(results),
            "results": results
        }
        
    except Exception as e:
        logger.error("Synchronous Twitter scan failed: %s", e)
        return {
            "success": False,
            "error": str(e),
            "tweets_processed": 0
        }


def schedule_twitter_scan() -> Dict[str, Any]:
    """
    Schedule a Twitter scan - uses Celery if available, otherwise runs synchronously.
    
    Returns:
        dict: Scheduling result including task_id if using Celery
    """
    celery_app = get_celery_app()
    
    if celery_app is not None:
        try:
            from celery_app import is_celery_available
            
            if is_celery_available():
                task = twitter_scan_task.delay()
                return {
                    "scheduled": True,
                    "task_id": task.id,
                    "backend": "celery"
                }
        except Exception as e:
            logger.warning("Failed to schedule via Celery, falling back to sync: %s", e)
    
    # Fall back to synchronous execution
    logger.info("Running Twitter scan synchronously (Celery not available)")
    result = twitter_scan_sync()
    return {
        "scheduled": False,
        "executed": True,
        "backend": "sync",
        "result": result
    }


def get_task_status(task_id: str) -> Dict[str, Any]:
    """
    Get the status of a background task.
    
    Args:
        task_id: The Celery task ID
        
    Returns:
        dict: Task status information
    """
    celery_app = get_celery_app()
    
    if celery_app is None:
        return {
            "error": "Celery not available",
            "task_id": task_id
        }
    
    try:
        from celery.result import AsyncResult
        
        result = AsyncResult(task_id, app=celery_app)
        
        status = {
            "task_id": task_id,
            "status": result.status,
            "ready": result.ready(),
            "successful": result.successful() if result.ready() else None
        }
        
        if result.ready():
            try:
                status["result"] = result.get(timeout=1)
            except Exception as e:
                status["error"] = str(e)
        
        return status
        
    except Exception as e:
        return {
            "task_id": task_id,
            "error": str(e)
        }


def revoke_task(task_id: str, terminate: bool = False) -> Dict[str, Any]:
    """
    Revoke/cancel a pending task.
    
    Args:
        task_id: The Celery task ID
        terminate: Whether to terminate the task if it's already running
        
    Returns:
        dict: Revocation result
    """
    celery_app = get_celery_app()
    
    if celery_app is None:
        return {
            "success": False,
            "error": "Celery not available"
        }
    
    try:
        celery_app.control.revoke(task_id, terminate=terminate)
        return {
            "success": True,
            "task_id": task_id,
            "revoked": True
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
