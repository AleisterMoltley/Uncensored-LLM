"""
Celery Application Configuration for LocalLLM
Provides background task processing for Twitter scanning and other async operations.
"""

import os
import json
from pathlib import Path

# Celery configuration
# Default to using Redis as message broker
# Can be overridden via environment variables or config.json
DEFAULT_BROKER_URL = "redis://localhost:6379/0"
DEFAULT_RESULT_BACKEND = "redis://localhost:6379/0"

# Path to config file
CONFIG_PATH = Path(__file__).parent / "config.json"


def get_celery_config():
    """
    Load Celery configuration from config.json or environment variables.
    
    Priority:
    1. Environment variables (CELERY_BROKER_URL, CELERY_RESULT_BACKEND)
    2. config.json celery section
    3. Default values
    
    Returns:
        dict: Celery configuration
    """
    config = {}
    
    # Try to load from config.json
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r") as f:
                full_config = json.load(f)
                celery_config = full_config.get("celery", {})
                config["broker_url"] = celery_config.get("broker_url", DEFAULT_BROKER_URL)
                config["result_backend"] = celery_config.get("result_backend", DEFAULT_RESULT_BACKEND)
                config["enabled"] = celery_config.get("enabled", False)
        except (json.JSONDecodeError, IOError):
            pass
    
    # Environment variables override config.json
    config["broker_url"] = os.environ.get("CELERY_BROKER_URL", config.get("broker_url", DEFAULT_BROKER_URL))
    config["result_backend"] = os.environ.get("CELERY_RESULT_BACKEND", config.get("result_backend", DEFAULT_RESULT_BACKEND))
    
    # Celery enabled by default if environment variable is set
    if os.environ.get("CELERY_ENABLED"):
        config["enabled"] = os.environ.get("CELERY_ENABLED", "false").lower() in ("true", "1", "yes")
    else:
        config.setdefault("enabled", False)
    
    return config


def create_celery_app():
    """
    Create and configure the Celery application.
    
    Returns:
        Celery: Configured Celery application instance
    """
    try:
        from celery import Celery
    except ImportError:
        # If Celery is not installed, return None
        return None
    
    config = get_celery_config()
    
    app = Celery(
        "llm_servant",
        broker=config["broker_url"],
        backend=config["result_backend"],
        include=["background_tasks"]
    )
    
    # Configure Celery app
    app.conf.update(
        # Task result settings
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        
        # Timezone settings
        timezone="UTC",
        enable_utc=True,
        
        # Task execution settings
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        
        # Worker settings
        worker_prefetch_multiplier=1,
        worker_concurrency=2,  # Conservative for LLM tasks
        
        # Task time limits
        task_soft_time_limit=300,  # 5 minutes soft limit
        task_time_limit=600,       # 10 minutes hard limit
        
        # Beat scheduler for periodic tasks
        beat_schedule={
            "twitter-scan-periodic": {
                "task": "background_tasks.twitter_scan_task",
                "schedule": 300.0,  # Every 5 minutes (default, overridden by config)
                "options": {"queue": "twitter"}
            }
        },
        
        # Task routing
        task_routes={
            "background_tasks.twitter_*": {"queue": "twitter"},
            "background_tasks.llm_*": {"queue": "llm"},
        }
    )
    
    return app


# Create global Celery app instance (may be None if Celery not installed)
celery_app = create_celery_app()


def is_celery_available():
    """
    Check if Celery is available and properly configured.
    
    Returns:
        bool: True if Celery can be used for background tasks
    """
    if celery_app is None:
        return False
    
    config = get_celery_config()
    if not config.get("enabled", False):
        return False
    
    # Try to ping the broker
    try:
        celery_app.control.ping(timeout=1)
        return True
    except Exception:
        return False


def get_celery_status():
    """
    Get current Celery status and statistics.
    
    Returns:
        dict: Celery status information
    """
    config = get_celery_config()
    
    status = {
        "installed": celery_app is not None,
        "enabled": config.get("enabled", False),
        "broker_url": config.get("broker_url", "").replace(":6379", ":****"),  # Mask port/password
        "workers_active": 0,
        "workers": []
    }
    
    if celery_app is not None and config.get("enabled", False):
        try:
            # Get active workers
            inspection = celery_app.control.inspect()
            active = inspection.active()
            
            if active:
                status["workers_active"] = len(active)
                status["workers"] = list(active.keys())
        except Exception:
            pass
    
    return status
