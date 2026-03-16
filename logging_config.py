"""
Centralized Logging Configuration for LLM Servant
Provides RotatingFileHandler with configurable file rotation.
All handlers should use get_logger() to obtain their logger instance.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


# ============================================================
# Configuration Constants
# ============================================================

# Log directory - can be overridden via environment variable
LOG_DIR = Path(os.environ.get("LLM_SERVANT_LOG_DIR", Path(__file__).parent / "logs"))

# Log file settings
DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10 MB per file
DEFAULT_BACKUP_COUNT = 5  # Keep 5 backup files
DEFAULT_LOG_FILE = "llm_servant.log"

# Debug mode from environment variable
DEBUG_MODE = os.environ.get("LLM_SERVANT_DEBUG", "false").lower() in ("true", "1", "yes")

# Root logger name
ROOT_LOGGER_NAME = "llm_servant"

# Global flag to track if logging has been configured
_logging_configured = False


# ============================================================
# Log Format Configuration
# ============================================================

# Detailed format for file logging
FILE_LOG_FORMAT = (
    "%(asctime)s - %(name)s - %(levelname)s - "
    "%(filename)s:%(lineno)d - %(funcName)s - %(message)s"
)

# Simpler format for console logging
CONSOLE_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Date format
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    log_dir: Optional[Path] = None,
    log_file: str = DEFAULT_LOG_FILE,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
    console_output: bool = True,
    debug_mode: Optional[bool] = None
) -> logging.Logger:
    """
    Configure the root logger with rotating file handler and optional console handler.
    
    This function should be called once at application startup.
    Subsequent calls will return the existing configured logger.
    
    Args:
        log_dir: Directory for log files (default: ./logs or LLM_SERVANT_LOG_DIR env var)
        log_file: Name of the log file (default: llm_servant.log)
        max_bytes: Maximum size of log file before rotation (default: 10 MB)
        backup_count: Number of backup files to keep (default: 5)
        console_output: Whether to also log to console (default: True)
        debug_mode: Override DEBUG_MODE if specified
        
    Returns:
        Configured root logger instance
        
    Example:
        >>> logger = setup_logging()
        >>> logger.info("Application started")
    """
    global _logging_configured
    
    root_logger = logging.getLogger(ROOT_LOGGER_NAME)
    
    # If already configured, return existing logger
    if _logging_configured:
        return root_logger
    
    # Determine debug mode
    is_debug = debug_mode if debug_mode is not None else DEBUG_MODE
    log_level = logging.DEBUG if is_debug else logging.INFO
    
    # Set logger level
    root_logger.setLevel(log_level)
    
    # Ensure log directory exists
    actual_log_dir = log_dir or LOG_DIR
    actual_log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file_path = actual_log_dir / log_file
    
    # Create rotating file handler
    file_handler = RotatingFileHandler(
        filename=log_file_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8"
    )
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter(FILE_LOG_FORMAT, datefmt=DATE_FORMAT)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # Create console handler if requested
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter(CONSOLE_LOG_FORMAT, datefmt=DATE_FORMAT)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    _logging_configured = True
    
    root_logger.debug(
        "Logging configured: file=%s, max_bytes=%d, backup_count=%d, debug=%s",
        log_file_path, max_bytes, backup_count, is_debug
    )
    
    return root_logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance for the specified module.
    
    If logging hasn't been configured yet, this will set up logging with default settings.
    
    Args:
        name: Logger name suffix (e.g., "twitter", "telegram", "discord")
              If None, returns the root llm_servant logger.
              
    Returns:
        Logger instance
        
    Example:
        >>> logger = get_logger("twitter")
        >>> logger.info("Starting Twitter handler")
    """
    # Ensure logging is configured
    if not _logging_configured:
        setup_logging()
    
    if name:
        return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")
    return logging.getLogger(ROOT_LOGGER_NAME)


def add_file_handler(
    logger: logging.Logger,
    log_file: str,
    log_dir: Optional[Path] = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT
) -> RotatingFileHandler:
    """
    Add an additional rotating file handler to a specific logger.
    
    Useful for creating separate log files for different components.
    
    Args:
        logger: Logger instance to add handler to
        log_file: Name of the log file
        log_dir: Directory for log file (default: LOG_DIR)
        max_bytes: Maximum size before rotation
        backup_count: Number of backup files to keep
        
    Returns:
        The created RotatingFileHandler
        
    Example:
        >>> logger = get_logger("twitter")
        >>> add_file_handler(logger, "twitter.log")
    """
    actual_log_dir = log_dir or LOG_DIR
    actual_log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file_path = actual_log_dir / log_file
    
    handler = RotatingFileHandler(
        filename=log_file_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8"
    )
    
    is_debug = DEBUG_MODE
    handler.setLevel(logging.DEBUG if is_debug else logging.INFO)
    formatter = logging.Formatter(FILE_LOG_FORMAT, datefmt=DATE_FORMAT)
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    
    return handler


def get_log_dir() -> Path:
    """
    Get the configured log directory path.
    
    Returns:
        Path to the log directory
    """
    return LOG_DIR


def reset_logging() -> None:
    """
    Reset logging configuration.
    
    This removes all handlers from the root logger and allows
    setup_logging() to be called again with new settings.
    
    Primarily useful for testing.
    """
    global _logging_configured
    
    root_logger = logging.getLogger(ROOT_LOGGER_NAME)
    
    # Remove all handlers
    for handler in root_logger.handlers[:]:
        handler.close()
        root_logger.removeHandler(handler)
    
    _logging_configured = False
