"""
Tests for Logging Configuration Module
"""

import os
import tempfile
import shutil
import logging
from pathlib import Path
import unittest

from logging_config import (
    setup_logging,
    get_logger,
    add_file_handler,
    reset_logging,
    get_log_dir,
    ROOT_LOGGER_NAME,
    DEFAULT_MAX_BYTES,
    DEFAULT_BACKUP_COUNT,
)


class TestLoggingConfig(unittest.TestCase):
    """Test cases for logging configuration module."""
    
    def setUp(self):
        """Reset logging before each test."""
        reset_logging()
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up after each test."""
        reset_logging()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_setup_logging_creates_logger(self):
        """Test that setup_logging creates a properly configured logger."""
        logger = setup_logging(log_dir=self.temp_dir, console_output=False)
        
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, ROOT_LOGGER_NAME)
    
    def test_setup_logging_creates_log_dir(self):
        """Test that setup_logging creates the log directory if it doesn't exist."""
        log_dir = self.temp_dir / "new_log_dir"
        self.assertFalse(log_dir.exists())
        
        setup_logging(log_dir=log_dir, console_output=False)
        
        self.assertTrue(log_dir.exists())
    
    def test_setup_logging_creates_log_file(self):
        """Test that setup_logging creates a log file."""
        logger = setup_logging(log_dir=self.temp_dir, console_output=False)
        logger.info("Test message")
        
        log_file = self.temp_dir / "llm_servant.log"
        self.assertTrue(log_file.exists())
    
    def test_setup_logging_custom_file_name(self):
        """Test that setup_logging can use a custom log file name."""
        logger = setup_logging(
            log_dir=self.temp_dir,
            log_file="custom.log",
            console_output=False
        )
        logger.info("Test message")
        
        log_file = self.temp_dir / "custom.log"
        self.assertTrue(log_file.exists())
    
    def test_setup_logging_idempotent(self):
        """Test that calling setup_logging multiple times returns the same logger."""
        logger1 = setup_logging(log_dir=self.temp_dir, console_output=False)
        logger2 = setup_logging(log_dir=self.temp_dir, console_output=False)
        
        self.assertIs(logger1, logger2)
    
    def test_get_logger_returns_child_logger(self):
        """Test that get_logger returns a child logger."""
        setup_logging(log_dir=self.temp_dir, console_output=False)
        
        logger = get_logger("test_module")
        
        self.assertEqual(logger.name, f"{ROOT_LOGGER_NAME}.test_module")
    
    def test_get_logger_root_logger(self):
        """Test that get_logger with no arguments returns root logger."""
        setup_logging(log_dir=self.temp_dir, console_output=False)
        
        logger = get_logger()
        
        self.assertEqual(logger.name, ROOT_LOGGER_NAME)
    
    def test_debug_mode_sets_level(self):
        """Test that debug mode sets the log level to DEBUG."""
        logger = setup_logging(
            log_dir=self.temp_dir,
            console_output=False,
            debug_mode=True
        )
        
        self.assertEqual(logger.level, logging.DEBUG)
    
    def test_non_debug_mode_sets_info_level(self):
        """Test that non-debug mode sets the log level to INFO."""
        logger = setup_logging(
            log_dir=self.temp_dir,
            console_output=False,
            debug_mode=False
        )
        
        self.assertEqual(logger.level, logging.INFO)
    
    def test_add_file_handler(self):
        """Test adding an additional file handler."""
        logger = setup_logging(log_dir=self.temp_dir, console_output=False)
        
        handler = add_file_handler(
            logger,
            "additional.log",
            log_dir=self.temp_dir
        )
        
        logger.info("Test message")
        
        # Check the additional log file was created
        additional_log = self.temp_dir / "additional.log"
        self.assertTrue(additional_log.exists())
    
    def test_reset_logging(self):
        """Test that reset_logging removes handlers."""
        logger = setup_logging(log_dir=self.temp_dir, console_output=False)
        initial_handlers = len(logger.handlers)
        
        reset_logging()
        
        self.assertEqual(len(logger.handlers), 0)
    
    def test_log_message_format(self):
        """Test that log messages are formatted correctly."""
        logger = setup_logging(log_dir=self.temp_dir, console_output=False)
        
        test_message = "Test log message 12345"
        logger.info(test_message)
        
        log_file = self.temp_dir / "llm_servant.log"
        with open(log_file, 'r') as f:
            content = f.read()
        
        self.assertIn(test_message, content)
        self.assertIn("INFO", content)
        self.assertIn(ROOT_LOGGER_NAME, content)
    
    def test_get_log_dir(self):
        """Test that get_log_dir returns the configured log directory."""
        log_dir = get_log_dir()
        
        self.assertIsInstance(log_dir, Path)


class TestLoggingRotation(unittest.TestCase):
    """Test cases for log file rotation."""
    
    def setUp(self):
        """Reset logging before each test."""
        reset_logging()
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up after each test."""
        reset_logging()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_rotating_file_handler_is_used(self):
        """Test that a RotatingFileHandler is used."""
        from logging.handlers import RotatingFileHandler
        
        logger = setup_logging(log_dir=self.temp_dir, console_output=False)
        
        has_rotating_handler = any(
            isinstance(h, RotatingFileHandler)
            for h in logger.handlers
        )
        
        self.assertTrue(has_rotating_handler)
    
    def test_file_rotation_on_size(self):
        """Test that log files rotate when they reach the size limit."""
        from logging.handlers import RotatingFileHandler
        
        # Use a very small max size to force rotation
        small_max_bytes = 100
        
        logger = setup_logging(
            log_dir=self.temp_dir,
            max_bytes=small_max_bytes,
            backup_count=2,
            console_output=False
        )
        
        # Write enough messages to cause rotation
        for i in range(100):
            logger.info(f"Test message number {i} with some extra content to make it longer")
        
        # Check that backup files were created
        log_files = list(self.temp_dir.glob("llm_servant.log*"))
        self.assertGreater(len(log_files), 1)


if __name__ == "__main__":
    unittest.main()
