"""
Tests for Configuration Schema Validation

This module tests the Pydantic schema validation for config.json
to ensure misconfigurations are detected early.
"""

import json
import tempfile
import unittest
from pathlib import Path
from pydantic import ValidationError

from server import (
    AppConfig,
    TwitterV2FiltersConfig,
    TwitterConfig,
    TelegramRateLimitConfig,
    TelegramConfig,
    CeleryConfig,
    RedisConfigModel,
    PersonalityConfig,
    validate_config,
    ConfigValidationError,
    LLMServantApp,
)


class TestAppConfigValidation(unittest.TestCase):
    """Test cases for AppConfig Pydantic model validation."""
    
    def test_valid_minimal_config(self):
        """Test that a minimal valid config passes validation."""
        config = {"model": "dolphin-llama3:8b"}
        validated = validate_config(config)
        self.assertEqual(validated.model, "dolphin-llama3:8b")
        self.assertEqual(validated.port, 7777)  # default
        self.assertEqual(validated.temperature, 0.7)  # default
    
    def test_valid_full_config(self):
        """Test validation with all fields present."""
        config = {
            "model": "dolphin-llama3:8b",
            "embedding_model": "nomic-embed-text",
            "host": "127.0.0.1",
            "port": 7777,
            "chunk_size": 600,
            "chunk_overlap": 100,
            "top_k": 5,
            "num_ctx": 2048,
            "temperature": 0.7,
            "max_memory_messages": 6,
            "force_uncensored": True,
            "low_memory_mode": False,
            "system_prompt": "You are an AI.",
            "active_personality": "uncensored_pdf",
            "personalities": {
                "uncensored_pdf": {
                    "name": "Unfiltered PDF Personality",
                    "description": "Test personality",
                    "use_knowledge_memory": True,
                    "use_uncensored_boost": True,
                    "system_prompt": "Be helpful."
                }
            },
            "twitter": {
                "api_key": "",
                "api_secret": "",
                "access_token": "",
                "access_token_secret": "",
                "bearer_token": "",
                "task": "",
                "search_keywords": [],
                "scan_interval_minutes": 5,
                "auto_reply": True,
                "v2_filters": {
                    "exclude_retweets": True,
                    "exclude_replies": True,
                    "language": "en"
                }
            },
            "telegram": {
                "bot_token": "",
                "bot_username": "",
                "respond_to_mentions": True,
                "respond_to_direct": True,
                "auto_respond": True,
                "task": "",
                "rate_limit": {
                    "enabled": True,
                    "messages_per_second": 1.0,
                    "messages_per_minute": 20
                }
            },
            "celery": {
                "enabled": False,
                "broker_url": "redis://localhost:6379/0",
                "result_backend": "redis://localhost:6379/0"
            },
            "redis": {
                "enabled": False,
                "url": "redis://localhost:6379/0",
                "embedding_cache_ttl": 86400
            }
        }
        validated = validate_config(config)
        self.assertEqual(validated.model, "dolphin-llama3:8b")
        self.assertEqual(validated.port, 7777)
        self.assertEqual(len(validated.personalities), 1)
    
    def test_missing_model_field(self):
        """Test that missing 'model' field raises validation error."""
        config = {"port": 7777}
        with self.assertRaises(ValidationError) as ctx:
            validate_config(config)
        errors = ctx.exception.errors()
        self.assertTrue(any(e["loc"] == ("model",) for e in errors))
    
    def test_invalid_port_too_high(self):
        """Test that port > 65535 raises validation error."""
        config = {"model": "test", "port": 99999}
        with self.assertRaises(ValidationError) as ctx:
            validate_config(config)
        errors = ctx.exception.errors()
        self.assertTrue(any(e["loc"] == ("port",) for e in errors))
    
    def test_invalid_port_too_low(self):
        """Test that port < 1 raises validation error."""
        config = {"model": "test", "port": 0}
        with self.assertRaises(ValidationError) as ctx:
            validate_config(config)
        errors = ctx.exception.errors()
        self.assertTrue(any(e["loc"] == ("port",) for e in errors))
    
    def test_invalid_temperature_too_high(self):
        """Test that temperature > 2.0 raises validation error."""
        config = {"model": "test", "temperature": 2.5}
        with self.assertRaises(ValidationError) as ctx:
            validate_config(config)
        errors = ctx.exception.errors()
        self.assertTrue(any(e["loc"] == ("temperature",) for e in errors))
    
    def test_invalid_temperature_negative(self):
        """Test that temperature < 0 raises validation error."""
        config = {"model": "test", "temperature": -0.5}
        with self.assertRaises(ValidationError) as ctx:
            validate_config(config)
        errors = ctx.exception.errors()
        self.assertTrue(any(e["loc"] == ("temperature",) for e in errors))
    
    def test_invalid_num_ctx_too_low(self):
        """Test that num_ctx < 512 raises validation error."""
        config = {"model": "test", "num_ctx": 256}
        with self.assertRaises(ValidationError) as ctx:
            validate_config(config)
        errors = ctx.exception.errors()
        self.assertTrue(any(e["loc"] == ("num_ctx",) for e in errors))
    
    def test_invalid_chunk_overlap_exceeds_chunk_size(self):
        """Test that chunk_overlap >= chunk_size raises validation error."""
        config = {"model": "test", "chunk_size": 100, "chunk_overlap": 150}
        with self.assertRaises(ValidationError) as ctx:
            validate_config(config)
        errors = ctx.exception.errors()
        self.assertTrue(any(e["loc"] == ("chunk_overlap",) for e in errors))
    
    def test_invalid_extra_field(self):
        """Test that extra/unknown fields raise validation error."""
        config = {"model": "test", "unknown_field": "value"}
        with self.assertRaises(ValidationError) as ctx:
            validate_config(config)
        errors = ctx.exception.errors()
        self.assertTrue(any(e["loc"] == ("unknown_field",) for e in errors))
    
    def test_invalid_personality_missing_name(self):
        """Test that personality without name raises validation error."""
        config = {
            "model": "test",
            "personalities": {
                "test": {
                    "description": "Test without name"
                }
            }
        }
        with self.assertRaises(ValidationError) as ctx:
            validate_config(config)
        errors = ctx.exception.errors()
        self.assertTrue(any("name" in str(e["loc"]) for e in errors))
    
    def test_invalid_active_personality_not_found(self):
        """Test that active_personality not in personalities raises error."""
        config = {
            "model": "test",
            "active_personality": "nonexistent",
            "personalities": {
                "existing": {
                    "name": "Existing Personality"
                }
            }
        }
        with self.assertRaises(ValidationError) as ctx:
            validate_config(config)
        errors = ctx.exception.errors()
        # Model validators have empty loc tuple, check msg content instead
        self.assertTrue(any("active_personality" in e["msg"] for e in errors))


class TestNestedConfigValidation(unittest.TestCase):
    """Test cases for nested configuration models."""
    
    def test_twitter_v2_filters_defaults(self):
        """Test Twitter v2 filters with default values."""
        filters = TwitterV2FiltersConfig()
        self.assertTrue(filters.exclude_retweets)
        self.assertTrue(filters.exclude_replies)
        self.assertEqual(filters.language, "en")
        self.assertEqual(filters.max_age_hours, 3)
    
    def test_twitter_v2_filters_invalid_min_retweets(self):
        """Test that negative min_retweets raises error."""
        with self.assertRaises(ValidationError):
            TwitterV2FiltersConfig(min_retweets=-1)
    
    def test_telegram_rate_limit_invalid_messages_per_second(self):
        """Test that non-positive messages_per_second raises error."""
        with self.assertRaises(ValidationError):
            TelegramRateLimitConfig(messages_per_second=0)
    
    def test_redis_config_invalid_ttl(self):
        """Test that negative TTL raises error."""
        with self.assertRaises(ValidationError):
            RedisConfigModel(embedding_cache_ttl=-1)
    
    def test_celery_config_defaults(self):
        """Test Celery config with default values."""
        celery = CeleryConfig()
        self.assertFalse(celery.enabled)
        self.assertEqual(celery.broker_url, "redis://localhost:6379/0")


class TestLLMServantAppValidation(unittest.TestCase):
    """Test cases for LLMServantApp config validation integration."""
    
    def test_app_validates_config_on_init(self):
        """Test that LLMServantApp validates config during initialization."""
        invalid_config = {"port": 99999}  # Missing model, invalid port
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(invalid_config, f)
            temp_path = Path(f.name)
        
        try:
            with self.assertRaises(ConfigValidationError) as ctx:
                LLMServantApp(config_path=temp_path)
            self.assertIn("model", str(ctx.exception))
        finally:
            temp_path.unlink()
    
    def test_app_skip_validation_flag(self):
        """Test that skip_validation=True bypasses validation."""
        invalid_config = {"port": 99999}  # Invalid but should be allowed
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(invalid_config, f)
            temp_path = Path(f.name)
        
        try:
            # Should not raise, even with invalid config
            app = LLMServantApp(config_path=temp_path, skip_validation=True)
            self.assertEqual(app.config["port"], 99999)
        finally:
            temp_path.unlink()
    
    def test_app_with_valid_config(self):
        """Test that LLMServantApp initializes with valid config."""
        valid_config = {
            "model": "test-model",
            "embedding_model": "nomic-embed-text",
            "port": 8080
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(valid_config, f)
            temp_path = Path(f.name)
        
        try:
            app = LLMServantApp(config_path=temp_path)
            self.assertEqual(app.config["model"], "test-model")
            self.assertEqual(app.config["port"], 8080)
        finally:
            temp_path.unlink()


class TestConfigValidationError(unittest.TestCase):
    """Test cases for ConfigValidationError exception."""
    
    def test_error_contains_message(self):
        """Test that ConfigValidationError contains the error message."""
        error = ConfigValidationError("Test error message", [{"loc": ("test",), "msg": "error"}])
        self.assertEqual(error.message, "Test error message")
        self.assertEqual(str(error), "Test error message")
    
    def test_error_contains_errors_list(self):
        """Test that ConfigValidationError contains the errors list."""
        errors_list = [{"loc": ("field1",), "msg": "error1"}, {"loc": ("field2",), "msg": "error2"}]
        error = ConfigValidationError("Multiple errors", errors_list)
        self.assertEqual(len(error.errors), 2)


if __name__ == '__main__':
    unittest.main()
