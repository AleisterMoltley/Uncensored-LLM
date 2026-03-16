"""
Tests for low_memory_mode functionality and system stats.
"""
import json
import tempfile
from pathlib import Path
import pytest

# Mock psutil before importing server
import sys
from unittest.mock import MagicMock, patch

# Create mock psutil module
mock_psutil = MagicMock()
mock_memory = MagicMock()
mock_memory.total = 16 * 1024 * 1024 * 1024  # 16 GB
mock_memory.available = 8 * 1024 * 1024 * 1024  # 8 GB
mock_memory.used = 8 * 1024 * 1024 * 1024  # 8 GB
mock_memory.percent = 50.0
mock_psutil.virtual_memory.return_value = mock_memory
sys.modules['psutil'] = mock_psutil

from server import (
    get_effective_top_k,
    get_effective_num_ctx,
    LLMServantApp,
)


class TestLowMemoryMode:
    """Tests for low_memory_mode configuration option."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.config_path = self.temp_path / "config.json"
        self.memory_dir = self.temp_path / "memory"
        self.memory_dir.mkdir()
        
        # Reset singleton
        LLMServantApp._instance = None
        
    def teardown_method(self):
        """Cleanup test fixtures."""
        LLMServantApp._instance = None
        self.temp_dir.cleanup()
        
    def _create_config(self, low_memory_mode: bool = False, top_k: int = 5, num_ctx: int = 2048):
        """Create a test config file."""
        config = {
            "model": "test-model",
            "embedding_model": "test-embed",
            "host": "127.0.0.1",
            "port": 7777,
            "chunk_size": 600,
            "chunk_overlap": 100,
            "top_k": top_k,
            "num_ctx": num_ctx,
            "temperature": 0.7,
            "max_memory_messages": 6,
            "force_uncensored": True,
            "low_memory_mode": low_memory_mode,
            "system_prompt": "Test prompt",
            "active_personality": "custom",
            "personalities": {
                "custom": {
                    "name": "Test",
                    "description": "Test personality",
                    "use_knowledge_memory": False,
                    "use_uncensored_boost": False,
                    "system_prompt": "Test"
                }
            }
        }
        with open(self.config_path, 'w') as f:
            json.dump(config, f)
        return config
    
    def test_low_memory_mode_disabled_top_k(self):
        """Test that top_k uses config value when low_memory_mode is disabled."""
        config = self._create_config(low_memory_mode=False, top_k=5)
        app = LLMServantApp(
            config=config,
            config_path=self.config_path,
            memory_dir=self.memory_dir
        )
        LLMServantApp._instance = app
        
        assert get_effective_top_k() == 5
        
    def test_low_memory_mode_enabled_top_k(self):
        """Test that top_k is reduced to 2 when low_memory_mode is enabled."""
        config = self._create_config(low_memory_mode=True, top_k=5)
        app = LLMServantApp(
            config=config,
            config_path=self.config_path,
            memory_dir=self.memory_dir
        )
        LLMServantApp._instance = app
        
        assert get_effective_top_k() == 2
        
    def test_low_memory_mode_disabled_num_ctx(self):
        """Test that num_ctx uses config value when low_memory_mode is disabled."""
        config = self._create_config(low_memory_mode=False, num_ctx=2048)
        app = LLMServantApp(
            config=config,
            config_path=self.config_path,
            memory_dir=self.memory_dir
        )
        LLMServantApp._instance = app
        
        assert get_effective_num_ctx() == 2048
        
    def test_low_memory_mode_enabled_num_ctx(self):
        """Test that num_ctx is reduced to 1024 when low_memory_mode is enabled."""
        config = self._create_config(low_memory_mode=True, num_ctx=2048)
        app = LLMServantApp(
            config=config,
            config_path=self.config_path,
            memory_dir=self.memory_dir
        )
        LLMServantApp._instance = app
        
        assert get_effective_num_ctx() == 1024
        
    def test_low_memory_mode_in_config(self):
        """Test that low_memory_mode is properly stored in config."""
        config = self._create_config(low_memory_mode=True)
        app = LLMServantApp(
            config=config,
            config_path=self.config_path,
            memory_dir=self.memory_dir
        )
        
        assert app.config.get('low_memory_mode') is True
        
    def test_low_memory_mode_default_false(self):
        """Test that low_memory_mode defaults to False if not specified."""
        config = self._create_config(low_memory_mode=False)
        del config['low_memory_mode']
        
        with open(self.config_path, 'w') as f:
            json.dump(config, f)
            
        app = LLMServantApp(
            config_path=self.config_path,
            memory_dir=self.memory_dir
        )
        LLMServantApp._instance = app
        
        # Should default to False and use normal top_k
        assert get_effective_top_k() == 5
        assert get_effective_num_ctx() == 2048


class TestSystemStats:
    """Tests for system stats functionality using psutil."""
    
    def test_psutil_memory_mock(self):
        """Test that psutil.virtual_memory returns expected mock values."""
        import psutil
        mem = psutil.virtual_memory()
        
        assert mem.total == 16 * 1024 * 1024 * 1024
        assert mem.used == 8 * 1024 * 1024 * 1024
        assert mem.percent == 50.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
