"""
Utility classes and functions for LLM Servant.
Contains reusable components for data persistence and other shared functionality.
"""

import json
import gzip
from pathlib import Path
from typing import Any, Dict, Optional, Callable


class PersistentStorage:
    """
    Handles compressed JSON persistence with GZIP compression.
    
    Provides load/save operations with:
    - GZIP compression for reduced file size
    - Size checking and optional compression callback
    - Error handling for corrupted or missing files
    
    Usage:
        storage = PersistentStorage(Path("data.json.gz"))
        data = storage.load(default={"key": "value"})
        storage.save(data)
    """
    
    def __init__(
        self,
        file_path: Path,
        max_size_mb: Optional[float] = None,
        on_size_exceeded: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
    ):
        """
        Initialize persistent storage.
        
        Args:
            file_path: Path to the compressed JSON file (should end with .gz)
            max_size_mb: Optional maximum size in MB before triggering compression callback
            on_size_exceeded: Optional callback function that receives data and returns
                              compressed data when size limit is exceeded
        """
        self.file_path = Path(file_path)
        self.max_size_mb = max_size_mb
        self.on_size_exceeded = on_size_exceeded
    
    def load(self, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Load data from compressed JSON file.
        
        Args:
            default: Default value to return if file doesn't exist or is corrupted
            
        Returns:
            Loaded dictionary or default value
        """
        if default is None:
            default = {}
        
        if not self.file_path.exists():
            return default
        
        try:
            with gzip.open(self.file_path, 'rt', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError, gzip.BadGzipFile) as e:
            print(f"⚠️ Could not load {self.file_path.name}: {e}")
            return default
    
    def save(self, data: Dict[str, Any]) -> bool:
        """
        Save data to compressed JSON file.
        
        If max_size_mb is set and the data exceeds this size,
        the on_size_exceeded callback will be called to compress the data.
        
        Args:
            data: Dictionary to save
            
        Returns:
            True if save was successful, False otherwise
        """
        # Serialize to JSON
        json_data = json.dumps(data, ensure_ascii=False, indent=None)
        
        # Check size and trigger compression if needed
        if self.max_size_mb is not None:
            size_bytes = len(json_data.encode('utf-8'))
            size_mb = size_bytes / (1024 * 1024)
            
            if size_mb > self.max_size_mb and self.on_size_exceeded is not None:
                data = self.on_size_exceeded(data)
                json_data = json.dumps(data, ensure_ascii=False, indent=None)
        
        try:
            with gzip.open(self.file_path, 'wt', encoding='utf-8') as f:
                f.write(json_data)
            return True
        except IOError as e:
            print(f"⚠️ Could not save {self.file_path.name}: {e}")
            return False
    
    def exists(self) -> bool:
        """Check if the storage file exists."""
        return self.file_path.exists()
    
    def get_size_mb(self) -> float:
        """
        Get the compressed file size in MB.
        
        Returns:
            File size in MB, or 0 if file doesn't exist
        """
        if self.file_path.exists():
            return self.file_path.stat().st_size / (1024 * 1024)
        return 0.0
    
    def delete(self) -> bool:
        """
        Delete the storage file.
        
        Returns:
            True if deleted, False if file didn't exist or couldn't be deleted
        """
        try:
            if self.file_path.exists():
                self.file_path.unlink()
                return True
            return False
        except IOError as e:
            print(f"⚠️ Could not delete {self.file_path.name}: {e}")
            return False
