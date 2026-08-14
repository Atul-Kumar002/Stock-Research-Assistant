import time
import threading
from typing import Dict, Any, Optional

class TTLCache:
    """
    In-memory thread-safe TTL Cache for backend stock data and research results.
    """
    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            item = self._cache.get(key)
            if item is None:
                return None
            if time.time() > item["expires_at"]:
                del self._cache[key]
                return None
            return item["value"]

    def set(self, key: str, value: Any, ttl_seconds: int = 300) -> None:
        with self._lock:
            self._cache[key] = {
                "value": value,
                "expires_at": time.time() + ttl_seconds
            }

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

# Global cache instance
backend_cache = TTLCache()
