# External imports with versions
import redis  # redis ^4.5.0
import numpy as np  # numpy ^1.23.0
import asyncio  # asyncio ^3.9.0
import msgpack  # msgpack ^1.0.5
from typing import Dict, Optional, Tuple, Any
from functools import wraps
import time

# Internal imports
from core.config import Settings
from core.metrics import track_generation_latency
from core.exceptions import GameGenBaseException

# Global constants
FRAME_CACHE_TTL = 86400  # 24 hours in seconds
MODEL_CACHE_TTL = 3600   # 1 hour in seconds
SESSION_CACHE_TTL = 86400  # 24 hours in seconds
MAX_FRAME_SIZE = 2097152  # 2MB max frame size
POOL_SIZE = 100  # Connection pool size
RETRY_ATTEMPTS = 3
EVICTION_POLICY = 'volatile-lru'

class CacheError(GameGenBaseException):
    """Custom exception class for cache-related errors with enhanced error context."""
    
    def __init__(self, message: str, operation: str, cache_key: str, 
                 error_context: Dict[str, Any], recovery_hint: str):
        super().__init__(
            message=message,
            recovery_hints={"operation": operation, "cache_key": cache_key, 
                          "context": error_context, "recovery": recovery_hint}
        )
        self.operation = operation
        self.cache_key = cache_key
        self.error_context = error_context
        self.recovery_hint = recovery_hint

def retry_on_failure(max_attempts: int = RETRY_ATTEMPTS):
    """Decorator for retrying failed cache operations."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
            raise CacheError(
                message=f"Operation failed after {max_attempts} attempts",
                operation=func.__name__,
                cache_key=kwargs.get('frame_key', 'unknown'),
                error_context={"last_error": str(last_error)},
                recovery_hint="Check Redis connection and retry"
            )
        return wrapper
    return decorator

class CacheService:
    """Redis-based caching service with FreeBSD optimizations and connection pooling."""
    
    def __init__(self, settings: Settings):
        """Initialize Redis client with FreeBSD-optimized configuration."""
        self._cache_config = settings.redis_config
        
        # FreeBSD-optimized Redis connection pool
        self._connection_pool = redis.ConnectionPool(
            host=self._cache_config.get('host', 'localhost'),
            port=self._cache_config.get('port', 6379),
            db=self._cache_config.get('db', 0),
            max_connections=POOL_SIZE,
            socket_timeout=2.0,
            socket_connect_timeout=1.0,
            socket_keepalive=True,
            socket_keepalive_options={
                'tcp_keepidle': 60,
                'tcp_keepintvl': 15,
                'tcp_keepcnt': 3
            }
        )
        
        # Initialize Redis client with optimized configuration
        self._redis_client = redis.Redis(
            connection_pool=self._connection_pool,
            decode_responses=False,
            retry_on_timeout=True,
            health_check_interval=30
        )
        
        # Configure cache policies
        self._redis_client.config_set('maxmemory-policy', EVICTION_POLICY)
        self._redis_client.config_set('maxmemory', '8gb')
        
        # Initialize metrics tracking
        self._metrics = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }

    @track_generation_latency
    @retry_on_failure()
    async def cache_frame(self, frame_key: str, frame_data: np.ndarray, 
                         metadata: Dict[str, Any]) -> bool:
        """Cache a video frame with compression and performance optimization."""
        try:
            # Validate frame size
            frame_size = frame_data.nbytes
            if frame_size > MAX_FRAME_SIZE:
                raise CacheError(
                    message="Frame size exceeds maximum limit",
                    operation="cache_frame",
                    cache_key=frame_key,
                    error_context={"size": frame_size, "max_size": MAX_FRAME_SIZE},
                    recovery_hint="Reduce frame resolution or quality"
                )
            
            # Compress frame data
            compressed_data = msgpack.packb({
                'frame': frame_data.tobytes(),
                'shape': frame_data.shape,
                'dtype': str(frame_data.dtype),
                'metadata': metadata
            })
            
            # Cache with TTL
            success = self._redis_client.setex(
                frame_key,
                FRAME_CACHE_TTL,
                compressed_data
            )
            
            if success:
                self._metrics['hits'] += 1
            
            return bool(success)
            
        except Exception as e:
            raise CacheError(
                message="Failed to cache frame",
                operation="cache_frame",
                cache_key=frame_key,
                error_context={"error": str(e)},
                recovery_hint="Verify frame data integrity and retry"
            )

    @track_generation_latency
    async def get_frame(self, frame_key: str, 
                       with_metadata: bool = True) -> Tuple[Optional[np.ndarray], dict]:
        """Retrieve and decompress cached frame with fallback handling."""
        try:
            # Get compressed data
            compressed_data = self._redis_client.get(frame_key)
            
            if not compressed_data:
                self._metrics['misses'] += 1
                return None, {}
            
            # Decompress and reconstruct frame
            cached_data = msgpack.unpackb(compressed_data)
            frame_array = np.frombuffer(
                cached_data['frame'],
                dtype=np.dtype(cached_data['dtype'])
            ).reshape(cached_data['shape'])
            
            self._metrics['hits'] += 1
            
            return frame_array, cached_data['metadata'] if with_metadata else {}
            
        except Exception as e:
            raise CacheError(
                message="Failed to retrieve frame",
                operation="get_frame",
                cache_key=frame_key,
                error_context={"error": str(e)},
                recovery_hint="Check frame cache integrity"
            )

    async def cache_model_state(self, model_key: str, model_state: Dict[str, Any], 
                              version: str) -> bool:
        """Cache model state with versioning and consistency checks."""
        try:
            # Prepare versioned key
            versioned_key = f"{model_key}:v{version}"
            
            # Compress state data
            compressed_state = msgpack.packb({
                'state': model_state,
                'version': version,
                'timestamp': time.time()
            })
            
            # Cache with TTL
            success = self._redis_client.setex(
                versioned_key,
                MODEL_CACHE_TTL,
                compressed_state
            )
            
            return bool(success)
            
        except Exception as e:
            raise CacheError(
                message="Failed to cache model state",
                operation="cache_model_state",
                cache_key=model_key,
                error_context={"version": version, "error": str(e)},
                recovery_hint="Verify model state integrity"
            )

    async def clear_frame_cache(self, generation_id: str, force: bool = False) -> bool:
        """Clear cached frames with safety checks and metrics update."""
        try:
            # Pattern for frame keys
            pattern = f"frame:{generation_id}:*"
            
            # Get keys matching pattern
            keys = self._redis_client.keys(pattern)
            
            if not keys and not force:
                return False
            
            # Delete matched keys
            deleted = self._redis_client.delete(*keys)
            self._metrics['evictions'] += deleted
            
            return bool(deleted)
            
        except Exception as e:
            raise CacheError(
                message="Failed to clear frame cache",
                operation="clear_frame_cache",
                cache_key=generation_id,
                error_context={"error": str(e)},
                recovery_hint="Retry with force=True"
            )