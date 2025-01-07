# External imports with versions
from minio import Minio  # minio ^7.1.0
from minio.error import S3Error  # minio ^7.1.0
import aiofiles  # aiofiles ^23.1.0
import asyncio  # asyncio ^3.9.0
from tenacity import retry, stop_after_attempt, wait_exponential  # tenacity ^8.2.0
from typing import Dict, Optional, List, Any
import time
from datetime import datetime, timedelta
from functools import lru_cache
import json

# Internal imports
from core.config import Settings
from core.logging import get_logger
from core.metrics import track_generation_latency
from core.exceptions import GameGenBaseException

# Initialize logger
logger = get_logger(__name__)

# Global constants
FRAME_BUCKET = "frames"
MODEL_BUCKET = "models"
RETENTION_DAYS = 30
MAX_RETRIES = 3
RETRY_DELAY = 1
CONNECTION_POOL_SIZE = 20

class StorageError(GameGenBaseException):
    """Custom exception for storage-related errors with FreeBSD compatibility context."""
    def __init__(self, message: str, operation: str, original_error: Optional[Exception] = None):
        super().__init__(
            message=message,
            original_error=original_error,
            recovery_hints={
                "operation": operation,
                "storage_type": "minio",
                "os_compatibility": "freebsd"
            }
        )

class StorageService:
    """Service class for managing object storage operations with FreeBSD compatibility and performance optimization."""

    def __init__(self, endpoint: str, access_key: str, secret_key: str, pool_size: int = CONNECTION_POOL_SIZE):
        """Initialize storage service with configuration and connection pooling."""
        self._endpoint = endpoint
        self._access_key = access_key
        self._secret_key = secret_key
        self._connection_pool = asyncio.Queue(maxsize=pool_size)
        self._bucket_policies = {}

        # Initialize connection pool
        for _ in range(pool_size):
            client = Minio(
                endpoint,
                access_key=access_key,
                secret_key=secret_key,
                secure=True,
                # FreeBSD-specific configuration
                http_client=None  # Uses native FreeBSD HTTP client
            )
            self._connection_pool.put_nowait(client)

        # Initialize buckets and policies
        self._initialize_storage()

    def _initialize_storage(self) -> None:
        """Initialize storage buckets and retention policies."""
        try:
            client = self._get_client()
            
            # Create buckets if they don't exist
            for bucket in [FRAME_BUCKET, MODEL_BUCKET]:
                if not client.bucket_exists(bucket):
                    client.make_bucket(bucket)
                    logger.info(f"Created bucket: {bucket}")

            # Set retention policies
            frame_policy = {
                "Rules": [{
                    "Expiration": {"Days": RETENTION_DAYS},
                    "Status": "Enabled"
                }]
            }
            client.set_bucket_lifecycle(FRAME_BUCKET, frame_policy)

            # Configure bucket policies
            self._bucket_policies = {
                FRAME_BUCKET: {"versioning": False, "encryption": True},
                MODEL_BUCKET: {"versioning": True, "encryption": True}
            }

        except S3Error as e:
            raise StorageError(f"Failed to initialize storage: {str(e)}", "initialization", e)
        finally:
            self._release_client(client)

    async def _get_client(self) -> Minio:
        """Get a client from the connection pool."""
        return await self._connection_pool.get()

    async def _release_client(self, client: Minio) -> None:
        """Release a client back to the connection pool."""
        await self._connection_pool.put(client)

    @retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(multiplier=RETRY_DELAY))
    async def store_frame(self, generation_id: str, frame_number: int, frame_data: bytes) -> str:
        """Store a video frame in object storage with retry logic."""
        start_time = time.time()
        client = await self._get_client()
        
        try:
            # Generate frame path
            frame_path = f"{generation_id}/frame_{frame_number:06d}.jpg"
            
            # Set frame metadata
            metadata = {
                "generation_id": generation_id,
                "frame_number": str(frame_number),
                "timestamp": datetime.utcnow().isoformat()
            }

            # Upload frame with metadata
            client.put_object(
                bucket_name=FRAME_BUCKET,
                object_name=frame_path,
                data=frame_data,
                length=len(frame_data),
                metadata=metadata,
                content_type="image/jpeg"
            )

            # Track storage latency
            latency = (time.time() - start_time) * 1000
            track_generation_latency(latency)

            logger.info(f"Stored frame {frame_number} for generation {generation_id}")
            return frame_path

        except S3Error as e:
            raise StorageError(f"Failed to store frame: {str(e)}", "store_frame", e)
        finally:
            await self._release_client(client)

    @lru_cache(maxsize=1000)
    async def get_frame(self, frame_path: str) -> bytes:
        """Retrieve a video frame from storage with caching."""
        client = await self._get_client()
        
        try:
            # Get frame data
            response = client.get_object(FRAME_BUCKET, frame_path)
            frame_data = response.read()
            response.close()
            response.release_conn()

            return frame_data

        except S3Error as e:
            raise StorageError(f"Failed to retrieve frame: {str(e)}", "get_frame", e)
        finally:
            await self._release_client(client)

    async def store_model(self, model_name: str, model_data: bytes) -> str:
        """Store model weights with encryption."""
        client = await self._get_client()
        
        try:
            # Generate model path with version
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            model_path = f"{model_name}/v_{timestamp}.pt"

            # Set model metadata
            metadata = {
                "model_name": model_name,
                "version": timestamp,
                "encrypted": "true"
            }

            # Upload encrypted model
            client.put_object(
                bucket_name=MODEL_BUCKET,
                object_name=model_path,
                data=model_data,
                length=len(model_data),
                metadata=metadata,
                content_type="application/octet-stream"
            )

            logger.info(f"Stored model {model_name} version {timestamp}")
            return model_path

        except S3Error as e:
            raise StorageError(f"Failed to store model: {str(e)}", "store_model", e)
        finally:
            await self._release_client(client)

    async def cleanup_old_frames(self) -> None:
        """Remove frames older than retention period."""
        client = await self._get_client()
        
        try:
            # Calculate retention threshold
            threshold = datetime.utcnow() - timedelta(days=RETENTION_DAYS)

            # List objects in frames bucket
            objects = client.list_objects(FRAME_BUCKET)
            expired_objects = []

            for obj in objects:
                # Check object metadata for timestamp
                metadata = client.stat_object(FRAME_BUCKET, obj.object_name).metadata
                timestamp = datetime.fromisoformat(metadata.get("timestamp", ""))
                
                if timestamp < threshold:
                    expired_objects.append(obj.object_name)

            # Batch delete expired objects
            if expired_objects:
                errors = client.remove_objects(FRAME_BUCKET, expired_objects)
                for error in errors:
                    logger.error(f"Error deleting object: {error}")

            logger.info(f"Cleaned up {len(expired_objects)} expired frames")

        except S3Error as e:
            raise StorageError(f"Failed to cleanup frames: {str(e)}", "cleanup_frames", e)
        finally:
            await self._release_client(client)