# External imports with versions
import asyncio  # asyncio ^3.9.0
from fastapi import HTTPException  # fastapi ^0.95.0
from sqlalchemy.ext.asyncio import AsyncSession  # sqlalchemy ^1.4.41
import structlog  # structlog ^23.1.0
from aiocache import Cache  # aiocache ^0.12.0
from prometheus_client import Counter, Histogram  # prometheus_client ^0.16.0
from typing import Dict, Any, List, Optional, Tuple
from uuid import UUID
import time

# Internal imports
from models.instructnet.control import ControlProcessor
from db.models.control import Control
from api.schemas.control import ControlBase

# Initialize logging
logger = structlog.get_logger(__name__)

# Initialize metrics
CONTROL_LATENCY = Histogram(
    'control_latency_seconds',
    'Control processing latency in seconds',
    buckets=[0.010, 0.025, 0.050, 0.075, 0.100]
)
CONTROL_SUCCESS = Counter(
    'control_success_total',
    'Successful control operations',
    ['type']
)
CONTROL_ERRORS = Counter(
    'control_errors_total',
    'Failed control operations',
    ['type', 'error']
)

# Cache configuration
CACHE_CONFIG = {
    'ttl': 300,  # 5 minutes
    'namespace': 'control',
    'key_builder': lambda *args, **kwargs: f"{kwargs.get('video_id')}:{kwargs.get('type')}"
}

class ControlService:
    """Enhanced service class for handling video control operations with optimized performance, caching, and monitoring."""

    def __init__(self, db_session: AsyncSession, processor: ControlProcessor):
        """Initialize control service with database session and processor."""
        self._db = db_session
        self._processor = processor
        self._cache = Cache(Cache.MEMORY, **CACHE_CONFIG)
        self._rate_limits = {
            'keyboard': {'max_rate': 60, 'window': 1.0},  # 60 per second
            'environment': {'max_rate': 10, 'window': 1.0},  # 10 per second
            'instruction': {'max_rate': 5, 'window': 1.0}   # 5 per second
        }
        self._error_counts = {}

    async def process_control(
        self, 
        generation_id: UUID, 
        video_id: UUID, 
        control_data: ControlBase
    ) -> Dict[str, Any]:
        """Process control signal with comprehensive monitoring and error handling."""
        start_time = time.perf_counter()
        
        try:
            # Rate limit check
            if not await self._check_rate_limit(control_data.type, video_id):
                raise HTTPException(status_code=429, detail="Rate limit exceeded")

            # Validate control data
            validation_result = await self.validate_control_data(control_data)
            if not validation_result[0]:
                CONTROL_ERRORS.labels(type=control_data.type, error='validation').inc()
                raise HTTPException(status_code=400, detail=validation_result[1])

            # Check cache for recent identical control
            cache_key = f"{video_id}:{control_data.type}:{hash(str(control_data.data))}"
            cached_result = await self._cache.get(cache_key)
            if cached_result:
                return cached_result

            # Process control through processor
            processed_control = await asyncio.to_thread(
                self._processor.process_control,
                control_data.dict()
            )

            # Store in database
            control_record = Control(
                generation_id=generation_id,
                video_id=video_id,
                type=control_data.type,
                data=processed_control
            )
            self._db.add(control_record)
            await self._db.commit()
            await self._db.refresh(control_record)

            # Calculate and track latency
            latency = time.perf_counter() - start_time
            CONTROL_LATENCY.observe(latency)

            # Track success
            CONTROL_SUCCESS.labels(type=control_data.type).inc()

            # Prepare result with metrics
            result = {
                'control_id': str(control_record.id),
                'processed_data': processed_control,
                'latency_ms': round(latency * 1000, 2),
                'success': True
            }

            # Cache result
            await self._cache.set(cache_key, result, ttl=5)  # 5 second cache

            # Log success with metrics
            logger.info(
                "control_processed",
                control_type=control_data.type,
                latency_ms=result['latency_ms'],
                video_id=str(video_id)
            )

            return result

        except Exception as e:
            # Track error
            CONTROL_ERRORS.labels(
                type=control_data.type,
                error=type(e).__name__
            ).inc()

            # Log error with context
            logger.error(
                "control_processing_failed",
                error=str(e),
                control_type=control_data.type,
                video_id=str(video_id)
            )

            raise HTTPException(
                status_code=500,
                detail=f"Control processing failed: {str(e)}"
            )

    async def get_control_history(
        self,
        video_id: UUID,
        limit: Optional[int] = 100,
        offset: Optional[int] = 0
    ) -> List[Dict[str, Any]]:
        """Retrieve paginated control history with performance data."""
        try:
            # Check cache
            cache_key = f"history:{video_id}:{limit}:{offset}"
            cached_history = await self._cache.get(cache_key)
            if cached_history:
                return cached_history

            # Query database with pagination
            query = (
                self._db.query(Control)
                .filter(Control.video_id == video_id)
                .order_by(Control.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            controls = await query.all()

            # Convert to dict with performance data
            history = []
            for control in controls:
                control_dict = control.to_dict()
                control_dict['performance'] = {
                    'latency_ms': CONTROL_LATENCY.observe(),
                    'success_rate': self._calculate_success_rate(control.type)
                }
                history.append(control_dict)

            # Cache results
            await self._cache.set(cache_key, history, ttl=30)  # 30 second cache

            return history

        except Exception as e:
            logger.error(
                "control_history_retrieval_failed",
                error=str(e),
                video_id=str(video_id)
            )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to retrieve control history: {str(e)}"
            )

    async def validate_control_data(
        self,
        control_data: ControlBase,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """Validate control data with security checks and rate limiting."""
        try:
            # Basic schema validation
            if not control_data.type or not control_data.data:
                return False, {"error": "Missing required fields"}

            # Type-specific validation
            validation_result = await self._processor.validate_control(
                control_data.dict()
            )
            if not validation_result:
                return False, {"error": "Invalid control data format"}

            # Security checks
            if not self._validate_security_constraints(control_data):
                return False, {"error": "Security validation failed"}

            # Rate limit check
            if context and not await self._check_rate_limit(
                control_data.type,
                context.get('video_id')
            ):
                return False, {"error": "Rate limit exceeded"}

            return True, {"status": "valid"}

        except Exception as e:
            logger.error(
                "control_validation_failed",
                error=str(e),
                control_type=control_data.type
            )
            return False, {"error": f"Validation failed: {str(e)}"}

    async def _check_rate_limit(self, control_type: str, video_id: UUID) -> bool:
        """Check rate limits for control operations."""
        cache_key = f"rate:{video_id}:{control_type}"
        current_count = await self._cache.get(cache_key) or 0
        
        if current_count >= self._rate_limits[control_type]['max_rate']:
            return False
            
        await self._cache.set(
            cache_key,
            current_count + 1,
            ttl=self._rate_limits[control_type]['window']
        )
        return True

    def _calculate_success_rate(self, control_type: str) -> float:
        """Calculate success rate for control type."""
        success_count = CONTROL_SUCCESS.labels(type=control_type)._value.get()
        error_count = sum(
            counter._value.get()
            for counter in CONTROL_ERRORS._metrics
            if counter._labelvalues[0] == control_type
        )
        total = success_count + error_count
        return (success_count / total) if total > 0 else 1.0

    def _validate_security_constraints(self, control_data: ControlBase) -> bool:
        """Validate security constraints for control data."""
        try:
            # Check for injection attempts
            if any(
                isinstance(v, str) and ('$' in v or ';' in v)
                for v in control_data.data.values()
            ):
                return False

            # Check value ranges
            if control_data.type == 'environment':
                if not all(
                    isinstance(v, (str, float)) and (
                        not isinstance(v, float) or -1 <= v <= 1
                    )
                    for v in control_data.data.values()
                ):
                    return False

            return True

        except Exception:
            return False