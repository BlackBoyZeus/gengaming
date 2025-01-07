# External imports with versions
from fastapi import Request, HTTPException, Depends  # fastapi ^0.95.0
from sqlalchemy.orm import Session  # sqlalchemy ^1.4.41
from functools import lru_cache  # ^3.9
from typing import Optional, Dict, Any
from contextlib import contextmanager
import time
import prometheus_client  # prometheus_client ^0.16.0
import redis  # redis ^4.5.0

# Internal imports
from api.config import api_settings
from api.security.jwt import decode_token, get_token_payload
from services.cache import FrameCache
from db.session import db_session
from core.metrics import track_generation_latency, CONTROL_RESPONSE
from core.exceptions import ValidationError

# Performance metrics
REQUEST_LATENCY = prometheus_client.Histogram(
    'api_request_latency_seconds',
    'Request latency in seconds',
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
)

ACTIVE_SESSIONS = prometheus_client.Gauge(
    'active_database_sessions',
    'Number of active database sessions'
)

CACHE_HITS = prometheus_client.Counter(
    'cache_hits_total',
    'Total number of cache hits'
)

class APIException(HTTPException):
    """Enhanced exception handling for API errors with detailed tracking."""
    
    def __init__(
        self,
        detail: str,
        status_code: int = 400,
        headers: Optional[Dict[str, str]] = None,
        error_code: Optional[str] = None
    ):
        super().__init__(status_code=status_code, detail=detail, headers=headers)
        self.error_code = error_code
        self.timestamp = time.time()
        
        # Track error metrics
        prometheus_client.Counter(
            'api_errors_total',
            'Total API errors',
            ['error_code', 'status_code']
        ).labels(error_code=error_code, status_code=status_code).inc()

@contextmanager
def get_db():
    """Optimized database session dependency with connection pooling and monitoring."""
    session_start = time.time()
    session = db_session()
    ACTIVE_SESSIONS.inc()
    
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        raise APIException(
            detail=str(e),
            status_code=500,
            error_code="DB_ERROR"
        )
    finally:
        session.close()
        ACTIVE_SESSIONS.dec()
        REQUEST_LATENCY.observe(time.time() - session_start)

@lru_cache()
def get_cache() -> FrameCache:
    """High-performance frame cache dependency with Redis."""
    try:
        redis_client = redis.Redis(
            host=api_settings.redis_host,
            port=api_settings.redis_port,
            db=0,
            socket_timeout=1.0,
            socket_keepalive=True,
            socket_keepalive_options={
                'tcp_keepidle': 30,
                'tcp_keepintvl': 5,
                'tcp_keepcnt': 3
            }
        )
        return FrameCache(redis_client)
    except Exception as e:
        raise APIException(
            detail=f"Failed to initialize cache: {str(e)}",
            status_code=500,
            error_code="CACHE_ERROR"
        )

async def get_current_user(request: Request, db: Session = Depends(get_db)):
    """Secure user authentication with comprehensive validation."""
    try:
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            raise APIException(
                detail="Missing authorization header",
                status_code=401,
                error_code="AUTH_MISSING"
            )

        scheme, token = auth_header.split()
        if scheme.lower() != "bearer":
            raise APIException(
                detail="Invalid authentication scheme",
                status_code=401,
                error_code="AUTH_SCHEME"
            )

        # Validate JWT token
        try:
            payload = decode_token(token)
            user_data = get_token_payload(token)
        except ValidationError as e:
            raise APIException(
                detail=str(e),
                status_code=401,
                error_code="TOKEN_INVALID"
            )

        # Get user from database
        user = db.query(User).filter(User.id == user_data["user_id"]).first()
        if not user or not user.is_active:
            raise APIException(
                detail="User not found or inactive",
                status_code=401,
                error_code="USER_INVALID"
            )

        return user

    except APIException:
        raise
    except Exception as e:
        raise APIException(
            detail=f"Authentication failed: {str(e)}",
            status_code=500,
            error_code="AUTH_ERROR"
        )

async def check_rate_limit(request: Request):
    """Advanced rate limiting with dynamic thresholds."""
    client_ip = request.client.host
    endpoint = request.url.path
    
    try:
        # Get rate limit configuration
        limit_config = api_settings.get_rate_limit(endpoint, client_ip)
        
        # Check rate with Redis
        redis_client = get_cache().redis_client
        key = f"ratelimit:{client_ip}:{endpoint}"
        current = redis_client.incr(key)
        
        # Set expiry on first request
        if current == 1:
            redis_client.expire(key, limit_config["per"])
        
        if current > limit_config["rate"]:
            raise APIException(
                detail="Rate limit exceeded",
                status_code=429,
                headers={"Retry-After": str(limit_config["per"])},
                error_code="RATE_LIMIT"
            )
            
        # Track metrics
        CONTROL_RESPONSE.observe(time.time())
        return True

    except redis.RedisError as e:
        # Fail open if Redis is unavailable
        return True
    except APIException:
        raise
    except Exception as e:
        raise APIException(
            detail=f"Rate limit check failed: {str(e)}",
            status_code=500,
            error_code="RATE_ERROR"
        )