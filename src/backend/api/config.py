# External imports with versions
from pydantic import BaseSettings, Field  # pydantic ^2.0.0
from dotenv import load_dotenv  # python-dotenv ^1.0.0
from typing import Dict, Any, List, Optional
import os
import logging
from datetime import timedelta

# Internal imports
from ..core.config import Settings

# Global constants
API_VERSION = "v1"
API_PREFIX = "/api/v1"
WS_PREFIX = "/ws"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APISettings(BaseSettings):
    """API-specific settings configuration with FreeBSD optimizations"""
    
    title: str = Field(default="GameGen-X API")
    version: str = Field(default=API_VERSION)
    prefix: str = Field(default=API_PREFIX)
    workers: int = Field(default=4)  # Optimized for FreeBSD worker processes
    
    cors_origins: Dict[str, List[str]] = Field(default_factory=lambda: {
        "development": ["http://localhost:*"],
        "staging": ["https://*.staging.gamegen-x.com"],
        "production": ["https://*.gamegen-x.com"]
    })
    
    rate_limits: Dict[str, Dict[str, Any]] = Field(default_factory=lambda: {
        "generate": {
            "rate": 10,
            "per": 60,  # per minute
            "concurrent": 100  # max concurrent users
        },
        "control": {
            "rate": 60,
            "per": 60,
            "concurrent": 100
        },
        "status": {
            "rate": 120,
            "per": 60,
            "concurrent": 200
        }
    })
    
    freebsd_worker_config: Dict[str, Any] = Field(default_factory=lambda: {
        "accept_filter": "httpready",
        "somaxconn": 1024,
        "sendfile": True,
        "tcp_nopush": True,
        "tcp_nodelay": True,
        "keepalive_requests": 100,
        "worker_connections": 1000
    })
    
    performance_monitoring: Dict[str, Any] = Field(default_factory=lambda: {
        "request_latency_threshold": 0.1,  # 100ms
        "response_time_threshold": 0.05,  # 50ms
        "worker_memory_limit": 512 * 1024 * 1024,  # 512MB per worker
        "connection_timeout": 30,  # seconds
        "enable_metrics": True
    })

    def get_rate_limit(self, endpoint: str, client_id: str) -> Dict[str, Any]:
        """Returns rate limit for endpoint with concurrent user handling"""
        if endpoint not in self.rate_limits:
            return self.rate_limits["status"]  # Default rate limit
            
        limit_config = self.rate_limits[endpoint].copy()
        
        # Check concurrent user count against limits
        current_users = self._get_concurrent_users(endpoint)
        if current_users >= limit_config["concurrent"]:
            limit_config["rate"] = limit_config["rate"] // 2  # Reduce rate under high load
            
        return {
            "client_id": client_id,
            "endpoint": endpoint,
            "rate": limit_config["rate"],
            "per": limit_config["per"],
            "remaining": self._get_remaining_requests(endpoint, client_id)
        }

    def _get_concurrent_users(self, endpoint: str) -> int:
        """Helper method to get current concurrent users for an endpoint"""
        # Implementation would track actual concurrent connections
        return 0  # Placeholder

    def _get_remaining_requests(self, endpoint: str, client_id: str) -> int:
        """Helper method to get remaining requests for a client"""
        # Implementation would track actual request counts
        return self.rate_limits[endpoint]["rate"]  # Placeholder

class SecuritySettings(BaseSettings):
    """Enhanced security and authentication settings with RS256 support"""
    
    secret_key: str = Field(..., env="API_SECRET_KEY")
    public_key: str = Field(..., env="API_PUBLIC_KEY")
    private_key: str = Field(..., env="API_PRIVATE_KEY")
    algorithm: str = Field(default="RS256")
    token_expire_minutes: int = Field(default=60)  # 1 hour expiry
    
    allowed_roles: List[str] = Field(default_factory=lambda: [
        "admin",
        "developer",
        "user"
    ])
    
    role_permissions: Dict[str, List[str]] = Field(default_factory=lambda: {
        "admin": ["generate", "control", "train", "configure"],
        "developer": ["generate", "control"],
        "user": ["generate", "control"]
    })
    
    token_rotation_policy: Dict[str, Any] = Field(default_factory=lambda: {
        "enabled": True,
        "rotation_interval": 24 * 60,  # 24 hours
        "grace_period": 60,  # 1 hour
        "max_active_tokens": 2
    })

class WebSocketSettings(BaseSettings):
    """WebSocket configuration optimized for FreeBSD"""
    
    ping_interval: int = Field(default=20)  # seconds
    ping_timeout: int = Field(default=10)  # seconds
    close_timeout: int = Field(default=5)  # seconds
    max_message_size: int = Field(default=1024 * 1024)  # 1MB
    
    freebsd_socket_opts: Dict[str, Any] = Field(default_factory=lambda: {
        "SO_REUSEPORT": True,
        "SO_KEEPALIVE": True,
        "TCP_KEEPIDLE": 60,
        "TCP_KEEPINTVL": 10,
        "TCP_KEEPCNT": 6,
        "SO_SNDBUF": 32768,
        "SO_RCVBUF": 32768
    })
    
    connection_pool_config: Dict[str, Any] = Field(default_factory=lambda: {
        "max_size": 1000,
        "max_lifetime": 3600,  # 1 hour
        "idle_timeout": 300,  # 5 minutes
        "enable_monitoring": True
    })
    
    performance_tuning: Dict[str, Any] = Field(default_factory=lambda: {
        "frame_compression": "high",
        "batch_size": 10,
        "buffer_size": 1024 * 1024,  # 1MB buffer
        "enable_zero_copy": True
    })

def load_api_settings() -> APISettings:
    """Loads and validates API settings with FreeBSD compatibility checks"""
    try:
        # Load core settings first
        core_settings = Settings()
        
        # Initialize API settings
        api_settings = APISettings()
        
        # Validate FreeBSD compatibility
        if core_settings.environment == "production":
            if not all(opt in os.sysconf_names for opt in api_settings.freebsd_worker_config):
                raise ValueError("FreeBSD worker configuration not supported")
        
        # Configure performance monitoring based on environment
        if core_settings.environment != "development":
            api_settings.performance_monitoring["enable_metrics"] = True
            api_settings.performance_monitoring["request_latency_threshold"] = 0.05  # Stricter in prod
        
        return api_settings
    except Exception as e:
        logger.error(f"Failed to load API settings: {str(e)}")
        raise

# Create singleton instances
api_settings = load_api_settings()
security_settings = SecuritySettings()
websocket_settings = WebSocketSettings()

# Export settings instances
__all__ = [
    "api_settings",
    "security_settings",
    "websocket_settings",
    "API_VERSION",
    "API_PREFIX",
    "WS_PREFIX"
]