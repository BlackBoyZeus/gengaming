# External imports with versions
from fastapi import FastAPI, Request, Response  # ^0.95.0
from starlette.middleware.cors import CORSMiddleware  # ^0.26.0
from starlette.middleware.base import BaseHTTPMiddleware  # ^0.26.0
import time  # ^3.9
import pyucl  # ^0.9.0
from typing import Callable, Dict, List, Optional

# Internal imports
from core.logging import get_logger
from core.metrics import track_control_response, track_concurrent_users
from core.exceptions import GameGenBaseException

# Initialize logger
logger = get_logger(__name__)

# Global constants
CORS_ORIGINS = ["*"]  # Configured for development, restrict in production
MAX_CONCURRENT_USERS = 100
JAIL_RESOURCE_THRESHOLD = 0.9
CIRCUIT_BREAKER_THRESHOLD = 0.95
ERROR_RECOVERY_ATTEMPTS = 3

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging API requests and responses with ELK stack integration."""
    
    def __init__(self, app: FastAPI, buffer_size: int = 100):
        super().__init__(app)
        self.logger = get_logger("request_logger")
        self.elk_buffer = []
        self.buffer_size = buffer_size

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        correlation_id = request.headers.get("X-Correlation-ID", str(time.time()))
        
        # Log request
        self.logger.info(
            "Incoming request",
            method=request.method,
            url=str(request.url),
            correlation_id=correlation_id,
            client_host=request.client.host
        )

        try:
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log response
            self.logger.info(
                "Request completed",
                correlation_id=correlation_id,
                status_code=response.status_code,
                process_time=process_time
            )
            
            # Add timing header
            response.headers["X-Process-Time"] = str(process_time)
            return response
            
        except Exception as e:
            self.logger.error(
                "Request failed",
                correlation_id=correlation_id,
                error=str(e),
                exc_info=True
            )
            raise

class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting API metrics with FreeBSD jail monitoring."""
    
    def __init__(self, app: FastAPI):
        super().__init__(app)
        self.active_requests = 0
        self.resource_metrics = {}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Track concurrent users
        self.active_requests += 1
        track_concurrent_users(self.active_requests)
        
        # Monitor jail resources
        jail_name = request.headers.get("X-Jail-Name", "default")
        start_time = time.time()
        
        try:
            # Check resource thresholds
            if self._check_jail_resources(jail_name):
                response = await call_next(request)
                process_time = (time.time() - start_time) * 1000  # Convert to ms
                
                # Track response time
                track_control_response(process_time)
                
                return response
            else:
                return Response(
                    content="Service temporarily unavailable due to resource constraints",
                    status_code=503
                )
                
        finally:
            self.active_requests -= 1
            track_concurrent_users(self.active_requests)

    def _check_jail_resources(self, jail_name: str) -> bool:
        """Check FreeBSD jail resource utilization."""
        try:
            # Get jail resource usage using pyucl
            jail_config = pyucl.load_file(f"/etc/jail.conf.d/{jail_name}.conf")
            current_usage = {
                "cpu": float(jail_config.get("cpu.usage", 0)),
                "memory": float(jail_config.get("memory.usage", 0))
            }
            
            # Check against thresholds
            return all(usage < JAIL_RESOURCE_THRESHOLD for usage in current_usage.values())
            
        except Exception as e:
            logger.error(f"Failed to check jail resources: {str(e)}")
            return True  # Default to allowing request on monitoring failure

class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for handling API errors with circuit breaker pattern."""
    
    def __init__(self, app: FastAPI):
        super().__init__(app)
        self.logger = get_logger("error_handler")
        self.circuit_breaker = {
            "failures": 0,
            "last_failure": 0,
            "state": "closed"  # closed, open, half-open
        }
        self.recovery_manager = {
            "attempts": 0,
            "backoff_factor": 2
        }

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Check circuit breaker state
        if self._is_circuit_open():
            return Response(
                content="Service temporarily unavailable",
                status_code=503
            )

        try:
            response = await call_next(request)
            
            # Reset circuit breaker on success
            if self.circuit_breaker["state"] != "closed":
                self._reset_circuit()
                
            return response
            
        except GameGenBaseException as e:
            # Handle custom exceptions
            self.logger.error(
                "Custom exception occurred",
                error_code=e.error_code,
                recovery_hints=e.recovery_hints
            )
            
            return self._handle_error(e)
            
        except Exception as e:
            # Handle unexpected exceptions
            self.logger.error(
                "Unexpected error occurred",
                error=str(e),
                exc_info=True
            )
            
            self._update_circuit_state()
            return Response(
                content="Internal server error",
                status_code=500
            )

    def _is_circuit_open(self) -> bool:
        """Check if circuit breaker is open."""
        if self.circuit_breaker["state"] == "open":
            # Check if cooling period has passed
            if time.time() - self.circuit_breaker["last_failure"] > 60:
                self.circuit_breaker["state"] = "half-open"
                return False
            return True
        return False

    def _update_circuit_state(self):
        """Update circuit breaker state on failure."""
        self.circuit_breaker["failures"] += 1
        self.circuit_breaker["last_failure"] = time.time()
        
        if self.circuit_breaker["failures"] >= CIRCUIT_BREAKER_THRESHOLD:
            self.circuit_breaker["state"] = "open"

    def _reset_circuit(self):
        """Reset circuit breaker state."""
        self.circuit_breaker["failures"] = 0
        self.circuit_breaker["state"] = "closed"
        self.recovery_manager["attempts"] = 0

    def _handle_error(self, error: GameGenBaseException) -> Response:
        """Handle custom exceptions with recovery attempts."""
        if error.recovery_hints.get("retry_recommended", False):
            if self.recovery_manager["attempts"] < ERROR_RECOVERY_ATTEMPTS:
                self.recovery_manager["attempts"] += 1
                return Response(
                    content="Retrying operation",
                    status_code=503,
                    headers={"Retry-After": str(self.recovery_manager["backoff_factor"] ** self.recovery_manager["attempts"])}
                )
        
        return Response(
            content=error.message,
            status_code=500
        )

def setup_middleware(app: FastAPI) -> None:
    """Configure and attach middleware to FastAPI application with FreeBSD integration."""
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Process-Time"]
    )
    
    # Add custom middleware
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(MetricsMiddleware)
    app.add_middleware(ErrorHandlingMiddleware)
    
    logger.info("Middleware stack configured successfully")